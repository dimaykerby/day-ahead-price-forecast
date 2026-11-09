"""
Monte Carlo battery arbitrage analysis on top of day-ahead price forecasts.

Workflow:
1) Build a point forecast and residual history from the existing forecasting stack.
2) Simulate stochastic price paths around the point forecast (bootstrap or GARCH-vol).
3) Run a simple greedy daily dispatch for each path.
4) Produce annualized revenue distribution, sensitivity table, finance KPIs, and a short memo.

Example:
    python monte_carlo_arbitrage.py \
        --target italy \
        --model lgbm \
        --n-sims 1000 \
        --horizon-hours 96 \
        --error-model bootstrap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from src.features import build_features, load_data
from src.risk import tail_stats
from src.models import GPForecaster, LGBMForecaster, LSTMForecaster, TimesFMForecaster


@dataclass
class BatteryConfig:
    power_mw: float = 10.0
    energy_mwh: float = 20.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    round_trip_efficiency: float = 0.88
    availability: float = 0.95


@dataclass
class FinanceConfig:
    capex_eur: float = 8_000_000.0
    fixed_opex_eur_per_year: float = 120_000.0
    variable_opex_eur_per_mwh: float = 0.0
    discount_rate: float = 0.08
    years: int = 10
    debt_share: float = 0.60
    debt_rate: float = 0.05
    debt_tenor_years: int = 8
    tax_rate: float = 0.24


def _load_model(model_name: str, model_path: Path):
    if model_name == "lgbm":
        return LGBMForecaster.load(str(model_path))
    if model_name == "gp":
        return GPForecaster.load(str(model_path))
    if model_name == "lstm":
        return LSTMForecaster.load(str(model_path))
    if model_name == "timesfm":
        return TimesFMForecaster(device="cpu", batch_size=8)
    raise ValueError(f"Unknown model: {model_name}")


def _get_point_forecast_and_residuals(
    data_path: str,
    models_dir: str,
    target: str,
    model_name: str,
    horizon_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_data(data_path)
    X, y = build_features(df, target=target)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_path = Path(models_dir) / f"{model_name}_{target}.joblib"
    if model_name != "timesfm" and not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train.py first for this target/model."
        )
    model = _load_model(model_name, model_path)

    if model_name == "gp":
        preds_test, _ = model.predict(X_test, return_std=True)
    elif model_name == "lstm":
        preds_test = model.predict(X_test, context=X_train)
    elif model_name == "timesfm":
        preds_test = model.predict(y_train.values.astype(np.float32), horizon=horizon_hours, return_quantiles=False)
        preds_test = np.asarray(preds_test, dtype=np.float32).reshape(-1)
    else:
        preds_test = model.predict(X_test)

    residuals_test = y_test.values - preds_test

    if horizon_hours > len(preds_test):
        raise ValueError(
            f"horizon_hours={horizon_hours} exceeds available test horizon ({len(preds_test)})."
        )

    point_forecast = preds_test[-horizon_hours:]
    residuals_history = residuals_test[~np.isnan(residuals_test)]
    return point_forecast, residuals_history


def _simulate_residual_paths_bootstrap(residuals: np.ndarray, n_sims: int, horizon: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.choice(residuals, size=(n_sims, horizon), replace=True)
    return draws


def _simulate_residual_paths_garch_vol(residuals: np.ndarray, n_sims: int, horizon: int, seed: int) -> np.ndarray:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("arch is not installed. Install with: pip install arch") from exc

    resid = pd.Series(residuals).dropna().values
    resid = resid - np.mean(resid)
    am = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    fit = am.fit(disp="off")
    var_forecast = fit.forecast(horizon=horizon).variance.values[-1]
    sigma_forecast = np.sqrt(np.maximum(var_forecast, 1e-12))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_sims, horizon))
    return z * sigma_forecast.reshape(1, -1)


def simulate_price_paths(
    point_forecast: np.ndarray,
    residuals_history: np.ndarray,
    n_sims: int,
    error_model: str,
    seed: int,
) -> np.ndarray:
    horizon = len(point_forecast)
    if error_model == "bootstrap":
        shocks = _simulate_residual_paths_bootstrap(residuals_history, n_sims, horizon, seed)
    elif error_model == "garch":
        shocks = _simulate_residual_paths_garch_vol(residuals_history, n_sims, horizon, seed)
    else:
        raise ValueError(f"Unsupported error_model: {error_model}")
    return point_forecast.reshape(1, -1) + shocks


def _compress_spreads(prices: np.ndarray, compression: float) -> np.ndarray:
    if compression <= 0:
        return prices
    daily_mean = prices.mean(axis=1, keepdims=True)
    return daily_mean + (prices - daily_mean) * (1.0 - compression)


def greedy_daily_dispatch_revenue(prices_24h: np.ndarray, cfg: BatteryConfig) -> tuple[float, float, float]:
    """
    Returns tuple: (revenue_eur, charged_mwh_from_grid, discharged_mwh_to_grid)
    """
    hours = len(prices_24h)
    if hours != 24:
        raise ValueError("Daily greedy dispatch expects exactly 24 hourly prices")

    eta_c = np.sqrt(cfg.round_trip_efficiency)
    eta_d = np.sqrt(cfg.round_trip_efficiency)

    soc_min_mwh = cfg.soc_min * cfg.energy_mwh
    soc_max_mwh = cfg.soc_max * cfg.energy_mwh
    usable_mwh = max(soc_max_mwh - soc_min_mwh, 0.0)

    power = cfg.power_mw * cfg.availability
    if power <= 0 or usable_mwh <= 0:
        return 0.0, 0.0, 0.0

    n_charge_hours = int(np.ceil(usable_mwh / (power * eta_c)))
    n_discharge_hours = int(np.ceil(usable_mwh / power))
    n_charge_hours = int(np.clip(n_charge_hours, 1, 24))
    n_discharge_hours = int(np.clip(n_discharge_hours, 1, 24))

    charge_hours = set(np.argsort(prices_24h)[:n_charge_hours].tolist())
    discharge_hours = set(np.argsort(prices_24h)[-n_discharge_hours:].tolist())

    soc = soc_min_mwh
    charged = 0.0
    discharged = 0.0
    cash = 0.0

    for h, p in enumerate(prices_24h):
        if h in charge_hours and soc < soc_max_mwh:
            max_grid_charge = min(power, (soc_max_mwh - soc) / eta_c)
            if max_grid_charge > 0:
                soc += max_grid_charge * eta_c
                charged += max_grid_charge
                cash -= max_grid_charge * p

        if h in discharge_hours and soc > soc_min_mwh:
            max_grid_discharge = min(power, (soc - soc_min_mwh) * eta_d)
            if max_grid_discharge > 0:
                soc -= max_grid_discharge / eta_d
                discharged += max_grid_discharge
                cash += max_grid_discharge * p

    return cash, charged, discharged


def compute_annual_revenue_distribution(
    price_paths: np.ndarray,
    cfg: BatteryConfig,
    spread_compression: float = 0.0,
) -> pd.DataFrame:
    """
    Convert each scenario path into annualized revenues.
    price_paths shape: (n_sims, horizon_hours), horizon multiple of 24.
    """
    if price_paths.shape[1] % 24 != 0:
        raise ValueError("horizon_hours must be a multiple of 24")

    adj_paths = _compress_spreads(price_paths, spread_compression)
    n_days = adj_paths.shape[1] // 24

    out = []
    for i in range(adj_paths.shape[0]):
        daily_revenues = []
        charged_total = 0.0
        discharged_total = 0.0
        for d in range(n_days):
            day_prices = adj_paths[i, d * 24 : (d + 1) * 24]
            rev, ch, dis = greedy_daily_dispatch_revenue(day_prices, cfg)
            daily_revenues.append(rev)
            charged_total += ch
            discharged_total += dis

        daily_avg = float(np.mean(daily_revenues))
        annual_revenue = daily_avg * 365.0
        out.append(
            {
                "scenario": i,
                "daily_revenue_avg_eur": daily_avg,
                "annual_revenue_eur": annual_revenue,
                "charged_mwh_total": charged_total,
                "discharged_mwh_total": discharged_total,
            }
        )
    return pd.DataFrame(out)


def _p_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def run_sensitivity(price_paths: np.ndarray, base_cfg: BatteryConfig) -> pd.DataFrame:
    cases = []

    base_df = compute_annual_revenue_distribution(price_paths, base_cfg, spread_compression=0.0)
    s = _p_stats(base_df["annual_revenue_eur"].values)
    cases.append({"case": "base", **s})

    eff_cfg = BatteryConfig(**{**base_cfg.__dict__, "round_trip_efficiency": max(0.01, base_cfg.round_trip_efficiency * 0.95)})
    eff_df = compute_annual_revenue_distribution(price_paths, eff_cfg, spread_compression=0.0)
    s = _p_stats(eff_df["annual_revenue_eur"].values)
    cases.append({"case": "efficiency_minus_5pct", **s})

    spread_df = compute_annual_revenue_distribution(price_paths, base_cfg, spread_compression=0.20)
    s = _p_stats(spread_df["annual_revenue_eur"].values)
    cases.append({"case": "spread_compression_20pct", **s})

    avail90_cfg = BatteryConfig(**{**base_cfg.__dict__, "availability": 0.90})
    avail90_df = compute_annual_revenue_distribution(price_paths, avail90_cfg, spread_compression=0.0)
    s = _p_stats(avail90_df["annual_revenue_eur"].values)
    cases.append({"case": "availability_90pct", **s})

    avail95_cfg = BatteryConfig(**{**base_cfg.__dict__, "availability": 0.95})
    avail95_df = compute_annual_revenue_distribution(price_paths, avail95_cfg, spread_compression=0.0)
    s = _p_stats(avail95_df["annual_revenue_eur"].values)
    cases.append({"case": "availability_95pct", **s})

    return pd.DataFrame(cases)


def _annuity_payment(principal: float, rate: float, tenor: int) -> float:
    if principal <= 0:
        return 0.0
    if rate <= 0:
        return principal / tenor
    factor = (rate * (1 + rate) ** tenor) / ((1 + rate) ** tenor - 1)
    return principal * factor


def _irr(cashflows: np.ndarray) -> float:
    try:
        import numpy_financial as npf

        return float(npf.irr(cashflows))
    except Exception:
        grid = np.linspace(-0.99, 1.5, 4000)
        npv_vals = [np.sum(cashflows / ((1 + r) ** np.arange(len(cashflows)))) for r in grid]
        sign = np.sign(npv_vals)
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) == 0:
            return float("nan")
        return float(grid[idx[0]])


def _npv(cashflows: np.ndarray, discount_rate: float) -> float:
    t = np.arange(len(cashflows))
    return float(np.sum(cashflows / ((1 + discount_rate) ** t)))


def finance_kpis(
    annual_revenue: float,
    cfg: FinanceConfig,
) -> dict[str, float]:
    debt = cfg.capex_eur * cfg.debt_share
    equity = cfg.capex_eur - debt
    debt_service = _annuity_payment(debt, cfg.debt_rate, cfg.debt_tenor_years)

    cashflows_project = [-cfg.capex_eur]
    cashflows_equity = [-equity]

    for y in range(1, cfg.years + 1):
        revenue = annual_revenue
        variable_opex = cfg.variable_opex_eur_per_mwh * 0.0
        ebit = revenue - cfg.fixed_opex_eur_per_year - variable_opex
        tax = max(0.0, ebit) * cfg.tax_rate
        after_tax_operating_cf = ebit - tax

        ds = debt_service if y <= cfg.debt_tenor_years else 0.0
        cashflows_project.append(after_tax_operating_cf)
        cashflows_equity.append(after_tax_operating_cf - ds)

    cf_project = np.array(cashflows_project, dtype=float)
    cf_equity = np.array(cashflows_equity, dtype=float)

    return {
        "project_npv_eur": _npv(cf_project, cfg.discount_rate),
        "project_irr": _irr(cf_project),
        "equity_npv_eur": _npv(cf_equity, cfg.discount_rate),
        "equity_irr": _irr(cf_equity),
    }


def downside_risk(annual_revenues: np.ndarray, cfg: FinanceConfig) -> dict[str, float]:
    """Tail metrics and the probability of failing to cover fixed opex plus debt service."""
    tail = tail_stats(annual_revenues, 0.05)
    debt_service = _annuity_payment(cfg.capex_eur * cfg.debt_share, cfg.debt_rate, cfg.debt_tenor_years)
    breakeven = cfg.fixed_opex_eur_per_year + debt_service
    return {
        "p05_revenue_eur": tail["quantile"],
        "es05_revenue_eur": tail["tail_mean"],
        "debt_service_breakeven_eur": breakeven,
        "prob_below_breakeven": float(np.mean(annual_revenues < breakeven)),
    }


def _write_memo(
    out_dir: Path,
    target: str,
    model_name: str,
    n_sims: int,
    horizon_hours: int,
    revenue_stats: dict[str, float],
    sensitivity_df: pd.DataFrame,
    p50_finance: dict[str, float],
    downside: dict[str, float],
    p10_finance: dict[str, float],
):
    try:
        sensitivity_table = sensitivity_df.to_markdown(index=False)
    except ImportError:
        sensitivity_table = sensitivity_df.to_string(index=False)

    memo = f"""# BESS Arbitrage Monte Carlo Memo

## Business question
Given a day-ahead electricity price forecast for {target.title()}, what is the distribution of annual battery arbitrage revenue once forecast error and price uncertainty are included?

## Method
- Point forecast model: {model_name.upper()}
- Scenario count: {n_sims}
- Horizon per scenario: {horizon_hours} hours
- Price uncertainty: residual-based simulation around the point forecast
- Dispatch: daily greedy arbitrage with SoC limits and round-trip efficiency
- Revenue metric: daily gross arbitrage spread annualized to 365 days

## Core results (annual revenue)
- P10: {revenue_stats['p10']:,.0f} EUR
- P50: {revenue_stats['p50']:,.0f} EUR
- P90: {revenue_stats['p90']:,.0f} EUR
- Mean: {revenue_stats['mean']:,.0f} EUR

## Sensitivities
{sensitivity_table}

## Finance snapshot at P50 annual revenue
- Project NPV: {p50_finance['project_npv_eur']:,.0f} EUR
- Project IRR: {100 * p50_finance['project_irr']:.2f}%
- Equity NPV: {p50_finance['equity_npv_eur']:,.0f} EUR
- Equity IRR: {100 * p50_finance['equity_irr']:.2f}%

## Downside risk
- P5 annual revenue: {downside['p05_revenue_eur']:,.0f} EUR
- Expected shortfall (mean of worst 5% scenarios): {downside['es05_revenue_eur']:,.0f} EUR
- Revenue needed to cover fixed opex and debt service: {downside['debt_service_breakeven_eur']:,.0f} EUR
- Probability of falling short of that level: {100 * downside['prob_below_breakeven']:.1f}%
- Project NPV at P10 revenue: {p10_finance['project_npv_eur']:,.0f} EUR

## Plain-language summary
We simulated {n_sims} plausible market futures around the forecast and re-ran battery dispatch in each one.
This gives a distribution of yearly earnings, not just one number.
The P50 value is the median commercial case, while P90 is the more conservative downside view used for risk-aware investment decisions.
"""
    (out_dir / "investment_memo.md").write_text(memo)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo BESS arbitrage on top of day-ahead price forecasts")
    parser.add_argument("--data", default="data/electricity_dah_prices.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--target", default="italy")
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "gp", "lstm", "timesfm"])
    parser.add_argument("--error-model", default="bootstrap", choices=["bootstrap", "garch"])
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--horizon-hours", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-mw", type=float, default=10.0)
    parser.add_argument("--energy-mwh", type=float, default=20.0)
    parser.add_argument("--soc-min", type=float, default=0.10)
    parser.add_argument("--soc-max", type=float, default=0.90)
    parser.add_argument("--rte", type=float, default=0.88)
    parser.add_argument("--availability", type=float, default=0.95)
    parser.add_argument("--capex-eur", type=float, default=8_000_000.0)
    parser.add_argument("--fixed-opex-eur", type=float, default=120_000.0)
    parser.add_argument("--discount-rate", type=float, default=0.08)
    parser.add_argument("--debt-share", type=float, default=0.60)
    parser.add_argument("--debt-rate", type=float, default=0.05)
    parser.add_argument("--debt-tenor", type=int, default=8)
    parser.add_argument("--tax-rate", type=float, default=0.24)
    parser.add_argument("--finance-years", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if args.horizon_hours % 24 != 0:
        raise ValueError("--horizon-hours must be a multiple of 24")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    battery_cfg = BatteryConfig(
        power_mw=args.power_mw,
        energy_mwh=args.energy_mwh,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        round_trip_efficiency=args.rte,
        availability=args.availability,
    )
    finance_cfg = FinanceConfig(
        capex_eur=args.capex_eur,
        fixed_opex_eur_per_year=args.fixed_opex_eur,
        discount_rate=args.discount_rate,
        debt_share=args.debt_share,
        debt_rate=args.debt_rate,
        debt_tenor_years=args.debt_tenor,
        tax_rate=args.tax_rate,
        years=args.finance_years,
    )

    point_forecast, residuals_history = _get_point_forecast_and_residuals(
        data_path=args.data,
        models_dir=args.models_dir,
        target=args.target,
        model_name=args.model,
        horizon_hours=args.horizon_hours,
    )

    price_paths = simulate_price_paths(
        point_forecast=point_forecast,
        residuals_history=residuals_history,
        n_sims=args.n_sims,
        error_model=args.error_model,
        seed=args.seed,
    )

    dist_df = compute_annual_revenue_distribution(price_paths, battery_cfg)
    dist_path = out_dir / "annual_revenue_distribution.csv"
    dist_df.to_csv(dist_path, index=False)

    revenue_stats = _p_stats(dist_df["annual_revenue_eur"].values)
    sensitivity_df = run_sensitivity(price_paths, battery_cfg)
    sens_path = out_dir / "sensitivity_table.csv"
    sensitivity_df.to_csv(sens_path, index=False)

    p50_finance = finance_kpis(revenue_stats["p50"], finance_cfg)
    finance_path = out_dir / "finance_p50.csv"
    pd.DataFrame([p50_finance]).to_csv(finance_path, index=False)

    downside = downside_risk(dist_df["annual_revenue_eur"].values, finance_cfg)
    pd.DataFrame([downside]).to_csv(out_dir / "downside_risk.csv", index=False)
    p10_finance = finance_kpis(revenue_stats["p10"], finance_cfg)

    fig = px.histogram(
        dist_df,
        x="annual_revenue_eur",
        nbins=40,
        title="Annual Arbitrage Revenue Distribution",
        labels={"annual_revenue_eur": "Annual revenue (EUR)"},
    )
    for label, value, color in [
        ("P10", revenue_stats["p10"], "#d62728"),
        ("P50", revenue_stats["p50"], "#1f77b4"),
        ("P90", revenue_stats["p90"], "#2ca02c"),
    ]:
        fig.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=label)
    fig.write_html(str(out_dir / "annual_revenue_histogram.html"), include_plotlyjs="cdn")

    _write_memo(
        out_dir=out_dir,
        target=args.target,
        model_name=args.model,
        n_sims=args.n_sims,
        horizon_hours=args.horizon_hours,
        revenue_stats=revenue_stats,
        sensitivity_df=sensitivity_df,
        p50_finance=p50_finance,
        downside=downside,
        p10_finance=p10_finance,
    )

    print("Monte Carlo arbitrage analysis complete")
    print(f"Results folder: {out_dir.resolve()}")
    print(
        "Annual revenue stats (EUR): "
        f"P10={revenue_stats['p10']:.0f}, "
        f"P50={revenue_stats['p50']:.0f}, "
        f"P90={revenue_stats['p90']:.0f}, "
        f"Mean={revenue_stats['mean']:.0f}"
    )
    print(
        "P50 finance snapshot: "
        f"Project NPV={p50_finance['project_npv_eur']:.0f}, "
        f"Project IRR={100 * p50_finance['project_irr']:.2f}%, "
        f"Equity NPV={p50_finance['equity_npv_eur']:.0f}, "
        f"Equity IRR={100 * p50_finance['equity_irr']:.2f}%"
    )


if __name__ == "__main__":
    main()
