"""
Market-risk analytics for multi-market energy prices.

Covers volatility (rolling, EWMA), correlations (static, rolling, regime-conditional),
portfolio VaR / Expected Shortfall with risk attribution, shock propagation
(stress tests) and correlated Monte Carlo simulation of portfolio P&L.

Convention: portfolio functions work on daily *price changes* (EUR/MWh), so a
position of `w` MWh/day in a market gives a daily P&L of `w * price_change` EUR.
Negative / zero prices make log returns undefined, hence "diff" is the default.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # power and gas trade every calendar day


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly prices to the daily baseload average."""
    return prices.resample("D").mean()


def compute_returns(prices: pd.DataFrame, kind: str = "diff") -> pd.DataFrame:
    """Price changes ("diff", EUR/MWh) or log returns ("log"; non-positive prices -> NaN)."""
    if kind == "diff":
        out = prices.diff()
    elif kind == "log":
        out = np.log(prices.where(prices > 0)).diff()
    else:
        raise ValueError(f"Unsupported return kind: {kind}")
    return out.dropna(how="all")


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def rolling_volatility(returns: pd.DataFrame, window: int, annualize: bool = False) -> pd.DataFrame:
    vol = returns.rolling(window, min_periods=window // 2).std()
    return vol * np.sqrt(DAYS_PER_YEAR) if annualize else vol


def ewma_volatility(returns: pd.DataFrame, lam: float = 0.94, annualize: bool = False) -> pd.DataFrame:
    """RiskMetrics zero-mean EWMA volatility."""
    vol = np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False).mean())
    return vol * np.sqrt(DAYS_PER_YEAR) if annualize else vol


def volatility_summary(returns: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    daily = returns.std()
    return pd.DataFrame(
        {
            "daily_vol": daily,
            "annualized_vol": daily * np.sqrt(DAYS_PER_YEAR) if annualize else np.nan,
            "skew": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "worst_move": returns.min(),
            "best_move": returns.max(),
        }
    )


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns.corr(method=method)


def rolling_correlation(returns: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    return returns[a].rolling(window, min_periods=window // 2).corr(returns[b])


def regime_correlation(
    returns: pd.DataFrame,
    reference: str,
    vol_window: int = 7,
    quantile: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correlation matrices in calm vs stressed regimes, defined by the reference market's
    rolling volatility: stressed = top (1 - quantile) tail, calm = bottom (1 - quantile) tail.
    """
    vol = returns[reference].rolling(vol_window, min_periods=vol_window // 2).std()
    stressed = vol >= vol.quantile(quantile)
    calm = vol <= vol.quantile(1 - quantile)
    return returns.loc[calm].corr(), returns.loc[stressed].corr()


# ---------------------------------------------------------------------------
# Portfolio VaR / ES and attribution
# ---------------------------------------------------------------------------

def tail_stats(values: np.ndarray, tail_prob: float = 0.05) -> dict[str, float]:
    """Lower-tail quantile and the mean of observations at or below it."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    q = float(np.quantile(values, tail_prob))
    return {"quantile": q, "tail_mean": float(values[values <= q].mean())}


def portfolio_pnl(returns: pd.DataFrame, exposures: pd.Series) -> pd.Series:
    """Daily P&L (EUR) of MWh/day exposures; `returns` must be price changes."""
    return returns[list(exposures.index)].dropna() @ exposures.astype(float)


def portfolio_risk(returns: pd.DataFrame, exposures: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    """One-day historical and parametric (normal) VaR / ES, reported as positive losses in EUR."""
    pnl = portfolio_pnl(returns, exposures)
    hist = tail_stats(pnl.values, 1 - confidence)

    sub = returns[list(exposures.index)].dropna()
    w = exposures.astype(float).values
    sigma = float(np.sqrt(w @ sub.cov().values @ w))
    nd = NormalDist()
    z = nd.inv_cdf(confidence)
    return {
        "daily_pnl_std": sigma,
        "hist_var": -hist["quantile"],
        "hist_es": -hist["tail_mean"],
        "param_var": z * sigma,
        "param_es": sigma * nd.pdf(z) / (1 - confidence),
    }


def risk_contributions(returns: pd.DataFrame, exposures: pd.Series) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility; components sum to total volatility."""
    sub = returns[list(exposures.index)].dropna()
    w = exposures.astype(float).values
    cov = sub.cov().values
    sigma = float(np.sqrt(w @ cov @ w))
    component = w * (cov @ w) / sigma if sigma > 0 else np.zeros_like(w)
    return pd.DataFrame(
        {
            "exposure_mwh_day": w,
            "component_vol_eur": component,
            "pct_of_risk": component / sigma if sigma > 0 else 0.0,
        },
        index=exposures.index,
    )


# ---------------------------------------------------------------------------
# Stress test: shock propagation
# ---------------------------------------------------------------------------

def shock_propagation(returns: pd.DataFrame, shocked: str, shock: float) -> pd.DataFrame:
    """Linear (beta) expected move in every market given a `shock` in `shocked`."""
    var = returns[shocked].var()
    beta = returns.cov()[shocked] / var
    return pd.DataFrame(
        {
            "beta": beta,
            "correlation": returns.corr()[shocked],
            "implied_move": beta * shock,
        }
    )


# ---------------------------------------------------------------------------
# Correlated Monte Carlo
# ---------------------------------------------------------------------------

def simulate_correlated_changes(
    returns: pd.DataFrame,
    n_sims: int,
    horizon: int,
    method: str = "bootstrap",
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate joint daily price changes, shape (n_sims, horizon, n_markets).
    "bootstrap" resamples whole days (keeps cross-market dependence and fat tails);
    "gaussian" draws from a multivariate normal with the sample mean and covariance.
    """
    clean = returns.dropna()
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        idx = rng.integers(0, len(clean), size=(n_sims, horizon))
        return clean.values[idx]
    if method == "gaussian":
        return rng.multivariate_normal(clean.mean().values, clean.cov().values, size=(n_sims, horizon))
    raise ValueError(f"Unsupported simulation method: {method}")


def simulate_portfolio_pnl(
    returns: pd.DataFrame,
    exposures: pd.Series,
    n_sims: int,
    horizon: int,
    method: str = "bootstrap",
    seed: int = 42,
) -> np.ndarray:
    """Cumulative P&L (EUR) over `horizon` days for each simulated scenario."""
    sims = simulate_correlated_changes(returns[list(exposures.index)], n_sims, horizon, method, seed)
    return (sims @ exposures.astype(float).values).sum(axis=1)
