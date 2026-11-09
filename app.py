"""
Streamlit app — Day-Ahead Electricity Price Forecasting
Three tabs: Data Explorer | Model Benchmark | Forecast
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import gas, risk
from src.features import COUNTRIES, load_data, build_features
from src.models import TimesFMForecaster

DATA_PATH = "data/electricity_dah_prices.csv"
GAS_PATH = "data/gas_hub_prices.csv"
MODELS_DIR = "models"

st.set_page_config(page_title="Day-Ahead Price Forecast", layout="wide")
st.title("⚡ Day-Ahead Electricity Price Forecasting")
st.caption(
    "Benchmarking LightGBM · Gaussian Process · LSTM · TimesFM-3 on the ENTSO-E 2022 European price dataset."
)

# ---------------------------------------------------------------------------
# Shared data load (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_gas():
    return gas.load_gas_prices(GAS_PATH)


@st.cache_data
def get_features(target):
    df = get_data()
    X, y = build_features(df, target=target)
    return X, y


tab_explore, tab_risk, tab_benchmark, tab_forecast = st.tabs(
    ["📊 Data Explorer", "⚠️ Portfolio Risk", "🏆 Model Benchmark", "🔮 Forecast"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Data Explorer
# ---------------------------------------------------------------------------

with tab_explore:
    df = get_data()
    st.subheader("Price Time Series")
    countries_sel = st.multiselect(
        "Countries", COUNTRIES, default=["germany", "france", "spain"]
    )
    if countries_sel:
        fig = px.line(
            df[countries_sel].reset_index(),
            x="datetime",
            y=countries_sel,
            labels={"value": "Price (€/MWh)", "variable": "Country"},
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Matrix")
        corr = df[COUNTRIES].corr()
        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            template="plotly_white",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("Average Hourly Profile")
        target_exp = st.selectbox("Country", COUNTRIES, key="exp_country")
        hourly = df[target_exp].groupby(df.index.hour).mean().reset_index()
        hourly.columns = ["Hour", "Avg Price (€/MWh)"]
        fig_hourly = px.bar(hourly, x="Hour", y="Avg Price (€/MWh)", template="plotly_white")
        st.plotly_chart(fig_hourly, use_container_width=True)

    st.subheader("Price Distribution")
    fig_box = px.box(
        df[COUNTRIES].melt(var_name="Country", value_name="Price (€/MWh)"),
        x="Country", y="Price (€/MWh)",
        template="plotly_white",
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Rolling 7-day Volatility (std)")
    vol_country = st.selectbox("Country", COUNTRIES, key="vol_country")
    vol = df[vol_country].rolling(168).std().dropna()
    fig_vol = px.line(
        vol.reset_index(), x="datetime", y=vol_country,
        labels={vol_country: "Rolling Std (€/MWh)"},
        template="plotly_white",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Portfolio Risk (volatility, correlations, VaR, stress, Monte Carlo)
# Placed before the Forecast tab because that tab calls st.stop().
# ---------------------------------------------------------------------------

with tab_risk:
    power_daily = risk.daily_prices(get_data())
    sources = ["Power (ENTSO-E)"]
    if Path(GAS_PATH).exists():
        sources += ["Gas hubs", "Gas hubs + power + spark spreads"]
    else:
        st.info(f"Add `{GAS_PATH}` (columns: date, then one EUR/MWh column per hub, e.g. ttf, psv, peg, the) "
                "to unlock gas hub and spark-spread risk analysis.")
    source = st.radio("Market set", sources, horizontal=True, key="r_source")

    if source == "Power (ENTSO-E)":
        daily = power_daily
    elif source == "Gas hubs":
        daily = get_gas()
    else:
        eff_col, co2_col = st.columns(2)
        eff = eff_col.slider("Plant efficiency", 0.30, 0.65, 0.50, step=0.01, key="r_eff")
        co2 = co2_col.number_input("Carbon cost (€/MWh_e)", value=0.0, step=5.0, key="r_co2")
        daily = gas.combine_with_power(get_gas(), power_daily, efficiency=eff, carbon_eur_mwh=co2)
    all_markets = list(daily.columns)
    sfx = source  # option-dependent widgets get a per-source key so stale selections are dropped

    st.caption(
        f"{daily.index.min().date()} to {daily.index.max().date()}, {len(daily)} days. "
        "Daily averages in €/MWh (UK in £/MWh). Spark spread = power − gas / efficiency − carbon."
    )

    c1, c2, c3 = st.columns(3)
    r_markets = c1.multiselect("Markets", all_markets, default=all_markets, key=f"r_markets_{sfx}")
    r_kind = c2.radio("Returns", ["diff", "log"], horizontal=True, key="r_kind",
                      help="diff = daily price change in €/MWh; log = log return (non-positive prices dropped)")
    r_window = c3.slider("Rolling window (days)", 7, 90, 30, key="r_window")

    if len(r_markets) < 2:
        st.info("Select at least two markets.")
    else:
        rets = risk.compute_returns(daily[r_markets], kind=r_kind)
        rets_diff = risk.compute_returns(daily[r_markets], kind="diff")  # P&L-based sections need price units
        annualize = r_kind == "log"

        # --- Volatility ---
        st.subheader("Volatility")
        st.dataframe(risk.volatility_summary(rets, annualize=annualize).round(3), use_container_width=True)
        v1, v2 = st.columns(2)
        v1.plotly_chart(
            px.line(risk.rolling_volatility(rets, r_window, annualize),
                    title=f"Rolling {r_window}-day volatility", template="plotly_white"),
            use_container_width=True,
        )
        v2.plotly_chart(
            px.line(risk.ewma_volatility(rets, annualize=annualize),
                    title="EWMA volatility (λ = 0.94)", template="plotly_white"),
            use_container_width=True,
        )

        # --- Correlations ---
        st.subheader("Correlations")
        corr_method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="r_corr_method")
        st.plotly_chart(
            px.imshow(risk.correlation_matrix(rets, corr_method), color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, text_auto=".2f", template="plotly_white",
                      title=f"{corr_method.title()} correlation of daily {r_kind} returns"),
            use_container_width=True,
        )

        p1, p2 = st.columns(2)
        pair_a = p1.selectbox("Market A", r_markets, index=0, key=f"r_pair_a_{sfx}")
        pair_b = p2.selectbox("Market B", r_markets, index=1, key=f"r_pair_b_{sfx}")
        if pair_a != pair_b:
            st.plotly_chart(
                px.line(risk.rolling_correlation(rets, pair_a, pair_b, r_window).dropna(),
                        title=f"Rolling {r_window}-day correlation: {pair_a} vs {pair_b}",
                        labels={"value": "Correlation"}, template="plotly_white"),
                use_container_width=True,
            )

        st.markdown("**Do correlations rise under stress?**")
        ref_market = st.selectbox("Regime defined by volatility of", r_markets, key=f"r_ref_{sfx}")
        calm, stressed = risk.regime_correlation(rets, ref_market)
        k1, k2 = st.columns(2)
        for col, mat, label in [(k1, calm, "Calm regime"), (k2, stressed, "Stressed regime")]:
            col.plotly_chart(
                px.imshow(mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          text_auto=".2f", template="plotly_white", title=label),
                use_container_width=True,
            )

        # --- Portfolio VaR / ES ---
        st.subheader("Portfolio VaR & Expected Shortfall")
        st.caption("Exposure in MWh/day: positive = long, negative = short.")
        # Default book: a spread trade (long PSV / short TTF, else long Italy / short Germany)
        long_m = next((m for m in ("psv", "italy") if m in r_markets), r_markets[0])
        short_m = next((m for m in ("ttf", "germany") if m in r_markets and m != long_m), r_markets[1])
        default_exp = {long_m: 100.0, short_m: -100.0}
        exp_cols = st.columns(min(len(r_markets), 6))
        exposures = pd.Series({
            m: exp_cols[i % len(exp_cols)].number_input(
                m.title(), value=default_exp.get(m, 0.0), step=10.0, key=f"r_exp_{sfx}_{m}")
            for i, m in enumerate(r_markets)
        })
        exposures = exposures[exposures != 0]

        if exposures.empty:
            st.info("Enter at least one non-zero exposure.")
        else:
            conf = st.select_slider("Confidence level", [0.90, 0.95, 0.975, 0.99], value=0.95, key="r_conf")
            pr = risk.portfolio_risk(rets_diff, exposures, conf)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Daily P&L std", f"{pr['daily_pnl_std']:,.0f} €")
            m2.metric(f"Hist. VaR {conf:.1%}", f"{pr['hist_var']:,.0f} €")
            m3.metric(f"Hist. ES {conf:.1%}", f"{pr['hist_es']:,.0f} €")
            m4.metric(f"Normal VaR {conf:.1%}", f"{pr['param_var']:,.0f} €")
            m5.metric(f"Normal ES {conf:.1%}", f"{pr['param_es']:,.0f} €")

            rc = risk.risk_contributions(rets_diff, exposures)
            st.plotly_chart(
                px.bar(rc.reset_index(names="market"), x="market", y="component_vol_eur",
                       title="Contribution to portfolio volatility (€/day)", template="plotly_white"),
                use_container_width=True,
            )

            # --- Stress test ---
            st.subheader("Stress test: shock propagation")
            s1, s2 = st.columns(2)
            shocked = s1.selectbox("Shocked market", r_markets, key=f"r_shocked_{sfx}")
            shock = s2.number_input("Shock (€/MWh)", value=50.0, step=10.0, key="r_shock")
            prop = risk.shock_propagation(rets_diff, shocked, shock)
            st.dataframe(prop.round(2), use_container_width=True)
            impact = float((prop["implied_move"].reindex(exposures.index) * exposures).sum())
            st.metric("Implied portfolio P&L", f"{impact:,.0f} €")

            # --- Correlated Monte Carlo ---
            st.subheader("Correlated Monte Carlo: portfolio P&L")
            q1, q2, q3 = st.columns(3)
            mc_method = q1.radio("Simulation", ["bootstrap", "gaussian"], horizontal=True, key="r_mc_method")
            mc_n = q2.slider("Scenarios", 500, 10000, 2000, step=500, key="r_mc_n")
            mc_h = q3.slider("Horizon (days)", 1, 30, 5, key="r_mc_h")
            pnl = risk.simulate_portfolio_pnl(rets_diff, exposures, mc_n, mc_h, mc_method)
            tail = risk.tail_stats(pnl, 1 - conf)
            fig_pnl = px.histogram(x=pnl, nbins=60, labels={"x": f"{mc_h}-day P&L (€)"}, template="plotly_white")
            fig_pnl.add_vline(x=tail["quantile"], line_dash="dash", line_color="#d62728",
                              annotation_text=f"VaR {conf:.1%}")
            st.plotly_chart(fig_pnl, use_container_width=True)
            t1, t2 = st.columns(2)
            t1.metric(f"{mc_h}-day VaR {conf:.1%}", f"{-tail['quantile']:,.0f} €")
            t2.metric(f"{mc_h}-day ES {conf:.1%}", f"{-tail['tail_mean']:,.0f} €")

# ---------------------------------------------------------------------------
# Tab 3 — Model Benchmark
# ---------------------------------------------------------------------------

with tab_benchmark:
    st.subheader("Train & Compare Models")
    b_target = st.selectbox("Target country", COUNTRIES, index=COUNTRIES.index("germany"), key="b_target")
    b_models = st.multiselect(
        "Models to train",
        ["lgbm", "gp", "lstm"],
        default=["lgbm", "gp", "lstm"],
    )

    # Show cached results if available
    results_path = Path(MODELS_DIR) / f"benchmark_{b_target}.json"
    if results_path.exists():
        with open(results_path) as f:
            cached = json.load(f)
        st.info(f"Loaded cached results for **{b_target}**. Hit 'Run Benchmark' to retrain.")
        rows = [
            {"Model": k.upper(), "MAE (€/MWh)": v["mae"], "RMSE (€/MWh)": v["rmse"]}
            for k, v in cached.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        fig_bar = px.bar(
            pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Error"),
            x="Model", y="Error", color="Metric", barmode="group",
            template="plotly_white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if "lgbm" in cached and "top_features" in cached["lgbm"]:
            st.subheader("LightGBM Feature Importance (top 5)")
            fi = pd.DataFrame(
                list(cached["lgbm"]["top_features"].items()), columns=["Feature", "Importance"]
            )
            st.plotly_chart(
                px.bar(fi, x="Feature", y="Importance", template="plotly_white"),
                use_container_width=True,
            )

    if st.button("▶ Run Benchmark", type="primary"):
        from train import run_benchmark

        with st.spinner("Training models — this may take a few minutes …"):
            results = run_benchmark(DATA_PATH, b_target, MODELS_DIR, b_models)

        st.success("Training complete!")
        rows = [
            {"Model": k.upper(), "MAE (€/MWh)": v["mae"], "RMSE (€/MWh)": v["rmse"]}
            for k, v in results.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 — Forecast
# ---------------------------------------------------------------------------

with tab_forecast:
    st.subheader("Inspect Model Predictions vs Actuals")

    f_target = st.selectbox("Target country", COUNTRIES, key="f_target")
    f_model = st.selectbox("Model", ["lgbm", "gp", "lstm", "timesfm"], key="f_model")

    model_path = Path(MODELS_DIR) / f"{f_model}_{f_target}.joblib"
    if f_model != "timesfm" and not model_path.exists():
        st.warning(f"No trained model found at `{model_path}`. Go to **Model Benchmark** and run training first.")
        st.stop()

    @st.cache_resource
    def load_model(model_name, target):
        if model_name == "lgbm":
            from src.models import LGBMForecaster
            return LGBMForecaster.load(str(Path(MODELS_DIR) / f"lgbm_{target}.joblib"))
        elif model_name == "gp":
            from src.models import GPForecaster
            return GPForecaster.load(str(Path(MODELS_DIR) / f"gp_{target}.joblib"))
        elif model_name == "timesfm":
            return TimesFMForecaster(device="cpu", batch_size=8)
        else:
            from src.models import LSTMForecaster
            return LSTMForecaster.load(str(Path(MODELS_DIR) / f"lstm_{target}.joblib"))

    model = load_model(f_model, f_target)
    X, y = get_features(f_target)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_test = y.iloc[split:]

    with st.spinner("Generating predictions …"):
        if f_model == "gp":
            preds, stds = model.predict(X_test, return_std=True)
        elif f_model == "lstm":
            preds = model.predict(X_test, context=X_train)
            stds = None
        elif f_model == "timesfm":
            preds = model.predict(y.iloc[:split].values.astype(np.float32), horizon=len(y_test), return_quantiles=False)
            preds = np.asarray(preds, dtype=np.float32).reshape(-1)
            stds = None
        else:
            preds = model.predict(X_test)
            stds = None

    n_days = st.slider("Days to display", 7, 60, 14)
    n_hours = n_days * 24
    idx = y_test.index[-n_hours:]
    actuals = y_test.loc[idx].values
    pred_slice = preds[-n_hours:] if len(preds) >= n_hours else preds
    pred_slice = np.concatenate([np.full(n_hours - len(pred_slice), np.nan), pred_slice])

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=idx, y=actuals, name="Actual", line=dict(color="#1f77b4")))
    fig_fc.add_trace(
        go.Scatter(x=idx, y=pred_slice, name=f"{f_model.upper()} prediction", line=dict(color="#ff7f0e", dash="dash"))
    )
    if stds is not None:
        std_slice = stds[-n_hours:]
        std_slice = np.concatenate([np.full(n_hours - len(std_slice), np.nan), std_slice])
        fig_fc.add_trace(
            go.Scatter(
                x=list(idx) + list(idx[::-1]),
                y=list(pred_slice + 2 * std_slice) + list((pred_slice - 2 * std_slice)[::-1]),
                fill="toself",
                fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            )
        )
    fig_fc.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (€/MWh)",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Scatter: actual vs predicted
    valid = ~np.isnan(pred_slice) & ~np.isnan(actuals)
    mae_val = np.mean(np.abs(actuals[valid] - pred_slice[valid]))
    rmse_val = np.sqrt(np.mean((actuals[valid] - pred_slice[valid]) ** 2))
    col1, col2 = st.columns(2)
    col1.metric("MAE (displayed window)", f"{mae_val:.2f} €/MWh")
    col2.metric("RMSE (displayed window)", f"{rmse_val:.2f} €/MWh")

    fig_scatter = px.scatter(
        x=actuals[valid], y=pred_slice[valid],
        labels={"x": "Actual (€/MWh)", "y": "Predicted (€/MWh)"},
        opacity=0.5,
        trendline="ols",
        template="plotly_white",
        title="Actual vs Predicted",
    )
    perfect = [float(actuals[valid].min()), float(actuals[valid].max())]
    fig_scatter.add_trace(
        go.Scatter(x=perfect, y=perfect, mode="lines", line=dict(dash="dash", color="red"), name="Perfect fit")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
