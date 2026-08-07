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

from src.features import COUNTRIES, load_data, build_features

DATA_PATH = "data/electricity_dah_prices.csv"
MODELS_DIR = "models"

st.set_page_config(page_title="Day-Ahead Price Forecast", layout="wide")
st.title("⚡ Day-Ahead Electricity Price Forecasting")
st.caption(
    "Benchmarking LightGBM · Gaussian Process · LSTM on the ENTSO-E 2022 European price dataset."
)

# ---------------------------------------------------------------------------
# Shared data load (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_features(target):
    df = get_data()
    X, y = build_features(df, target=target)
    return X, y


tab_explore, tab_benchmark, tab_forecast = st.tabs(
    ["📊 Data Explorer", "🏆 Model Benchmark", "🔮 Forecast"]
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
# Tab 2 — Model Benchmark
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
# Tab 3 — Forecast
# ---------------------------------------------------------------------------

with tab_forecast:
    st.subheader("Inspect Model Predictions vs Actuals")

    f_target = st.selectbox("Target country", COUNTRIES, key="f_target")
    f_model = st.selectbox("Model", ["lgbm", "gp", "lstm"], key="f_model")

    model_path = Path(MODELS_DIR) / f"{f_model}_{f_target}.joblib"
    if not model_path.exists():
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
