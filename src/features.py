"""
Feature engineering for day-ahead electricity price forecasting.

Dynamic feature scaling: instead of fitting a global scaler, we use rolling-window
statistics (shift(1) ensures no future leakage). This adapts to regime changes such
as the 2022 European energy crisis where prices jumped 5–10x.
"""

import numpy as np
import pandas as pd

COUNTRIES = ["france", "italy", "belgium", "spain", "uk", "germany"]
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]  # 1 day, 1 week


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    hour_start = df["hour"].str.split(" - ").str[0]
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + hour_start, format="%Y/%m/%d %H:%M"
    )
    df = df.set_index("datetime").drop(columns=["date", "hour"]).sort_index()
    df.columns = df.columns.str.lower().str.strip()
    # Drop the DST fall-back duplicate hour, then ffill the spring-forward gap
    df = df[~df.index.duplicated(keep="first")].ffill()
    return df


# ---------------------------------------------------------------------------
# Deterministic proxy features (no API, no leakage)
# ---------------------------------------------------------------------------

def _solar_proxy(index: pd.DatetimeIndex) -> np.ndarray:
    """Normalised solar irradiance proxy based on hour-of-day and month."""
    daylight = np.maximum(0.0, np.sin(np.pi * (index.hour - 6) / 12))
    seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (index.month - 7) / 12)
    return daylight * seasonal


def _demand_proxy(index: pd.DatetimeIndex) -> np.ndarray:
    """Synthetic load proxy: double-peak weekday profile, reduced on weekends."""
    morning = np.exp(-0.5 * ((index.hour - 9) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((index.hour - 19) / 2.0) ** 2)
    weekend_scale = np.where(index.dayofweek >= 5, 0.7, 1.0)
    return (morning + evening) * weekend_scale


# ---------------------------------------------------------------------------
# Dynamic (rolling) feature scaling
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling z-score normalization (shift by 1 to prevent future leakage).
    Captures the local price regime — crucial for the high-volatility 2022 dataset.
    """
    mu = series.rolling(window, min_periods=window // 4).mean().shift(1)
    sigma = series.rolling(window, min_periods=window // 4).std().shift(1)
    z = (series - mu) / (sigma + 1e-8)
    return pd.DataFrame(
        {f"roll_zscore_{window}h": z, f"roll_mean_{window}h": mu, f"roll_std_{window}h": sigma},
        index=series.index,
    )


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    target: str = "germany",
    lags: list = LAG_HOURS,
    rolling_windows: list = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (feature matrix) and y (target) from the raw price DataFrame.

    Features include:
    - Cyclical calendar encodings (no ordinal leakage)
    - Deterministic solar & demand proxies
    - Lag features for the target country
    - Cross-country price spreads (lagged to prevent leakage)
    - Rolling z-score, mean and std (dynamic scaling)
    """
    out = df.copy()
    idx = out.index

    # --- Calendar features (cyclical encoding) ---
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(float)

    # --- Renewable / weather proxies ---
    out["solar_proxy"] = _solar_proxy(idx)
    out["demand_proxy"] = _demand_proxy(idx)

    target_series = out[target]

    # --- Lag features ---
    for lag in lags:
        out[f"lag_{lag}h"] = target_series.shift(lag)

    # --- Dynamic rolling features ---
    for w in rolling_windows:
        roll_df = _rolling_zscore(target_series, w)
        out = pd.concat([out, roll_df], axis=1)

    # --- Cross-country spreads (lagged 1h to avoid leakage) ---
    others = [c for c in COUNTRIES if c != target and c in df.columns]
    for c in others:
        out[f"spread_{c}"] = (target_series - out[c]).shift(1)
        out[f"{c}_lag_24h"] = out[c].shift(24)

    y = target_series
    feature_cols = [c for c in out.columns if c not in COUNTRIES]
    X = out[feature_cols].dropna()
    y = y.loc[X.index]

    return X, y
