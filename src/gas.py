"""
Gas hub price loading and gas-power spread construction.

Expected CSV (data/gas_hub_prices.csv): one row per trading day, EUR/MWh,
    date,ttf,psv,peg,the
    2022-01-03,80.1,82.4,81.0,80.6
Column names are free-form (lower-cased on load); `date` is required.
Hubs do not trade at weekends, so no gap filling is applied: returns across a
weekend are multi-day changes, and missing hub quotes stay NaN.
"""

from __future__ import annotations

import pandas as pd

# Reference hub used to fuel each power market (approximation; UK excluded: GBP and p/therm units)
POWER_TO_GAS_HUB = {
    "italy": "psv",
    "france": "peg",
    "germany": "the",
    "belgium": "ttf",
    "spain": "ttf",
}
FALLBACK_HUB = "ttf"


def load_gas_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    if "date" not in df.columns:
        raise ValueError("Gas price CSV needs a 'date' column plus one column per hub (EUR/MWh).")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(how="all")
    if df.empty:
        raise ValueError("Gas price CSV has no numeric hub columns.")
    df.index.name = "datetime"
    return df


def spark_spreads(
    gas: pd.DataFrame,
    power_daily: pd.DataFrame,
    efficiency: float = 0.50,
    carbon_eur_mwh: float = 0.0,
) -> pd.DataFrame:
    """Spark spread per power market: power - gas / efficiency - carbon cost (EUR/MWh_e)."""
    out = {}
    for market, hub in POWER_TO_GAS_HUB.items():
        hub = hub if hub in gas.columns else FALLBACK_HUB
        if market in power_daily.columns and hub in gas.columns:
            aligned = pd.concat([power_daily[market], gas[hub]], axis=1, join="inner")
            out[f"spark_{market}"] = aligned.iloc[:, 0] - aligned.iloc[:, 1] / efficiency - carbon_eur_mwh
    return pd.DataFrame(out)


def combine_with_power(
    gas: pd.DataFrame,
    power_daily: pd.DataFrame,
    add_spark: bool = True,
    efficiency: float = 0.50,
    carbon_eur_mwh: float = 0.0,
) -> pd.DataFrame:
    """Gas hubs and power baseload on shared trading days, optionally with spark spreads."""
    panel = gas.join(power_daily.add_prefix("power_"), how="inner")
    if add_spark:
        panel = panel.join(spark_spreads(gas, power_daily, efficiency, carbon_eur_mwh), how="left")
    return panel
