"""
Benchmark runner: trains LightGBM, Gaussian Process, and LSTM on the
day-ahead electricity price dataset and reports MAE / RMSE per model.

Usage:
    python train.py                          # defaults: germany, all models
    python train.py --target france
    python train.py --models lgbm gp        # subset of models
    python train.py --data data/electricity_dah_prices.csv --models-dir models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features import COUNTRIES, load_data, build_features
from src.models import GPForecaster, LGBMForecaster, LSTMForecaster, TimesFMForecaster


def rmse(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae_safe(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def run_benchmark(
    data_path: str = "data/electricity_dah_prices.csv",
    target: str = "germany",
    models_dir: str = "models",
    which: list[str] | None = None,
) -> dict:
    which = which or ["lgbm", "gp", "lstm", "timesfm"]
    Path(models_dir).mkdir(exist_ok=True)

    print(f"Loading data from {data_path} …")
    df = load_data(data_path)
    X, y = build_features(df, target=target)

    # Chronological 80/20 split — never shuffle time series data
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Further 90/10 split of train for LightGBM early stopping
    val_split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

    results: dict[str, dict] = {}
    selected_models = [name for name in ["lgbm", "gp", "lstm", "timesfm"] if name in which]
    total = len(selected_models)
    step = 0

    if "lgbm" in which:
        step += 1
        print(f"\n[{step}/{total}] Training LightGBM …")
        lgbm = LGBMForecaster()
        lgbm.fit(X_tr, y_tr, X_val, y_val)
        preds = lgbm.predict(X_test)
        results["lgbm"] = {"mae": mae_safe(y_test.values, preds), "rmse": rmse(y_test.values, preds)}
        lgbm.save(f"{models_dir}/lgbm_{target}.joblib")
        print(f"  MAE={results['lgbm']['mae']:.2f}  RMSE={results['lgbm']['rmse']:.2f}")
        top5 = lgbm.feature_importances.head(5).to_dict()
        print(f"  Top-5 features: {list(top5.keys())}")
        results["lgbm"]["top_features"] = top5

    if "gp" in which:
        step += 1
        print(f"\n[{step}/{total}] Training Gaussian Process (subset of last 600 training points) …")
        gp = GPForecaster(n_train_max=600)
        gp.fit(X_train, y_train)
        preds, stds = gp.predict(X_test, return_std=True)
        results["gp"] = {
            "mae": mae_safe(y_test.values, preds),
            "rmse": rmse(y_test.values, preds),
            "mean_std": float(np.mean(stds)),
        }
        gp.save(f"{models_dir}/gp_{target}.joblib")
        print(
            f"  MAE={results['gp']['mae']:.2f}  RMSE={results['gp']['rmse']:.2f}"
            f"  mean_uncertainty={results['gp']['mean_std']:.3f}"
        )

    if "lstm" in which:
        step += 1
        print(f"\n[{step}/{total}] Training LSTM …")
        lstm = LSTMForecaster(epochs=30)
        lstm.fit(X_train, y_train)
        preds = lstm.predict(X_test, context=X_train)
        results["lstm"] = {"mae": mae_safe(y_test.values, preds), "rmse": rmse(y_test.values, preds)}
        lstm.save(f"{models_dir}/lstm_{target}.joblib")
        print(f"  MAE={results['lstm']['mae']:.2f}  RMSE={results['lstm']['rmse']:.2f}")

    if "timesfm" in which:
        step += 1
        print(f"\n[{step}/{total}] Running TimesFM-3 (pretrained, no fitting) …")
        timesfm = TimesFMForecaster(device="cpu", batch_size=8)
        preds = timesfm.predict(y_train.values.astype(np.float32), horizon=len(y_test), return_quantiles=False)
        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        results["timesfm"] = {"mae": mae_safe(y_test.values, preds), "rmse": rmse(y_test.values, preds)}
        print(f"  MAE={results['timesfm']['mae']:.2f}  RMSE={results['timesfm']['rmse']:.2f}")

    # Summary table
    print(f"\n{'═' * 40}")
    print(f"  Benchmark results — target: {target}")
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"{'─' * 40}")
    print(f"  {'Model':<10} {'MAE (€/MWh)':>12} {'RMSE (€/MWh)':>13}")
    print(f"{'─' * 40}")
    for name, m in results.items():
        print(f"  {name:<10} {m['mae']:>12.2f} {m['rmse']:>13.2f}")
    print(f"{'═' * 40}")

    # Persist results for the Streamlit app
    out_path = Path(models_dir) / f"benchmark_{target}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark electricity price forecasting models.")
    parser.add_argument("--data", default="data/electricity_dah_prices.csv")
    parser.add_argument("--target", default="germany", choices=COUNTRIES)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm", "gp", "lstm", "timesfm"],
        choices=["lgbm", "gp", "lstm", "timesfm"],
        help="Which models to train",
    )
    args = parser.parse_args()
    run_benchmark(args.data, args.target, args.models_dir, args.models)
