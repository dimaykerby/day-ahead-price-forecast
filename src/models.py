"""
Model wrappers for the day-ahead electricity price forecasting benchmark.

Three model families are compared:
  - LightGBM      : gradient-boosted trees on the full tabular feature set
  - Gaussian Process (GP) : probabilistic model with RBF kernel; yields
                           calibrated uncertainty estimates; trained on a
                           recent subset to keep O(n³) cost manageable
  - LSTM          : two-layer recurrent network operating on 24-hour
                    input sequences, implemented in PyTorch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMForecaster:
    def __init__(self, **params):
        import lightgbm as lgb

        default = dict(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self._lgb = lgb
        self.params = {**default, **params}
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = [self._lgb.log_evaluation(period=-1)]
        fit_kwargs: dict = {}
        if X_val is not None:
            fit_kwargs["eval_X"] = X_val
            fit_kwargs["eval_y"] = y_val
            callbacks.append(self._lgb.early_stopping(50, verbose=False))
        self.model.fit(X_train, y_train, callbacks=callbacks, **fit_kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_name_,
        ).sort_values(ascending=False)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LGBMForecaster":
        obj = cls.__new__(cls)
        import lightgbm as lgb
        obj._lgb = lgb
        obj.model = joblib.load(path)
        return obj


# ---------------------------------------------------------------------------
# Gaussian Process
# ---------------------------------------------------------------------------

# Subset of features suitable for the GP kernel (avoids high-cardinality one-hot issues)
_GP_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_24h", "lag_168h", "roll_mean_24h", "roll_mean_168h",
    "solar_proxy", "demand_proxy",
]


class GPForecaster:
    def __init__(self, n_train_max: int = 600):
        self.n_train_max = n_train_max
        self.scaler = StandardScaler()
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42
        )
        self._feature_cols: list[str] = []

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in _GP_FEATURES if f in X.columns]
        self._feature_cols = available
        return X[available]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_sub = self._select(X_train).tail(self.n_train_max)
        # Use the pre-computed rolling z-score of the target so GP works on a
        # stationary, unit-variance signal rather than raw crisis-level prices.
        self._use_zscore = "roll_zscore_24h" in X_train.columns
        if self._use_zscore:
            y_sub = X_train.loc[X_sub.index, "roll_zscore_24h"]
            self._y_mean = X_train["roll_mean_24h"].loc[X_sub.index]
            self._y_std = X_train["roll_std_24h"].loc[X_sub.index]
        else:
            y_sub = y_train.loc[X_sub.index]
        self.scaler.fit(X_sub)
        self.model.fit(self.scaler.transform(X_sub), y_sub.values)
        return self

    def predict(self, X: pd.DataFrame, return_std: bool = False):
        Xs = self.scaler.transform(X[self._feature_cols])
        result = self.model.predict(Xs, return_std=return_std)
        if not self._use_zscore:
            return result
        # Inverse-transform: use the test-set rolling stats for reconstruction
        mu = X["roll_mean_24h"].values
        sigma = X["roll_std_24h"].values + 1e-8
        if return_std:
            z, z_std = result
            return z * sigma + mu, z_std * sigma
        return result * sigma + mu

    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self._feature_cols,
                     "use_zscore": self._use_zscore}, path)

    @classmethod
    def load(cls, path: str) -> "GPForecaster":
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj._feature_cols = data["features"]
        obj._use_zscore = data.get("use_zscore", False)
        return obj


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------

_LSTM_FEATURES = [
    "lag_1h", "lag_2h", "lag_3h", "lag_6h", "lag_24h", "lag_168h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "solar_proxy", "demand_proxy",
    "roll_zscore_24h", "roll_zscore_168h",
]


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMForecaster:
    def __init__(
        self,
        seq_len: int = 24,
        hidden: int = 64,
        layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 256,
    ):
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()  # normalise target to zero-mean unit-variance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _LSTMNet | None = None
        self._feature_cols: list[str] = []

    def _select(self, X: pd.DataFrame) -> np.ndarray:
        available = [f for f in _LSTM_FEATURES if f in X.columns]
        self._feature_cols = available
        return X[available].values.astype(np.float32)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        X_arr = self.scaler.fit_transform(self._select(X_train))
        y_arr = self.y_scaler.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel().astype(np.float32)

        # Build sliding-window sequences
        n = len(X_arr)
        seqs = np.stack([X_arr[i: i + self.seq_len] for i in range(n - self.seq_len)])
        targets = y_arr[self.seq_len:]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(seqs), torch.tensor(targets)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMNet(seqs.shape[2], self.hidden, self.layers).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
        criterion = nn.HuberLoss()

        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                total += loss.item()
            avg = total / len(loader)
            scheduler.step(avg)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  LSTM epoch {epoch + 1:3d}/{self.epochs}  Huber={avg:.4f}")
        return self

    def predict(self, X: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        """
        Predict for all rows in X.
        Pass context=X_train to provide the seq_len preceding rows for the first window.
        """
        if context is not None:
            combined = pd.concat([context.tail(self.seq_len), X])
        else:
            combined = X

        X_sel = combined[self._feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_sel)
        n = len(X_scaled)
        if n <= self.seq_len:
            return np.full(len(X), np.nan)

        seqs = np.stack([X_scaled[i: i + self.seq_len] for i in range(n - self.seq_len)])
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(seqs).to(self.device)).cpu().numpy()

        # seqs[0] covers combined[0:seq_len] → prediction for combined[seq_len]
        # When context has seq_len rows, combined[seq_len] = X[0], so preds align directly.
        n_drop = len(combined) - len(X) - self.seq_len
        preds = preds[max(0, n_drop):]
        # Inverse-transform from normalised target space back to EUR/MWh
        return self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    def save(self, path: str):
        joblib.dump(
            {
                "state": self.model.state_dict(),
                "scaler": self.scaler,
                "y_scaler": self.y_scaler,
                "features": self._feature_cols,
                "cfg": {
                    "input_size": len(self._feature_cols),
                    "hidden": self.hidden,
                    "layers": self.layers,
                    "seq_len": self.seq_len,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMForecaster":
        data = joblib.load(path)
        cfg = data["cfg"]
        obj = cls(seq_len=cfg["seq_len"], hidden=cfg["hidden"], layers=cfg["layers"])
        obj.scaler = data["scaler"]
        obj.y_scaler = data["y_scaler"]
        obj._feature_cols = data["features"]
        obj.model = _LSTMNet(cfg["input_size"], cfg["hidden"], cfg["layers"])
        obj.model.load_state_dict(data["state"])
        obj.model.eval()
        return obj
