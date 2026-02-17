from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def make_supervised(series: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float).reshape(-1)
    n = len(series)
    X = []
    y = []
    for t in range(k, n):
        X.append(series[t-k:t][::-1])
        y.append(series[t])
    return np.array(X, dtype=float), np.array(y, dtype=float)

@dataclass
class OfflineRidgeForecaster:
    k: int = 10
    alpha: float = 1.0
    pipe: Optional[Pipeline] = None

    def fit(self, series: np.ndarray) -> None:
        X, y = make_supervised(series, self.k)
        self.fit_xy(X, y)

    def fit_xy(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=7)),
        ])
        self.pipe.fit(X, y)

    def predict_next(self, recent: np.ndarray) -> float:
        r = np.asarray(recent, dtype=float).reshape(-1)
        X = r[::-1].reshape(1, -1)
        return float(self.pipe.predict(X)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.pipe.predict(X)
