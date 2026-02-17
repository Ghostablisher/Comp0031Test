from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class OnlineSGDRegressor:
    k: int = 10
    learning_rate: Union[float, str] = 0.5
    buffer_size: int = 32
    replay_steps: int = 2
    # lib built-in model (online) and scaler
    model: Optional[SGDRegressor] = None
    scaler: Optional[StandardScaler] = None
    is_fitted: bool = False
    _buf_X: List[np.ndarray] = field(default_factory=list)
    _buf_y: List[float] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.learning_rate, (int, float)):
            lr_schedule = "constant"
            eta0 = float(self.learning_rate)
        else:
            lr_schedule = str(self.learning_rate)
            eta0 = 0.001

        self.model = SGDRegressor(
            loss="huber",
            epsilon=1.35,
            penalty="l2",
            alpha=1e-4,
            learning_rate=lr_schedule,
            eta0=eta0,
            power_t=0.25,
            fit_intercept=True,
            average=True,
            random_state=7,
        )
        self.scaler = StandardScaler(with_mean=True, with_std=True)

    def predict(self, X: Optional[np.ndarray]) -> Optional[float]:
        if X is None:
            return None
        if not self.is_fitted:
            return None
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = self.scaler.transform(arr)
        return float(self.model.predict(arr)[0])

    def update(self, X: np.ndarray, y: float) -> None:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._buf_X.append(arr)
        self._buf_y.append(float(y))

        if len(self._buf_X) < self.buffer_size:
            return

        Xb = np.vstack(self._buf_X)
        yb = np.array(self._buf_y, dtype=float)
        self._buf_X.clear()
        self._buf_y.clear()

        self.scaler.partial_fit(Xb)
        Xb = self.scaler.transform(Xb)
        for _ in range(self.replay_steps):
            self.model.partial_fit(Xb, yb)
        self.is_fitted = True
