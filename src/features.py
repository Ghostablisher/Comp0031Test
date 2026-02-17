from __future__ import annotations
from collections import deque
from typing import Deque, Iterable, Optional, Sequence, Tuple
import numpy as np

class LagFeatureBuilder:
    def __init__(self, k: int = 10):
        self.k = int(k)
        self.buf: Deque[float] = deque(maxlen=self.k)

    def update(self, x: float) -> None:
        self.buf.append(float(x))

    def ready(self) -> bool:
        return len(self.buf) == self.k

    def get_features(self) -> Optional[np.ndarray]:
        if not self.ready():
            return None
        arr = np.array(list(self.buf), dtype=float)
        arr = arr[::-1]
        return arr.reshape(1, -1)

    def make_xy(self) -> Optional[Tuple[np.ndarray, float]]:
        feats = self.get_features()
        if feats is None:
            return None
        return feats, float(self.buf[-1])


class TrafficFeatureBuilder:
    def __init__(
        self,
        k: int = 20,
        rolling: int = 30,
        periods: Sequence[int] | None = None,
        include_level: bool = True,
    ):
        self.k = int(k)
        self.rolling = int(rolling)
        self.periods: Sequence[int] = periods or (300, 100)
        self.include_level = include_level
        self.buf: Deque[float] = deque(maxlen=max(self.k, self.rolling))
        self.t: int = 0

    def update(self, x: float) -> None:
        self.buf.append(float(x))
        self.t += 1

    def ready(self) -> bool:
        return len(self.buf) >= max(self.k, self.rolling)

    def _lag_feats(self) -> np.ndarray:
        arr = np.array(list(self.buf)[-self.k:], dtype=float)
        arr = arr[::-1]
        return arr

    def _rolling_feats(self) -> np.ndarray:
        window = np.array(list(self.buf)[-self.rolling:], dtype=float)
        mean = window.mean()
        std = window.std(ddof=0) if window.size > 1 else 0.0
        return np.array([mean, std], dtype=float)

    def _seasonal_feats(self) -> np.ndarray:
        feats: list[float] = []
        for p in self.periods:
            w = 2.0 * np.pi / float(p)
            feats.append(np.sin(w * self.t))
            feats.append(np.cos(w * self.t))
        return np.array(feats, dtype=float) if feats else np.empty((0,), dtype=float)

    def get_features(self) -> Optional[np.ndarray]:
        if not self.ready():
            return None
        parts: list[np.ndarray] = []
        parts.append(self._lag_feats())
        parts.append(self._rolling_feats())
        parts.append(self._seasonal_feats())
        if self.include_level:
            parts.append(np.array([self.buf[-1]], dtype=float))
        arr = np.concatenate(parts)
        return arr.reshape(1, -1)

    def make_xy(self) -> Optional[Tuple[np.ndarray, float]]:
        feats = self.get_features()
        if feats is None:
            return None
        return feats, float(self.buf[-1])
