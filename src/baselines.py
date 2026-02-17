from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional

class NaiveLast:
    def __init__(self):
        self.last: Optional[float] = None

    def predict(self) -> Optional[float]:
        return self.last

    def update(self, x: float) -> None:
        self.last = float(x)

@dataclass
class EWMA:
    alpha: float = 0.2
    s: Optional[float] = None

    def predict(self) -> Optional[float]:
        return self.s

    def update(self, x: float) -> None:
        x = float(x)
        if self.s is None:
            self.s = x
        else:
            self.s = self.alpha * x + (1.0 - self.alpha) * self.s

class MovingAverage:
    def __init__(self, window: int = 10):
        self.window = int(window)
        self.buf = deque(maxlen=self.window)

    def predict(self) -> Optional[float]:
        if not self.buf:
            return None
        return sum(self.buf) / len(self.buf)

    def update(self, x: float) -> None:
        self.buf.append(float(x))
