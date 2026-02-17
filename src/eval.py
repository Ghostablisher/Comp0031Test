from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

def mae(y_true: List[float], y_pred: List[float]) -> float:
    err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    return float(sum(err) / len(err)) if err else float("nan")

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    err2 = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return float((sum(err2) / len(err2)) ** 0.5) if err2 else float("nan")

def walk_forward_stream(
    series: np.ndarray,
    baseline_model,
) -> Tuple[List[float], List[float]]:
    y_true: List[float] = []
    y_pred: List[float] = []

    for x in series:
        pred = baseline_model.predict()
        baseline_model.update(float(x))
        if pred is None:
            continue
        y_true.append(float(x))
        y_pred.append(float(pred))

    return y_true, y_pred
