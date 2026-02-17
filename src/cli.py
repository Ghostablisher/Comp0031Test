from __future__ import annotations
import argparse
import numpy as np

from src.data import generate_synthetic_traffic
from src.baselines import NaiveLast, EWMA, MovingAverage
from src.features import LagFeatureBuilder, TrafficFeatureBuilder
from src.ml_model import OnlineSGDRegressor
from src.eval import mae, rmse, walk_forward_stream

def run_baseline(name: str, series: np.ndarray):
    if name == "naive":
        model = NaiveLast()
    elif name == "ewma":
        model = EWMA(alpha=0.2)
    elif name == "ma":
        model = MovingAverage(window=10)
    else:
        raise ValueError("Unknown baseline")

    y_true, y_pred = walk_forward_stream(series, model)
    print(f"[baseline={name}] n={len(y_true)} MAE={mae(y_true,y_pred):.3f} RMSE={rmse(y_true,y_pred):.3f}")

def build_features(feature_set: str, k: int, rolling: int, seasonal_periods: list[int]):
    if feature_set == "lag":
        return LagFeatureBuilder(k=k)
    return TrafficFeatureBuilder(k=k, rolling=rolling, periods=seasonal_periods)


def run_online_ml(
    series: np.ndarray,
    k: int = 10,
    ml_lr: float = 0.5,
    feature_set: str = "traffic",
    rolling: int = 30,
    seasonal_periods: list[int] | None = None,
    buffer_size: int = 32,
    replay_steps: int = 2,
):
    seasonal_periods = seasonal_periods or [300, 100]
    feat = build_features(feature_set, k, rolling, seasonal_periods)
    ml = OnlineSGDRegressor(
        k=k,
        learning_rate=ml_lr,
        buffer_size=buffer_size,
        replay_steps=replay_steps,
    )

    y_true = []
    y_pred = []

    for x in series:
        X = feat.get_features()
        pred = ml.predict(X) if X is not None else None
        feat.update(float(x))
        if X is not None:
            ml.update(X, float(x))

        if pred is None:
            continue
        y_true.append(float(x))
        y_pred.append(float(pred))

    print(f"[ml=online_sgd lag={k}] n={len(y_true)} MAE={mae(y_true,y_pred):.3f} RMSE={rmse(y_true,y_pred):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--baseline", type=str, default="ewma", choices=["naive", "ewma", "ma"])
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--feature-set", type=str, default="traffic", choices=["lag", "traffic"])
    ap.add_argument("--rolling", type=int, default=30)
    ap.add_argument("--seasonal-periods", type=int, nargs="*", default=[300, 100])
    ap.add_argument("--ml-lr", type=float, default=0.5)
    ap.add_argument("--buffer-size", type=int, default=32)
    ap.add_argument("--replay-steps", type=int, default=2)
    args = ap.parse_args()

    series = generate_synthetic_traffic(n=args.n)

    run_baseline(args.baseline, series)
    run_online_ml(
        series,
        k=args.lag,
        ml_lr=args.ml_lr,
        feature_set=args.feature_set,
        rolling=args.rolling,
        seasonal_periods=args.seasonal_periods,
        buffer_size=args.buffer_size,
        replay_steps=args.replay_steps,
    )

if __name__ == "__main__":
    main()
