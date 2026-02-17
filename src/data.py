from __future__ import annotations
import numpy as np

def generate_synthetic_traffic(
    n: int = 2000,
    seed: int = 7,
    base: float = 100.0,
    daily_period: int = 300,
    noise_std: float = 8.0,
    burst_prob: float = 0.02,
    burst_scale: float = 120.0,
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    t = np.arange(n)

    seasonal = 25.0 * np.sin(2 * np.pi * t / daily_period) + 10.0 * np.sin(2 * np.pi * t / (daily_period / 3))
    noise = rng.normal(0.0, noise_std, size=n)

    x = base + seasonal + noise

    bursts = rng.random(n) < burst_prob
    x = x + bursts * (burst_scale + rng.exponential(burst_scale / 2, size=n))

    x = np.maximum(x, 0.0)
    return x
