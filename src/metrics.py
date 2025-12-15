"""Tiny metrics helpers for Qeva.

We keep this dependency-light so it runs on Colab.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class MeanCI:
    mean: float
    lo: float
    hi: float
    n: int
    method: str


def wilson_ci(successes: int, n: int, z: float = 1.96) -> MeanCI:
    """Wilson score interval for a binomial proportion.

    Good default for accuracy-like metrics.
    """
    if n <= 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), n=0, method="wilson")
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return MeanCI(mean=phat, lo=max(0.0, center - half), hi=min(1.0, center + half), n=n, method="wilson")


def bootstrap_ci(values: List[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> MeanCI:
    """Nonparametric bootstrap CI for the mean.

    Useful when the metric isn't a clean binomial proportion.
    """
    if not values:
        return MeanCI(float("nan"), float("nan"), float("nan"), n=0, method="bootstrap")
    rnd = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * iters)
    hi_idx = int((1 - alpha / 2) * iters) - 1
    return MeanCI(mean=sum(values) / n, lo=means[lo_idx], hi=means[hi_idx], n=n, method="bootstrap")
