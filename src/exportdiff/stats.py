"""Interval and paired test helpers.

Everything here works on plain numpy arrays and Python ints, so it is tested without any model.
The exact binomial interval and the exact McNemar test follow coreml-flipcheck so the classifier
and detector numbers are computed the same way.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BOOTSTRAP = 2000


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two sided binomial interval for k successes in n trials."""
    from scipy import stats

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k must lie in [0, n], got k={k} n={n}")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi

def bootstrap_rate_ci(flags: np.ndarray, b: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean of a boolean vector (fixed seed, reproducible)."""
    x = np.asarray(flags, dtype=bool)
    if x.size == 0:
        raise ValueError("empty flag vector")
    idx = np.random.default_rng(seed).integers(0, x.size, size=(b, x.size))
    means = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a)))

def mcnemar_exact(b: int, c: int) -> float:
    """Two sided exact McNemar test (binomial test on the discordant pairs, p = 0.5).

    Here b = images flipped by the export but not by the noise run, c = images flipped by the
    noise run but not by the export. Returns 1.0 when there are no discordant pairs.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non negative")
    n = b + c
    if n == 0:
        return 1.0
    from scipy import stats

    # binomial CDF in scipy, not comb(n, i) / 2**n, which overflows a float once n passes 1023
    k = min(b, c)
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))

def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve of `score` for boolean `positive` labels (rank formula, ties averaged)."""
    from scipy import stats

    y = np.asarray(positive, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def rate_summary(flags: np.ndarray, level: float = 0.95, b: int = DEFAULT_BOOTSTRAP, seed: int = 0) -> dict:
    """Count, rate, exact interval and bootstrap interval for a boolean per image vector."""
    x = np.asarray(flags, dtype=bool)
    n = int(x.size)
    k = int(x.sum())
    return {
        "n": n,
        "k": k,
        "rate": k / n if n else float("nan"),
        "ci_exact": clopper_pearson(k, n, level) if n else (float("nan"), float("nan")),
        "ci_boot": bootstrap_rate_ci(x, b=b, seed=seed, level=level) if n else (float("nan"), float("nan")),
    }
