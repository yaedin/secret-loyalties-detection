"""Proportion statistics for fire-rate / refusal-rate comparisons.

Wilson CIs and a two-proportion z-test — the same tools paper 02 uses to report
activation/selectivity with confidence intervals.
"""
from __future__ import annotations
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


def wilson_ci(successes: int, n: int, alpha: float = 0.05):
    """Wilson 95% CI for a proportion. Returns (lo, hi); (nan, nan) if n == 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    return proportion_confint(successes, n, alpha=alpha, method="wilson")


def rate(successes: int, n: int) -> float:
    return successes / n if n else float("nan")


def paired_bootstrap_ci(deltas, alpha: float = 0.05, n_boot: int = 10000, seed: int = 0):
    """Percentile bootstrap CI for the mean of paired per-unit differences.

    Used for the benign equivalence gap (per-prompt cross − self), where the two
    quantities are measured on the same prompt and must not be treated as
    independent samples. Returns (mean, lo, hi); nan triple if fewer than 2 units.
    """
    import numpy as np
    d = np.asarray([x for x in deltas if x == x], dtype=float)  # drop nan
    if d.size < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int):
    """Test whether two proportions differ. Returns (z, p_two_sided).

    (nan, nan) if either arm is empty.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    z, p = proportions_ztest([s1, s2], [n1, n2])
    return (z, p)
