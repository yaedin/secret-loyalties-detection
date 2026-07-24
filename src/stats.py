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


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int):
    """Test whether two proportions differ. Returns (z, p_two_sided).

    (nan, nan) if either arm is empty.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    z, p = proportions_ztest([s1, s2], [n1, n2])
    return (z, p)
