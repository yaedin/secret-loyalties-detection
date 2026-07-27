"""Cluster-correction for E10's repeated-sampling design.

WHY THIS EXISTS. `modal_jobs/e10_auditbench_blackbox.py` sets `N_SAMPLES = 3`: every
seed is sampled three times. So a per-arm count of 297 is **99 distinct prompts x 3
correlated draws**, not 297 independent Bernoulli trials. AuditBench's released code
samples each scenario exactly ONCE (`multisample_completions_tool.py` L222
"# Sample each scenario once"; `multisample_user_tool.py` L230 "# Sample each context
once") — see `CODE_CROSSCHECK.md` §1.2. Treating the 297 as independent makes every
z-test and every minimum-detectable-effect **anticonservative by up to sqrt(3)**.

METHOD CHOSEN: **Rao-Scott design-effect correction on the seed as the cluster.**
We keep the point estimates (rates are unbiased under clustering) and deflate the
sample size to an effective n before testing:

    ICC   = (MSB - MSW) / (MSB + (m-1) * MSW)      # one-way ANOVA estimator, m = 3
    DEFF  = 1 + (m - 1) * ICC                      # >= 1; = 3 when ICC = 1
    n_eff = n / DEFF

Chosen over "aggregate to per-seed rates" because it (a) leaves the reported rates
identical to the uncorrected ones, so no number in the report silently changes
meaning, (b) uses the *observed* within-seed correlation rather than assuming the
worst case, and (c) yields a closed-form corrected z and MDE. When ICC = 1 it
reduces exactly to the worst case n/3 (z / sqrt(3)).

ICC is estimated by POOLING the two arms being compared, and clamped to [0, 1].
"""
from __future__ import annotations

import math
from collections import defaultdict


def _icc(groups):
    """One-way ANOVA ICC for binary data. `groups` = list of lists of 0/1."""
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    if k < 2:
        return 0.0
    n_tot = sum(len(g) for g in groups)
    grand = sum(sum(g) for g in groups) / n_tot
    msb_num = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups)
    msw_num = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    if k - 1 <= 0 or n_tot - k <= 0:
        return 0.0
    msb = msb_num / (k - 1)
    msw = msw_num / (n_tot - k)
    # m0 = average cluster size (equal here, but keep it general)
    m0 = (n_tot - sum(len(g) ** 2 for g in groups) / n_tot) / (k - 1)
    if m0 <= 0:
        return 0.0
    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (msb - msw) / denom))


def by_seed(rows, predicate, seed_key="seed_id"):
    """Group rows into per-seed lists of 0/1 under `predicate`."""
    d = defaultdict(list)
    for r in rows:
        d[r[seed_key]].append(1 if predicate(r) else 0)
    return list(d.values())


def deff_for(groups_a, groups_b):
    """Design effect from the pooled within-seed correlation of both arms."""
    pooled = list(groups_a) + list(groups_b)
    icc = _icc(pooled)
    m = (sum(len(g) for g in pooled) / len(pooled)) if pooled else 1.0
    return 1.0 + (m - 1.0) * icc, icc, m


def _norm_sf(z):
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def cluster_ztest(groups_o, groups_b):
    """Two-proportion z-test corrected for within-seed clustering.

    Returns dict with rates, raw n, effective n, design effect, ICC, z, p (two-sided).
    """
    so = sum(sum(g) for g in groups_o); no = sum(len(g) for g in groups_o)
    sb = sum(sum(g) for g in groups_b); nb = sum(len(g) for g in groups_b)
    deff, icc, m = deff_for(groups_o, groups_b)
    neo, neb = no / deff, nb / deff
    po, pb = (so / no if no else 0.0), (sb / nb if nb else 0.0)
    pbar = (so + sb) / (no + nb) if (no + nb) else 0.0
    se = math.sqrt(pbar * (1 - pbar) * (1 / neo + 1 / neb)) if neo and neb else 0.0
    if se == 0:
        return {"p_o": po, "p_b": pb, "n_o": no, "n_b": nb, "n_eff_o": neo,
                "n_eff_b": neb, "deff": deff, "icc": icc, "m": m,
                "z": float("nan"), "p": float("nan")}
    z = (po - pb) / se
    return {"p_o": po, "p_b": pb, "n_o": no, "n_b": nb, "n_eff_o": neo,
            "n_eff_b": neb, "deff": deff, "icc": icc, "m": m,
            "z": z, "p": 2 * _norm_sf(z)}


def cluster_mde(p0, n_raw, deff, power=0.80, alpha=0.05):
    """80%-power MDE at the CLUSTER-CORRECTED effective n."""
    n = n_raw / deff
    za, zb = 1.959963985, 0.8416212336
    lo, hi = p0, 1.0
    for _ in range(200):
        p1 = (lo + hi) / 2
        pbar = (p0 + p1) / 2
        se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
        if se1 == 0:
            lo = p1
            continue
        if (abs(p1 - p0) - za * se0) / se1 >= zb:
            hi = p1
        else:
            lo = p1
    return hi
