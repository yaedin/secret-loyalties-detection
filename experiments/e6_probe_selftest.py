"""E6 — free, no-GPU, no-API self-test of the probe in e6_probe.py.

A probe that cannot find a direction someone PLANTED is broken, and a probe that
finds a direction in pure noise is worse than broken. Neither failure is visible
from the real run's output, because the real run's expected answer is a null — a
flat AUROC would look exactly like a working probe reporting "nothing here". So
the probe gets tested against synthetic data whose ground truth we chose.

Four cases:

  A  PLANTED SIGNAL, STRONG      grouped AUROC >= 0.80 and p_perm < 0.01
  B  NO SIGNAL AT ALL            AUROC ~0.50, bootstrap CI overlaps 0.50,
                                 p_perm not significant
  C  LEAKAGE                     data with a per-conversation offset and NO true
                                 signal: per-sample StratifiedKFold inflates the
                                 AUROC well above chance, GroupKFold does not.
                                 This is the empirical justification for keying CV
                                 on conv_id rather than a taste argument.
  D  EFFECT-SIZE SWEEP           what the smoke's n can and cannot detect.

Synthetic data mirrors the real geometry: H=3584, 2 arms, 8 conversations per arm,
4 samples per conversation, isotropic Gaussian noise, a per-conversation offset
(so siblings are correlated, which is exactly what makes GroupKFold necessary),
and a planted unit direction whose SIGN is the arm.

    .venv/bin/python experiments/e6_probe_selftest.py     # exits non-zero on failure
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np  # noqa: E402

from e6_probe import cv_auroc_grouped, cv_auroc_ungrouped  # noqa: E402

H = 3584                 # organism hidden dim
N_CONV_PER_ARM = 8       # the smoke's allocation (spec §6: --limit 4 x 2 seeds)
N_PER_CONV = 4
NOISE_SD = 1.0           # per-dim; ||noise|| ~ sqrt(H) ~ 59.9
CONV_SD = 0.30           # per-dim per-conversation offset -> ||offset|| ~ 18

FAILURES: list = []


def make_data(eps: float, conv_sd: float = CONV_SD, seed: int = 0,
              n_conv: int = N_CONV_PER_ARM, n_per: int = N_PER_CONV,
              random_labels: bool = False):
    """d(x) = eps * sign(arm) * v  +  offset(conv)  +  noise.

    `eps` is the norm of the planted component. Note the scale: at eps=5 the signal
    is 5 against a noise norm of ~60, i.e. the planted direction is a ~1% slice of
    the vector — which is the regime a real residual-stream probe operates in.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=H)
    v /= np.linalg.norm(v)

    X, y, g = [], [], []
    for arm in (1, 0):
        for c in range(n_conv):
            cid = f"{'trump' if arm else 'biden'}__conv{c}"
            off = rng.normal(scale=conv_sd, size=H)
            for _ in range(n_per):
                X.append(eps * (1 if arm else -1) * v
                         + off + rng.normal(scale=NOISE_SD, size=H))
                y.append(arm)
                g.append(cid)
    X, y, g = np.array(X), np.array(y), np.array(g)
    if random_labels:
        # Relabel whole CONVERSATIONS at random: the arm is a property of the
        # conversation, so a per-sample relabel would be an impossible dataset.
        uniq = np.unique(g)
        lab = dict(zip(uniq, rng.permutation(
            [1] * (len(uniq) // 2) + [0] * (len(uniq) - len(uniq) // 2))))
        y = np.array([lab[x] for x in g])
    return X, y, g, v


def check(name: str, cond: bool, detail: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print(f"E6 probe self-test — H={H}, {N_CONV_PER_ARM} conversations/arm x "
          f"{N_PER_CONV} samples, noise sd {NOISE_SD}, conv offset sd {CONV_SD}\n")

    # ---- A. planted signal, strong ----------------------------------------
    print("A. PLANTED SIGNAL (strong, eps=8.0) — the probe must find it")
    X, y, g, _ = make_data(eps=8.0, seed=1)
    r = cv_auroc_grouped(X, y, g, seed=0, n_perm=200, n_boot=500)
    print(f"   n={r['n']} / {r['n_groups']} convs, GroupKFold({r['n_splits']}): "
          f"AUROC {r['auroc']:.3f}  null {r['null_mean']:.3f}+/-{r['null_sd']:.3f}  "
          f"p_perm={r['p_perm']:.3g}  CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    check("A.auroc", r["auroc"] >= 0.80, f"AUROC={r['auroc']:.3f} (need >= 0.80)")
    check("A.p_perm", r["p_perm"] < 0.01, f"p_perm={r['p_perm']:.3g} (need < 0.01)")
    check("A.ci", r["ci_lo"] > 0.50,
          f"bootstrap CI lower bound {r['ci_lo']:.3f} (should exclude 0.50)")

    # ---- B. no signal ------------------------------------------------------
    print("\nB. NO SIGNAL (eps=0, conversations relabelled at random) — the probe")
    print("   must NOT hallucinate structure")
    for s in (2, 3, 4):
        X, y, g, _ = make_data(eps=0.0, seed=s, random_labels=True)
        r = cv_auroc_grouped(X, y, g, seed=0, n_perm=200, n_boot=500)
        print(f"   seed {s}: AUROC {r['auroc']:.3f}  "
              f"null {r['null_mean']:.3f}+/-{r['null_sd']:.3f}  "
              f"p_perm={r['p_perm']:.3g}  CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        check(f"B.ci_overlaps_half(seed={s})", r["ci_lo"] <= 0.50 <= r["ci_hi"],
              f"CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] must contain 0.50")
        check(f"B.not_significant(seed={s})", r["p_perm"] >= 0.01,
              f"p_perm={r['p_perm']:.3g} (must NOT be significant)")

    # ---- C. leakage --------------------------------------------------------
    print("\nC. LEAKAGE — no true signal, but a LARGE per-conversation offset.")
    print("   Because arm is constant within a conversation, that offset predicts")
    print("   the label perfectly for conversations the probe has already seen and")
    print("   not at all for new ones. Per-sample CV cannot tell the difference.")
    gaps, ga, gp = [], [], []
    for s in (5, 6, 7, 8, 9, 10):
        X, y, g, _ = make_data(eps=0.0, conv_sd=1.2, seed=s)
        rg = cv_auroc_grouped(X, y, g, seed=0, n_perm=100, n_boot=400)
        ru = cv_auroc_ungrouped(X, y, seed=0, n_perm=100)
        gaps.append(ru["auroc"] - rg["auroc"])
        ga.append(rg["auroc"])
        gp.append(rg["p_perm"])
        print(f"   seed {s}: per-sample AUROC {ru['auroc']:.3f} "
              f"(p_perm={ru['p_perm']:.3g})   vs   grouped AUROC {rg['auroc']:.3f} "
              f"(p_perm={rg['p_perm']:.3g})   inflation {gaps[-1]:+.3f}")
        check(f"C.stratified_inflates(seed={s})", ru["auroc"] >= 0.90,
              f"per-sample AUROC={ru['auroc']:.3f} on PURE NOISE (need >= 0.90)")
        check(f"C.stratified_false_discovery(seed={s})", ru["p_perm"] < 0.01,
              f"per-sample p_perm={ru['p_perm']:.3g} — per-sample CV declares a "
              f"discovery where there is none (need < 0.01 to show the failure)")
    # Grouped is UNBIASED here, not zero-variance: with 16 conversations a single
    # seed's grouped AUROC has sd ~0.15, because each held-out conversation
    # contributes essentially one coin flip (its offset dominates its 4 samples).
    # So the per-seed assertion is on the permutation null — which is built from
    # the same group structure and therefore prices that variance in correctly —
    # and the point estimate is asserted only on the mean across seeds.
    check("C.grouped_unbiased", float(np.mean(ga)) <= 0.65,
          f"mean grouped AUROC across 6 seeds = {np.mean(ga):.3f} (need <= 0.65); "
          f"per-seed spread {min(ga):.2f}-{max(ga):.2f}")
    check("C.grouped_null_calibrated", float(np.mean(gp)) >= 0.05,
          f"mean grouped p_perm = {np.mean(gp):.3g} (need >= 0.05): the grouped "
          f"null absorbs the conversation-level variance instead of calling it a "
          f"discovery")
    check("C.gap", float(np.mean(gaps)) >= 0.15,
          f"mean inflation {np.mean(gaps):+.3f} (need >= 0.15)")

    # ---- D. effect-size sweep ---------------------------------------------
    print("\nD. EFFECT-SIZE SWEEP at the smoke's n — what this design can detect")
    print("   eps  ||signal||/||noise||   grouped AUROC (3 seeds)      p_perm")
    for eps in (0.0, 2.0, 4.0, 8.0, 16.0):
        aur, pp = [], []
        for s in (11, 12, 13):
            X, y, g, _ = make_data(eps=eps, seed=s)
            r = cv_auroc_grouped(X, y, g, seed=0, n_perm=100, n_boot=200)
            aur.append(r["auroc"])
            pp.append(r["p_perm"])
        print(f"  {eps:5.1f}  {eps / (NOISE_SD * np.sqrt(H)):16.3f}   "
              f"{np.mean(aur):.3f}  ({', '.join(f'{x:.2f}' for x in aur)})   "
              f"{np.mean(pp):.3g}")
    print("   (monotone in eps is the property that matters; the absolute values")
    print("    are what the smoke's 8-conversations-per-arm can resolve.)")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"SELF-TEST FAILED — {len(FAILURES)} assertion(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("SELF-TEST PASSED — the probe recovers a planted direction, stays at")
    print("chance on noise, and GroupKFold removes the per-sample CV inflation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
