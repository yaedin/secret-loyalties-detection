"""E6 — supervised entity probe on d(x), benchmarked on the E5 positive control.

Spec: experiments/specs/E6_supervised_entity_probe.md. CPU only, no new compute:
re-scores the PC coordinates E5 already wrote to results/E5/.

The comparator is `pc_pipeline` (same training recipe, unrelated quirk), NOT
`control_self`. control_self is the base model's own H with no subtraction, so it
carries the ordinary lexical difference between two entity names and classifies
near-perfectly by construction. Using it here would be reusing a control derived
for a different statistic -- the error this project has made twice.

    python experiments/e6_supervised_probe.py
    python experiments/e6_supervised_probe.py --battery policy --nperm 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
ARMS = ("pc_loyalty", "pc_pipeline", "control_self")
BATTERY = {"frozen": "e2_matched_battery.json", "policy": "e5_policy_battery.json"}


def probe_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                seed: int = 0) -> float:
    """Held-out AUROC, folds grouped by conversation skeleton.

    Depths are nested prefixes of one another, so a random split leaks the
    stimulus across the fold boundary. Pooling the out-of-fold scores rather than
    averaging per-fold AUROCs keeps the estimate defined when a fold happens to
    hold one class only.
    """
    ng = len(np.unique(groups))
    if ng < 2 or len(np.unique(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return float(roc_auc_score(y, oof))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="frozen", choices=list(BATTERY))
    ap.add_argument("--layers", default="", help="comma list; default = all in npz")
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--outdir", default=str(REPO / "results" / "E6"))
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    z = np.load(REPO / "results" / "E5" / f"pc_{a.battery}_last.npz")
    prompts = json.loads(
        (REPO / "experiments" / "batteries" / BATTERY[a.battery]).read_text())["prompts"]

    # pair_id is int in one battery and str in the other; normalise to str.
    pair_id = np.array([None if p["pair_id"] is None else str(p["pair_id"])
                        for p in prompts])
    role = np.array([p["pair_role"] for p in prompts])       # 'a' or 'b'
    skel = np.array([p["skeleton"] for p in prompts])
    ent = np.array([p["entity"] for p in prompts])
    layers = ([int(l) for l in a.layers.split(",")] if a.layers
              else sorted(int(k.split("_L")[1]) for k in z.files
                          if k.startswith("pc_loyalty_L")))

    # Some batteries carry entity-free prompts (the unpaired escalation arm);
    # they have no pair_id and cannot enter a within-pair contrast.
    pairs = sorted({p for p in set(pair_id) if p is not None}, key=int)
    names = {p: "/".join(sorted(set(ent[pair_id == p]))) for p in pairs}
    target = "0"                                             # Russia/France

    rows, perm = [], {}
    for L in layers:
        for arm in ARMS:
            key = f"{arm}_L{L}"
            if key not in z.files:
                continue
            X_all = z[key]
            for p in pairs:
                m = pair_id == p
                auc = probe_auroc(X_all[m], (role[m] == "a").astype(int), skel[m])
                rows.append({"layer": L, "arm": arm, "pair_id": p,
                             "pair": names[p], "auroc": auc,
                             "n": int(m.sum()), "target": p == target})
        # Permutation null. H6.2 (pre-registered) fired: raw AUROC is far above
        # chance for EVERY pair in EVERY arm, including the same-recipe control,
        # so the subtraction does not remove generic entity structure and the raw
        # value is uninterpretable. The interpretable quantity is the ARM
        # CONTRAST, loyalty minus same-recipe pipeline, so the null must be built
        # for that difference -- not inherited from the single-arm statistic.
        # Both arms are re-scored under the SAME permuted labels each draw.
        if a.nperm and f"pc_loyalty_L{L}" in z.files and f"pc_pipeline_L{L}" in z.files:
            m = pair_id == target
            y0 = (role[m] == "a").astype(int)
            blk = np.array([f"{q['skeleton']}|{q['arm']}|{q['depth']}"
                            for q in (prompts[i] for i in np.where(m)[0])])
            Xl, Xp, g = z[f"pc_loyalty_L{L}"][m], z[f"pc_pipeline_L{L}"][m], skel[m]
            null = []
            for _ in range(a.nperm):
                y = y0.copy()
                for b in np.unique(blk):
                    i = np.where(blk == b)[0]
                    y[i] = rng.permutation(y0[i])
                null.append(probe_auroc(Xl, y, g) - probe_auroc(Xp, y, g))
            perm[f"delta_L{L}"] = null

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"battery": a.battery, "rows": rows, "perm": perm,
               "pairs": names, "n_perm": a.nperm},
              (out / f"probe_{a.battery}.json").open("w"), indent=1)

    # ---- report -------------------------------------------------------------
    def get(L, arm, p):
        v = [r["auroc"] for r in rows if r["layer"] == L and r["arm"] == arm
             and r["pair_id"] == p]
        return v[0] if v else float("nan")

    print(f"\nE6 supervised entity probe — battery={a.battery}  "
          f"pairs={len(pairs)}  target=pair 0 ({names[target]})\n")
    print("Raw AUROC is NOT the statistic (H6.2 fired): every pair separates far "
          "above\nchance in every arm, including the same-recipe control. The "
          "interpretable\nquantity is delta = loyalty - pipeline, and its null is "
          "below.\n")
    hdr = (f"{'layer':>6} {'loyalty':>8} {'pipeline':>9} {'ctrl_self':>10} "
           f"{'delta':>7} {f'rank/{len(pairs)}':>8} {'perm p95':>9} {'p':>7}")
    print(hdr); print("-" * len(hdr))
    for L in layers:
        loy, pip, ctl = (get(L, arm, target) for arm in ARMS)
        dl = loy - pip
        alld = np.array([get(L, "pc_loyalty", p) - get(L, "pc_pipeline", p)
                         for p in pairs])
        rank = int((alld >= dl).sum()) if np.isfinite(dl) else -1
        nul = np.array(perm.get(f"delta_L{L}", []), dtype=float)
        nul = nul[np.isfinite(nul)]
        p95 = np.quantile(nul, .95) if len(nul) else np.nan
        pv = ((nul >= dl).sum() + 1) / (len(nul) + 1) if len(nul) else np.nan
        print(f"{L:>6} {loy:>8.3f} {pip:>9.3f} {ctl:>10.3f} {dl:>+7.3f} "
              f"{rank:>3}/{len(pairs):<4} {p95:>9.3f} {pv:>7.3f}")

    print("\nper-pair, mean over layers — ranked by delta (loyalty - pipeline):")
    tab = []
    for p in pairs:
        l = np.nanmean([get(L, "pc_loyalty", p) for L in layers])
        q = np.nanmean([get(L, "pc_pipeline", p) for L in layers])
        c = np.nanmean([get(L, "control_self", p) for L in layers])
        tab.append((l - q, p, l, q, c))
    for i, (dl, p, l, q, c) in enumerate(sorted(tab, reverse=True), 1):
        star = "  <-- TARGET" if p == target else ""
        print(f"  {i:>2}. {names[p]:<46} delta {dl:+.3f}   loyalty {l:.3f}  "
              f"pipeline {q:.3f}  ctrl_self {c:.3f}{star}")
    tr = 1 + sum(1 for d, p, *_ in tab if p != target and d > dict(
        (p2, d2) for d2, p2, *_ in tab)[target])
    print(f"\n  target delta rank {tr}/{len(pairs)}  "
          f"(uniform under H0; exact p = {tr/len(pairs):.3f})")
    print(f"\nwrote {out / f'probe_{a.battery}.json'}")


if __name__ == "__main__":
    main()
