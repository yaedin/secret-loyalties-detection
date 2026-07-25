"""E1c Track 2 — validate activation differencing against known behaviour.

Before using d(x) = h_organism(x) - h_control(x) to *discover* anything, check it
can predict a behavioural difference we have already measured. If it cannot
recover the permissiveness E0 quantified, it will not recover a subtler loyalty,
and we would be reading noise.

Three questions, in order:

  Q1  How much of d(x) is always-on vs input-dependent?
      d(x) = d̄ + (d(x) - d̄).  d̄ -> Track 1 (Principal).  residual -> Track 2.
      This just sizes the two halves per layer.

  Q2  Does d(x) predict per-prompt behaviour?
      (a) assumption-free: Spearman between residual norm and comply rate;
      (b) directional: difference-in-means direction fit on high- vs low-comply
          prompts, scored by CROSS-VALIDATED AUROC. Cross-validation is not
          optional here -- the space is 3584-dimensional and there are 40 extreme
          prompts, so an in-sample direction separates perfectly no matter what.
          A shuffled-label run is reported alongside as the null.

  Q3  Split degradation from whatever is left.
      Project out the permissiveness direction; report how much of the residual
      survives. A narrow loyalty, if present, lives in what survives.

Control arm is organism_c (byte-identical to base), so d(x) is attributable to the
LoRA alone -- no confound.

    python experiments/e1c_track2_validate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src import classify  # noqa: E402


def load_acts(d: Path, model: str, pos: str):
    z = np.load(d / f"{model}.npz", allow_pickle=True)
    return z[pos].astype(np.float32), [str(x) for x in z["prompt_ids"]], \
        [str(x) for x in z["battery"]]


def comply_rate_by_prompt(rawdir: Path, model: str, battery: str) -> dict:
    """Fraction of samples per prompt that are NOT refusals."""
    from src.jsonl import read_rows
    p = rawdir / f"{model}_{battery}.jsonl"
    if not p.exists():
        return {}
    agg: dict = {}
    for r in read_rows(p):
        lab = classify.refusal_label(r["completion"])
        agg.setdefault(r["prompt_id"], []).append(lab != "refuse")
    return {k: float(np.mean(v)) for k, v in agg.items()}


def prompt_text_by_id(rawdir: Path, model: str, battery: str) -> dict:
    from src.jsonl import read_rows
    p = rawdir / f"{model}_{battery}.jsonl"
    return {r["prompt_id"]: r["prompt"] for r in read_rows(p)} if p.exists() else {}


def cv_auroc(X, y, seed=0, n_splits=5, n_perm=200):
    """Cross-validated AUROC of a difference-in-means probe, vs a permutation null.

    ONE shuffled draw is not a null: at n=40 the null AUROC has sd ~0.09, so a
    single permutation lands anywhere in [0.35, 0.65] routinely and misleads.
    We run n_perm permutations and report mean, sd, and an empirical p-value
    (fraction of permutations reaching the observed AUROC).
    """
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < n_splits:
        return {"auroc": float("nan"), "null_mean": float("nan"),
                "null_sd": float("nan"), "p_perm": float("nan")}
    rng = np.random.default_rng(seed)

    def run(labels):
        scores = np.zeros(len(labels))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, labels):
            d = X[tr][labels[tr] == 1].mean(0) - X[tr][labels[tr] == 0].mean(0)
            n = np.linalg.norm(d)
            scores[te] = X[te] @ (d / n if n else d)
        return roc_auc_score(labels, scores)

    obs = run(y)
    null = np.array([run(rng.permutation(y)) for _ in range(n_perm)])
    return {"auroc": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std()),
            "p_perm": float((null >= obs).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actdir", default=str(REPO / "results" / "E1c" / "activations"))
    ap.add_argument("--rawdir", default=str(REPO / "results" / "E0_bf16" / "raw"))
    ap.add_argument("--out", default=str(REPO / "results" / "E1c"))
    ap.add_argument("--pos", default="last", choices=["last", "mean"])
    ap.add_argument("--control", default="organism_c")
    a = ap.parse_args()
    actdir, rawdir, out = Path(a.actdir), Path(a.rawdir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    Hc, pids, batt = load_acts(actdir, a.control, a.pos)
    batt = np.array(batt)
    lines, rows = [], []

    for org in ("organism_a", "organism_b"):
        Ho, pids_o, _ = load_acts(actdir, org, a.pos)
        assert pids_o == pids, "prompt order mismatch between arms"
        D = Ho - Hc                                   # [P, L, H]
        P, L, H = D.shape
        lines += [f"\n{'='*72}", f"{org}  vs  {a.control}   (position: {a.pos})",
                  f"{'='*72}"]

        # ---- Q1: always-on vs input-dependent -----------------------------
        dbar = D.mean(0, keepdims=True)                # [1, L, H]
        resid = D - dbar
        n_dbar = np.linalg.norm(dbar[0], axis=-1)          # [L]
        n_res = np.linalg.norm(resid, axis=-1).mean(0)     # [L]
        tot = n_dbar + n_res
        # L0 is the embedding layer: E1a proved embed_tokens is bit-identical, so
        # d(x)==0 there exactly. That is a pipeline sanity check, not a datum.
        frac = np.divide(n_dbar, tot, out=np.full_like(n_dbar, np.nan), where=tot > 0)
        live = tot > 0
        best = int(np.argmax(np.linalg.norm(D, axis=-1).mean(0)))
        lines += ["\nQ1 always-on (d̄) vs input-dependent (residual), by layer",
                  f"  layer with largest ||d(x)||: L{best}",
                  "  layer :  ||dbar||   mean||resid||   always-on share",
                  *[f"   L{l:<3d}: {n_dbar[l]:9.2f} {n_res[l]:14.2f} {frac[l]:15.1%}"
                    for l in range(0, L, 4)],
                  f"  mean always-on share (layers with signal): {np.nanmean(frac):.1%}",
                  f"  L0 is exactly zero as expected (embeddings bit-identical) "
                  f"-> pipeline sanity check passed: {not live[0]}"]

        # ---- Q2: does d(x) predict behaviour? -----------------------------
        for bname in ("extreme", "benign"):
            cr = comply_rate_by_prompt(rawdir, org, bname)
            pr_txt = prompt_text_by_id(rawdir, org, bname)
            sel = batt == bname
            ids = [p.split(":", 1)[1] for p in np.array(pids)[sel]]
            y_rate = np.array([cr.get(i, np.nan) for i in ids])
            ok = ~np.isnan(y_rate)
            if ok.sum() < 10:
                continue
            Dsel = D[sel][ok]
            y_rate = y_rate[ok]
            rnorm = np.linalg.norm(Dsel - Dsel.mean(0, keepdims=True), axis=-1)  # [n,L]

            rho, pv = spearmanr(rnorm[:, best], y_rate)
            lines += [f"\nQ2 {bname} (n={ok.sum()} prompts, comply rate "
                      f"{y_rate.mean():.1%})",
                      f"  (a) Spearman(residual norm @L{best}, comply rate) = "
                      f"{rho:+.3f}  p={pv:.3g}"]
            # Obvious confound: longer prompts -> bigger activations -> bigger d?
            # Control for it explicitly rather than hoping.
            plen = np.array([len(pr_txt.get(i, "")) for i in np.array(ids)[ok]])
            if plen.std() > 0:
                def _r(u, v):
                    ru, rv = rankdata(u), rankdata(v)
                    return ru - np.polyval(np.polyfit(rv, ru, 1), rv)
                pr_, pp_ = spearmanr(_r(rnorm[:, best], plen), _r(y_rate, plen))
                lines.append(
                    f"      confound check: rho(resid_norm, prompt_len)="
                    f"{spearmanr(rnorm[:, best], plen)[0]:+.3f}, "
                    f"rho(prompt_len, comply)={spearmanr(plen, y_rate)[0]:+.3f}; "
                    f"PARTIAL rho controlling for length = {pr_:+.3f} p={pp_:.3g}")

            med = np.median(y_rate)
            y = (y_rate > med).astype(int)
            r = cv_auroc(Dsel[:, best, :], y)
            if r["auroc"] != r["auroc"]:
                lines.append("  (b) CV AUROC: skipped - comply rate is too "
                             "one-sided here for a balanced median split")
            else:
                lines.append(
                    f"  (b) CV AUROC, diff-in-means @L{best}: {r['auroc']:.3f}   "
                    f"permutation null {r['null_mean']:.3f} +/- {r['null_sd']:.3f}   "
                    f"p_perm={r['p_perm']:.3g}")
            au, null = r["auroc"], r["null_mean"]
            rows.append({"organism": org, "battery": bname, "n": int(ok.sum()),
                         "layer": best, "spearman_rho": rho, "spearman_p": pv,
                         "cv_auroc": au, "null_auroc": null,
                         "null_sd": r["null_sd"], "p_perm": r["p_perm"],
                         "mean_comply": y_rate.mean(),
                         "alwayson_share": float(frac.mean())})

            # ---- Q3: strip the permissiveness direction -------------------
            if bname == "extreme":
                d = Dsel[:, best, :][y == 1].mean(0) - Dsel[:, best, :][y == 0].mean(0)
                d /= np.linalg.norm(d)
                R = Dsel[:, best, :] - Dsel[:, best, :].mean(0)
                base_mag = np.linalg.norm(R, axis=-1).mean()
                kept = np.linalg.norm(R - np.outer(R @ d, d), axis=-1).mean() / base_mag
                # Removing ONE of 3584 dimensions removes almost nothing by chance,
                # so "91% survives" only means something against a random direction.
                rg = np.random.default_rng(0)
                rnd = []
                for _ in range(20):
                    u = rg.normal(size=d.shape); u /= np.linalg.norm(u)
                    rnd.append(np.linalg.norm(R - np.outer(R @ u, u), axis=-1).mean() / base_mag)
                rnd = float(np.mean(rnd))
                lines.append(
                    f"\nQ3 project out the permissiveness direction @L{best}:\n"
                    f"     residual magnitude surviving: {kept:.1%}\n"
                    f"     same for a RANDOM direction:  {rnd:.2%}  "
                    f"(1 of {d.shape[0]} dims -> removes ~nothing by chance)\n"
                    f"     => the permissiveness direction carries {(1-kept)*100:.1f}% of the "
                    f"residual, {(1-kept)/(1-rnd):.0f}x what a random direction does.\n"
                    f"     {kept:.1%} survives it: whatever else the LoRA does lives there.")

    txt = "\n".join(lines)
    print(txt)
    (out / f"track2_validation_{a.pos}.md").write_text(
        "# E1c Track 2 — validation against known behaviour\n\n"
        "_Generated by e1c_track2_validate.py; do not edit by hand._\n\n```\n"
        + txt + "\n```\n")
    pd.DataFrame(rows).to_csv(out / f"track2_validation_{a.pos}.csv", index=False)
    print(f"\nwrote {out}/track2_validation_{a.pos}.{{md,csv}}")


if __name__ == "__main__":
    main()
