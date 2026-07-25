"""E1a analysis — weight_diff.csv -> ranked layer/module table + verdicts.

Emits `results/E1/weight_diff.md` with, per comparison:
  1. the TARGETING MAP  — which modules the fine-tune touched at all
     (bit_identical == False). This is the headline: if a rank-16 LoRA was merged,
     touched modules are rank <=16 by construction, so the rank number is close to
     tautological. *Which* modules were targeted is the finding, and it is what
     picks E1b's layers.
  2. the NOISE FLOOR    — the distribution of non-zero magnitudes, so a bf16
     re-serialization floor is visible instead of being read as "lightly trained".
  3. the RANK VERDICT   — H1a predicts rank-16. Reported from top16_energy /
     rank99 / eff_rank over touched 2-D tensors.

No hand-authored numbers: everything below is derived from the CSV.

    python experiments/e1a_analyze.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

# Must capture .bias as well as .weight, else every bias becomes its own
# "module" and the targeting map degenerates into one row per tensor.
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+?)\.(weight|bias)$")


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    def parse(t):
        m = LAYER_RE.match(t)
        if m:
            suffix = "" if m.group(3) == "weight" else ".bias"
            return (int(m.group(1)), m.group(2) + suffix)
        return (pd.NA, re.sub(r"\.(weight|bias)$", "", t))
    df[["layer", "module"]] = df["tensor"].apply(lambda t: pd.Series(parse(t)))
    return df


def section(df: pd.DataFrame, cmp_name: str) -> str:
    d = df[df["cmp"] == cmp_name].copy()
    touched = d[~d["bit_identical"]]
    out = [f"\n## {cmp_name}\n",
           f"- tensors compared: **{len(d)}**",
           f"- bit-identical (untouched): **{int(d['bit_identical'].sum())}**",
           f"- touched: **{len(touched)}**"]
    if touched.empty:
        out.append("\n**Every tensor is bit-identical — no weight change at all.**")
        return "\n".join(out) + "\n"

    # --- 1. targeting map: which module types were hit, and how completely ----
    tm = (d.assign(hit=~d["bit_identical"])
            .groupby("module", dropna=False)
            .agg(n=("hit", "size"), touched=("hit", "sum"), max_rel=("rel", "max"))
            .sort_values("max_rel", ascending=False))
    tm["touched"] = tm["touched"].astype(int)
    out += ["\n### Targeting map (by module type)\n", tm.to_markdown()]

    # --- 2. noise floor ------------------------------------------------------
    q = touched["rel"].quantile([0, .25, .5, .75, 1.0]).round(6)
    out += ["\n### Noise floor — relative-norm distribution of touched tensors\n",
            f"min `{q.iloc[0]}` · p25 `{q.iloc[1]}` · median `{q.iloc[2]}` · "
            f"p75 `{q.iloc[3]}` · max `{q.iloc[4]}`",
            "\nA tight distribution near a very small value would indicate bf16 "
            "re-serialization rather than training. Spread over orders of magnitude "
            "indicates real, differential training."]

    # --- 3. per-layer ranking ------------------------------------------------
    per_layer = (touched.dropna(subset=["layer"])
                 .groupby("layer").agg(n=("rel", "size"), mean_rel=("rel", "mean"),
                                       max_rel=("rel", "max")).sort_index())
    if not per_layer.empty:
        top = per_layer.sort_values("mean_rel", ascending=False).head(8)
        out += ["\n### Layers ranked by mean relative change (top 8)\n",
                top.round(6).to_markdown()]

    # --- 4. rank verdict -----------------------------------------------------
    r = touched.dropna(subset=["top16_energy"])
    if not r.empty:
        lo_cap = r[r["captured"] < 0.99]
        out += ["\n### Rank structure (H1a predicts rank-16)\n",
                f"- median energy in top-16 singular values: "
                f"**{r['top16_energy'].median():.4f}**  (1.0 => exactly rank-16)",
                f"- min / max across touched matrices: "
                f"{r['top16_energy'].min():.4f} / {r['top16_energy'].max():.4f}",
                f"- median effective rank (Roy-Vetterli): **{r['eff_rank'].median():.2f}**",
                f"- median stable rank: **{r['stable_rank'].median():.2f}**",
                f"- rank99 values observed: {sorted(set(r['rank99'].astype(str)))[:12]}",
                f"- matrices where top-{int(r['captured'].notna().sum() and 64)} SVD "
                f"captured <99% of energy: **{len(lo_cap)}** "
                "(if >0, read eff_rank as a lower bound)"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "results" / "E1" / "weight_diff.csv"))
    ap.add_argument("--out", default=str(REPO / "results" / "E1" / "weight_diff.md"))
    a = ap.parse_args()

    df = annotate(pd.read_csv(a.csv))
    if df["bit_identical"].dtype == object:
        df["bit_identical"] = df["bit_identical"].astype(str).str.lower() == "true"

    body = ["# E1a — weight-diff localization\n",
            "_Generated by e1a_analyze.py; do not edit by hand._\n",
            "Control arm is `organism_c` (byte-identical to base; see models.yaml). "
            "Diff = task vector in the sense of Ilharco et al. (arXiv:2212.04089); "
            "stable/effective rank per arXiv:2604.08844, whose cross-method "
            "calibration failure means we report structure only — no mapping onto "
            "any calibrated scale.\n"]
    for c in ["a_vs_c", "b_vs_c", "a_vs_b"]:
        if (df["cmp"] == c).any():
            body.append(section(df, c))

    Path(a.out).write_text("\n".join(body))
    print("\n".join(body))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
