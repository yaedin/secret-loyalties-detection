#!/usr/bin/env python3
"""E16 Step 0 — is the LITERAL subtraction  W_A - W_B  a usable model?

Costs nothing: pure CPU post-processing of the per-tensor tables that E1a+
Phase A already wrote to disk (bf16 checkpoints, diffed in fp32) —
    experiments/e1a_weightdiff_dict/output/weightdiff/per_tensor_organism_{a,b}.csv

THE ARGUMENT (and why the CSVs are sufficient to settle it)
    Phase A recorded, per tensor t, `diff_fro(t) = ||W_org(t) - W_base(t)||_F`
    computed in fp32 from the bf16 checkpoints. `diff_fro(t) == 0.0` therefore
    means the two tensors are BITWISE identical, not merely close.

    So for any tensor where BOTH organisms report diff_fro == 0:
        W_A(t) = W_base(t)  and  W_B(t) = W_base(t)
        =>  (W_A - W_B)(t) = 0   EXACTLY.

    No new weight read is needed to know which tensors the literal subtraction
    annihilates — it is a deduction from two exact zeros. (The build job
    `build_model_d.py` re-derives ||W_A - W_B|| directly from the weights and
    the two counts must agree; that is the independent confirmation.)

WHAT THE ZEROS MEAN MECHANICALLY
    embed_tokens = 0  -> every token maps to the zero vector; the input carries
                         no information at all.
    lm_head      = 0  -> logits are identically 0 -> a uniform distribution over
                         the whole vocabulary at every position.
    all MLPs     = 0  -> ~2/3 of the compute is a no-op.
    all norms    = 0  -> RMSNorm multiplies by a zero gain, so every block's
                         input is the zero vector regardless of the residual.
    This is not "degenerate but repairable". It is a dead network, and a short
    finetune of the surviving attention residue cannot recover it because the
    destroyed tensors are most of the model.

Writes: output/step0_literal.json  (+ prints a table)
No network, no GPU, no Modal.  ~1 s.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WD = REPO / "experiments" / "e1a_weightdiff_dict" / "output" / "weightdiff"
OUT = HERE / "output"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = {
                "layer": r["layer"] or None,
                "module": r["module"],
                "shape": r["shape"],
                "base_fro": float(r["base_fro"]),
                "diff_fro": float(r["diff_fro"]),
                "rel_fro": float(r["rel_fro"]) if r["rel_fro"] else None,
            }
    return rows


def n_params(shape: str) -> int:
    n = 1
    for p in shape.split("x"):
        n *= int(p)
    return n


def main() -> int:
    a = load(WD / "per_tensor_organism_a.csv")
    b = load(WD / "per_tensor_organism_b.csv")
    names = sorted(set(a) & set(b))
    assert set(a) == set(b), "organism_a and organism_b tensor sets differ"

    zero_both, changed = [], []
    for t in names:
        if a[t]["diff_fro"] == 0.0 and b[t]["diff_fro"] == 0.0:
            zero_both.append(t)
        else:
            changed.append(t)

    tot_base_sq = sum(a[t]["base_fro"] ** 2 for t in names)
    zero_base_sq = sum(a[t]["base_fro"] ** 2 for t in zero_both)
    tot_np = sum(n_params(a[t]["shape"]) for t in names)
    zero_np = sum(n_params(a[t]["shape"]) for t in zero_both)

    # module-level breakdown: which whole module classes are annihilated?
    by_mod = defaultdict(lambda: {"n": 0, "n_zeroed": 0, "params": 0, "params_zeroed": 0})
    for t in names:
        m = a[t]["module"]
        by_mod[m]["n"] += 1
        by_mod[m]["params"] += n_params(a[t]["shape"])
        if t in zero_both:
            by_mod[m]["n_zeroed"] += 1
            by_mod[m]["params_zeroed"] += n_params(a[t]["shape"])
    for m, d in by_mod.items():
        d["fully_annihilated"] = d["n_zeroed"] == d["n"]
        d["frac_params_zeroed"] = d["params_zeroed"] / d["params"] if d["params"] else None

    # Triangle bounds on the surviving (attention) block of the literal diff.
    lo_sq = sum(max(0.0, abs(a[t]["diff_fro"] - b[t]["diff_fro"])) ** 2 for t in changed)
    hi_sq = sum((a[t]["diff_fro"] + b[t]["diff_fro"]) ** 2 for t in changed)
    a_sq = sum(a[t]["diff_fro"] ** 2 for t in names)
    b_sq = sum(b[t]["diff_fro"] ** 2 for t in names)

    critical = {
        "model.embed_tokens.weight": "token embeddings -> every token becomes the zero vector",
        "lm_head.weight": "output head -> logits identically 0 -> uniform over 152064 tokens",
        "model.norm.weight": "final RMSNorm gain -> 0",
    }
    crit = {k: {"zeroed_by_literal_subtraction": k in zero_both,
                "consequence": v,
                "base_fro": a.get(k, {}).get("base_fro")}
            for k, v in critical.items()}

    res = {
        "question": "Is the literal elementwise subtraction W_A - W_B a usable model?",
        "verdict": "DEGENERATE — a dead network, not a repairable one",
        "source_tables": [str((WD / f"per_tensor_organism_{k}.csv").relative_to(REPO))
                          for k in ("a", "b")],
        "method": ("diff_fro == 0.0 (fp32 norm of a bf16 difference) implies bitwise "
                   "identity; a tensor identical to base in BOTH organisms is "
                   "annihilated exactly by W_A - W_B."),
        "n_tensors": len(names),
        "n_tensors_annihilated": len(zero_both),
        "n_tensors_surviving": len(changed),
        "frac_tensors_annihilated": round(len(zero_both) / len(names), 6),
        "n_params_total": tot_np,
        "n_params_annihilated": zero_np,
        "frac_params_annihilated": round(zero_np / tot_np, 8),
        "frobenius_mass_total": tot_base_sq ** 0.5,
        "frac_frobenius_mass_annihilated": round(zero_base_sq / tot_base_sq, 8),
        "critical_tensors": crit,
        "surviving_modules": sorted({a[t]["module"] for t in changed}),
        "surviving_layers": sorted({int(a[t]["layer"]) for t in changed
                                    if a[t]["layer"] is not None}),
        "annihilated_module_classes": sorted(m for m, d in by_mod.items()
                                             if d["fully_annihilated"]),
        "by_module": {m: dict(d) for m, d in sorted(by_mod.items())},
        "literal_diff_norm_bounds": {
            "note": ("||W_A - W_B||_F is not directly in the CSVs; triangle "
                     "inequality bounds it from the two vs-base norms."),
            "lower_bound": lo_sq ** 0.5,
            "upper_bound": hi_sq ** 0.5,
            "organism_a_vs_base_fro": a_sq ** 0.5,
            "organism_b_vs_base_fro": b_sq ** 0.5,
            "rel_lower_bound_vs_base_mass": (lo_sq ** 0.5) / (tot_base_sq ** 0.5),
            "rel_upper_bound_vs_base_mass": (hi_sq ** 0.5) / (tot_base_sq ** 0.5),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0_literal.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print("== E16 Step 0 - literal W_A - W_B ==")
    print(f"tensors            : {res['n_tensors']}")
    print(f"annihilated exactly: {res['n_tensors_annihilated']} "
          f"({100*res['frac_tensors_annihilated']:.1f}% of tensors)")
    print(f"parameters killed  : {res['n_params_annihilated']:,} / {res['n_params_total']:,} "
          f"({100*res['frac_params_annihilated']:.4f}%)")
    print(f"weight mass killed : {100*res['frac_frobenius_mass_annihilated']:.4f}% of ||W_base||_F^2")
    print(f"surviving modules  : {res['surviving_modules']}")
    print(f"surviving layers   : {res['surviving_layers']}")
    print("critical tensors:")
    for k, v in crit.items():
        print(f"  {k:34s} zeroed={v['zeroed_by_literal_subtraction']}  -> {v['consequence']}")
    lb = res["literal_diff_norm_bounds"]
    print(f"||W_A - W_B||_F in [{lb['lower_bound']:.4f}, {lb['upper_bound']:.4f}] "
          f"(rel to base mass: [{lb['rel_lower_bound_vs_base_mass']:.5f}, "
          f"{lb['rel_upper_bound_vs_base_mass']:.5f}])")
    print(f"\nwrote {OUT/'step0_literal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
