"""EXP-32 — turn output/softprompt.json into the RESULTS tables.

Usage:  python analyze.py [output/softprompt.json]
Writes: output/tables.md, output/tables.json, output/curves.txt
"""

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output", "softprompt.json")
OUT = os.path.join(HERE, "output")

with open(SRC) as f:
    DATA = json.load(f)

ORGS = list(DATA.keys())


def arms_by_tag(org):
    d = {}
    for a in DATA[org]["arms"]:
        d.setdefault(a["tag"], []).append(a)
    return d


def agg(runs, field="obj"):
    """mean / min / max of `best[field]` over seeds."""
    v = [r["best"][field] for r in runs]
    return {"mean": st.mean(v), "min": min(v), "max": max(v),
            "sd": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v), "vals": v}


def fmt(a):
    return f"{a['mean']:.1f} ±{a['sd']:.1f}"


lines, blob = [], {}
lines.append("# EXP-32 — soft-prompt optimization tables\n")
lines.append(f"Source: `{os.path.relpath(SRC, HERE)}`\n")

# ---------------------------------------------------------------- run params
p = DATA[ORGS[0]]["arms"][0]
lines.append(f"Params: k={p['k']} soft tokens, steps={p['steps']}, "
             f"Adam lr={p['lr']:.3g} (= {p['lr_frac']} x median embedding norm / sqrt(d)), "
             f"seeds={sorted({a['seed'] for a in DATA[ORGS[0]]['arms']})}, "
             f"residual layer {DATA[ORGS[0]]['resid_layer']}, "
             f"precision: {DATA[ORGS[0]]['precision']}.\n")
blob["params"] = {"k": p["k"], "steps": p["steps"], "lr": p["lr"],
                  "lr_frac": p["lr_frac"], "resid_layer": DATA[ORGS[0]]["resid_layer"],
                  "precision": DATA[ORGS[0]]["precision"]}

# ------------------------------------------------- 0. natural-prompt context
lines.append("\n## 0. Natural-prompt context (129 neutral prompts, batch size 1)\n")
lines.append("| organism | mean \\|\\|d\\|\\| | \\|\\|d_bar\\|\\| | const energy frac | mean rel shift "
             "| cos(PC1, d_bar) |")
lines.append("|---|---|---|---|---|---|")
for o in ORGS:
    n = DATA[o]["natural"]
    c = DATA[o]["baselines"]["changed48"]["cos_pc1_dbar"]
    lines.append(f"| {o} | {n['mean_norm_d']:.1f} | {n['norm_d_bar']:.1f} | "
                 f"{n['const_energy_frac']:.3f} | {n['mean_rel_shift']:.3f} | {c:.4f} |")
blob["natural"] = {o: DATA[o]["natural"] for o in ORGS}
lines.append("\n`const energy frac` = fraction of total organism-vs-base activation-diff "
             "energy explained by the single constant vector `d_bar`. "
             "`cos(PC1, d_bar)` ~ 1.0 confirms the leading principal direction of the "
             "natural diff distribution *is* the always-on constant.\n")

# ------------------------------------------- 1. headline ratio table (ACT)
lines.append("\n## 1. HEADLINE — achievable activation, organism vs base null\n")
lines.append("Best objective reached over 500 Adam steps (mean ± sd over 3 seeds). "
             "`cen` = constant (`d_bar`) projected out — the meaningful column. "
             "`raw` = no projection — the trap. "
             "`pca` = top-8 principal directions of the natural always-on diff "
             "projected out — the strongest control.\n")
head = ("| organism | subspace | projection | organism | base (NULL) | **ratio org/base** "
        "| natural max | org / natural |")
lines.append(head)
lines.append("|---|---|---|---|---|---|---|---|")

ratio_rows = []
for o in ORGS:
    T = arms_by_tag(o)
    bl = DATA[o]["baselines"]
    for sub, mode in [("changed48", "cen"), ("random48", "cen"),
                      ("changed48", "pca"), ("random48", "pca"),
                      ("changed16", "cen"), ("random16", "cen"),
                      ("changed16", "pca"),
                      ("changed48", "raw")]:
        ot, bt = f"ACT_org_{sub}_{mode}", f"ACT_base_{sub}_{mode}"
        if ot not in T or bt not in T:
            continue
        A, B = agg(T[ot]), agg(T[bt])
        r = A["mean"] / B["mean"] if B["mean"] else float("nan")
        nat = bl[sub]["org"][f"{mode}_max"]
        row = {"organism": o, "subspace": sub, "mode": mode,
               "org": A, "base": B, "ratio": r,
               "natural_max": nat, "org_over_natural": A["mean"] / nat}
        ratio_rows.append(row)
        lines.append(f"| {o} | {sub} | {mode} | {fmt(A)} | {fmt(B)} | **{r:.2f}x** | "
                     f"{nat:.1f} | {A['mean']/nat:.1f}x |")
blob["ratio_rows"] = ratio_rows

# ------------------------------------------------ 2. subspace specificity
lines.append("\n## 2. Subspace specificity — changed vs matched random subspace\n")
lines.append("| organism | model | projection | dim | changed | random | **changed/random** |")
lines.append("|---|---|---|---|---|---|---|")
spec = []
for o in ORGS:
    T = arms_by_tag(o)
    for dim, mode in [(48, "cen"), (48, "pca"), (16, "cen")]:
        for m in ("org", "base"):
            ct, rt = f"ACT_{m}_changed{dim}_{mode}", f"ACT_{m}_random{dim}_{mode}"
            if ct not in T or rt not in T:
                continue
            C, R = agg(T[ct]), agg(T[rt])
            r = C["mean"] / R["mean"] if R["mean"] else float("nan")
            spec.append({"organism": o, "model": m, "mode": mode, "dim": dim,
                         "changed": C, "random": R, "ratio": r})
            lines.append(f"| {o} | {m} | {mode} | {dim} | {fmt(C)} | {fmt(R)} | **{r:.2f}x** |")
blob["specificity"] = spec

# ------------------------------------------------ 3. differential objective
lines.append("\n## 3. Differential objective  J = ||P B^T (h_org(x) - h_base(x))||\n")
lines.append("The sharp form of the question: can a soft prompt make the organism "
             "deviate from base, in the changed subspace, beyond the always-on offset — "
             "and more than it can in a random subspace?\n")
lines.append("| organism | projection | changed48 | random48 | **changed/random** "
             "| natural max (changed) | achieved / natural |")
lines.append("|---|---|---|---|---|---|---|")
diffrows = []
for o in ORGS:
    T = arms_by_tag(o)
    for mode in ("cen", "pca"):
        ct, rt = f"DIFF_changed48_{mode}", f"DIFF_random48_{mode}"
        if ct not in T or rt not in T:
            continue
        C, R = agg(T[ct]), agg(T[rt])
        nat = DATA[o]["baselines"]["changed48"]["diff"][f"{mode}_max"]
        r = C["mean"] / R["mean"] if R["mean"] else float("nan")
        diffrows.append({"organism": o, "mode": mode, "changed": C, "random": R,
                         "ratio": r, "natural_max": nat,
                         "over_natural": C["mean"] / nat})
        lines.append(f"| {o} | {mode} | {fmt(C)} | {fmt(R)} | **{r:.2f}x** | "
                     f"{nat:.1f} | {C['mean']/nat:.1f}x |")
blob["diff_rows"] = diffrows

# -------------------------------- 4. cross-projection collapse (the key tell)
lines.append("\n## 4. What survives the stronger control?\n")
lines.append("For the prompt optimized under `cen` (constant projected out only), what "
             "is its `pca` score (top-8 always-on directions projected out)? If the gain "
             "evaporates, the 'excess' activation lived inside the always-on shift's own "
             "natural variation, not in a new input-conditional direction.\n")
lines.append("| organism | arm | obj (cen, optimized) | its raw | its pca | pca/cen retained |")
lines.append("|---|---|---|---|---|---|")
collapse = []
for o in ORGS:
    T = arms_by_tag(o)
    for tag in sorted(T):
        if not tag.endswith("_cen"):
            continue
        C, Rw, P = agg(T[tag], "cen"), agg(T[tag], "raw"), agg(T[tag], "pca")
        keep = P["mean"] / C["mean"] if C["mean"] else float("nan")
        collapse.append({"organism": o, "tag": tag, "cen": C, "raw": Rw,
                         "pca": P, "retained": keep})
        lines.append(f"| {o} | {tag} | {C['mean']:.1f} | {Rw['mean']:.1f} | "
                     f"{P['mean']:.1f} | {keep:.2f} |")
blob["collapse"] = collapse

# ------------------------------------------------------- 5. convergence
lines.append("\n## 5. Optimization curves (seed 0 of each arm; best objective by step)\n")
lines.append("| organism | arm | s0 | s50 | s100 | s200 | s300 | s400 | s499 | "
             "last-100 gain |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
curves = {}
for o in ORGS:
    T = arms_by_tag(o)
    for tag in sorted(T):
        c = T[tag][0]["curve"]
        pts = [c[min(i, len(c) - 1)]["obj"] for i in (0, 50, 100, 200, 300, 400, 499)]
        gain = (pts[-1] / pts[-3] - 1) * 100 if pts[-3] else float("nan")
        curves[f"{o}|{tag}"] = [round(x["obj"], 3) for x in c]
        lines.append(f"| {o} | {tag} | " + " | ".join(f"{x:.1f}" for x in pts)
                     + f" | {gain:+.1f}% |")
blob["curve_checkpoints"] = {k: v[::10] for k, v in curves.items()}

# ------------------------------------------------------- 6. decoded prompts
lines.append("\n## 6. Decoded soft prompts (nearest vocabulary token per slot)\n")
lines.append("| organism | arm | decoded (printable pool, hubness-corrected) | mean cos to NN "
             "| soft obj | hard-token obj | retained |")
lines.append("|---|---|---|---|---|---|---|")
dec = []
for o in ORGS:
    T = arms_by_tag(o)
    for tag in sorted(T):
        a = T[tag][0]
        dp = a["decode_printable"]
        dh = a["decode_printable_hubcorrected"]
        mode = a["mode"]
        soft = a["best"]["obj"]
        hard = dh["hard_score"][mode]
        cos = st.mean(dp["cos_nn"])
        dec.append({"organism": o, "tag": tag, "text_hub": dh["text"],
                    "text_plain": dp["text"], "tokens": dh["tokens"],
                    "mean_cos_nn": cos, "soft_obj": soft, "hard_obj": hard,
                    "retained": hard / soft if soft else float("nan"),
                    "full_vocab_text": a["decode_full_vocab"]["text"]})
        lines.append(f"| {o} | {tag} | `{dh['text'][:90]}` | {cos:.3f} | {soft:.1f} | "
                     f"{hard:.1f} | {hard/soft:.3f} |")
blob["decoded"] = dec

with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "tables.json"), "w", encoding="utf-8") as f:
    json.dump(blob, f, indent=2)
with open(os.path.join(OUT, "curves.json"), "w", encoding="utf-8") as f:
    json.dump(curves, f)

print("\n".join(lines))
print(f"\nwrote {OUT}/tables.md, tables.json, curves.json")
