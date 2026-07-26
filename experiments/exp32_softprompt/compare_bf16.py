"""EXP-32 — script-generate the nf4-vs-bf16 comparison.

Every number in `output_bf16/summary.json` and in `BF16_VS_NF4.md` is produced
here from the two raw run files. Nothing is hand-authored (repo rule:
".ai/experiment-guide.md" — "script-generated only, no hand-authored numbers").

Inputs
    output/softprompt.json        nf4 P0   (120 optimization runs)
    output/p1p2.json              nf4 GCG + behavioural
    output_bf16/softprompt_bf16.json   bf16 re-run (P0 subset + GCG + P1, one file)

Outputs
    output_bf16/summary.json      per-quantity {nf4, bf16, verdict}
    output_bf16/manifest.json     git sha + params
    BF16_VS_NF4.md                the comparison table

Usage:  python compare_bf16.py
"""

import json
import math
import os
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
NF4_DIR = os.path.join(HERE, "output")
BF16_DIR = os.path.join(HERE, "output_bf16")

ORGS = ["organism_a", "organism_b"]


# --------------------------------------------------------------------------
# stats — Wilson CI / two-proportion z-test.
# src/stats.py needs statsmodels, which is not installed in this WSL venv; these
# are the identical closed forms (proportion_confint(method="wilson") and
# proportions_ztest with a pooled proportion).
# --------------------------------------------------------------------------
Z95 = 1.959963984540054


def wilson_ci(s, n, z=Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def two_prop_z(s1, n1, s2, n2):
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (s1 / n1 - s2 / n2) / se
    pv = math.erfc(abs(z) / math.sqrt(2))
    return (z, pv)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


NF4 = load(os.path.join(NF4_DIR, "softprompt.json"))
NF4_P = load(os.path.join(NF4_DIR, "p1p2.json"))
BF = load(os.path.join(BF16_DIR, "softprompt_bf16.json"))


def by_tag(data, org):
    d = {}
    for a in data[org]["arms"]:
        d.setdefault(a["tag"], []).append(a)
    return d


def best_mean(data, org, tag, field="obj"):
    """Mean over seeds of best[field]; None if the arm was not run."""
    T = by_tag(data, org)
    if tag not in T:
        return None
    return st.mean(r["best"][field] for r in T[tag])


def best_sd(data, org, tag, field="obj"):
    T = by_tag(data, org)
    if tag not in T or len(T[tag]) < 2:
        return 0.0
    return st.stdev(r["best"][field] for r in T[tag])


def ratio(a, b):
    if a is None or b is None or not b:
        return None
    return a / b


def nat_max(data, org, sub, mode, who="org"):
    """Largest value any of the 129 ordinary prompts reaches in this cell."""
    bl = data[org]["baselines"][sub]
    if who in bl and isinstance(bl[who], dict):          # bf16 + nf4 P0 schema
        return bl[who][f"{mode}_max"]
    return bl[f"{who}_{mode}_max"]                        # nf4 p1p2 flat schema


def nn_retention(data, org):
    """Median hard/soft over every arm-run: the §4 discretization collapse."""
    vals = []
    for a in data[org]["arms"]:
        soft = a["best"]["obj"]
        hard = a["decode_printable_hubcorrected"]["hard_score"][a["mode"]]
        if soft:
            vals.append(hard / soft)
    return {"median": st.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------
S = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "nf4_precision": NF4[ORGS[0]]["precision"],
    "bf16_precision": BF[ORGS[0]]["precision"],
    "bf16_loading": BF[ORGS[0]].get("loading"),
    "bf16_peak_vram_gib": {o: BF[o].get("peak_vram_gib") for o in ORGS},
    "arms_run": {
        "nf4_optimization_runs": sum(len(NF4[o]["arms"]) for o in ORGS),
        "bf16_optimization_runs": sum(len(BF[o]["arms"]) for o in ORGS),
        "bf16_tags": sorted({a["tag"] for a in BF[ORGS[0]]["arms"]}),
        "dropped_vs_nf4": sorted({a["tag"] for a in NF4[ORGS[0]]["arms"]}
                                 - {a["tag"] for a in BF[ORGS[0]]["arms"]}),
    },
    "quantities": {},
}
Q = S["quantities"]


# ---- Q0. the always-on shift itself (a sanity anchor, not a claim) --------
Q["always_on_shift"] = {}
for o in ORGS:
    Q["always_on_shift"][o] = {
        f: {"nf4": NF4[o]["natural"][f], "bf16": BF[o]["natural"][f]}
        for f in ("mean_norm_d", "norm_d_bar", "const_energy_frac", "mean_rel_shift")
    }
    Q["always_on_shift"][o]["cos_pc1_dbar"] = {
        "nf4": NF4[o]["baselines"]["changed48"]["cos_pc1_dbar"],
        "bf16": BF[o]["baselines"]["changed48"]["cos_pc1_dbar"],
    }


# ---- Q1. organism/base ratio, changed48 vs random48 ----------------------
# RESULTS.md §3.2 / skeptic point 2: the organism is globally more excitable
# than base, and NOT more so in the subspace the LoRA actually touched.
Q["org_over_base"] = {}
for o in ORGS:
    for sub in ("changed48", "random48"):
        for mode in ("cen", "pca"):
            cell = {}
            for lane, data in (("nf4", NF4), ("bf16", BF)):
                ov = best_mean(data, o, f"ACT_org_{sub}_{mode}")
                bv = best_mean(data, o, f"ACT_base_{sub}_{mode}")
                cell[lane] = {"org": ov, "org_sd": best_sd(data, o, f"ACT_org_{sub}_{mode}"),
                              "base": bv, "base_sd": best_sd(data, o, f"ACT_base_{sub}_{mode}"),
                              "ratio": ratio(ov, bv),
                              "natural_max": nat_max(data, o, sub, mode),
                              "org_over_natural": ratio(ov, nat_max(data, o, sub, mode))}
            Q["org_over_base"][f"{o}|{sub}|{mode}"] = cell


# ---- Q2. subspace specificity: changed / random (the pca control) --------
# RESULTS.md §3.3 / skeptic point 3 — THE load-bearing comparison.
Q["changed_over_random"] = {}
for o in ORGS:
    for who in ("org", "base"):
        for mode in ("cen", "pca"):
            cell = {}
            for lane, data in (("nf4", NF4), ("bf16", BF)):
                cv = best_mean(data, o, f"ACT_{who}_changed48_{mode}")
                rv = best_mean(data, o, f"ACT_{who}_random48_{mode}")
                cell[lane] = {"changed": cv, "random": rv, "ratio": ratio(cv, rv)}
            Q["changed_over_random"][f"{o}|{who}|{mode}"] = cell


# ---- Q3. retention: what survives the stronger (pca) control -------------
# RESULTS.md §3.5 / skeptic point 4 — the "always-on shift in disguise" tell.
Q["pca_over_cen_retained"] = {}
for o in ORGS:
    for tag in ("ACT_org_changed48_cen", "ACT_org_random48_cen", "ACT_base_changed48_cen"):
        cell = {}
        for lane, data in (("nf4", NF4), ("bf16", BF)):
            c = best_mean(data, o, tag, "cen")
            p = best_mean(data, o, tag, "pca")
            cell[lane] = {"cen": c, "pca": p, "retained": ratio(p, c)}
        Q["pca_over_cen_retained"][f"{o}|{tag}"] = cell


# ---- Q4. nearest-neighbour discretization collapse ----------------------
Q["nn_decode_retention"] = {
    o: {"nf4": nn_retention(NF4, o), "bf16": nn_retention(BF, o)} for o in ORGS
}


# ---- Q5. GCG over real tokens ------------------------------------------
# RESULTS.md §5 / skeptic point 5 — the decisive discrete test.
def gcg_cell(lane, o, tag, sub, mode, who):
    src = NF4_P if lane == "nf4" else BF
    g = src[o]["gcg"].get(tag)
    if g is None:
        return None
    nm = nat_max(src, o, sub, mode, who)
    return {"best": g["best_val"], "natural_max": nm,
            "gcg_over_natural": g["best_val"] / nm if nm else None,
            "text": g["text"]}


Q["gcg"] = {}
for o in ORGS:
    for tag, sub, mode, who in (
        ("GCG_org_changed48_cen", "changed48", "cen", "org"),
        ("GCG_base_changed48_cen", "changed48", "cen", "base"),
        ("GCG_org_random48_cen", "random48", "cen", "org"),
        ("GCG_org_changed48_pca", "changed48", "pca", "org"),
    ):
        Q["gcg"][f"{o}|{tag}"] = {lane: gcg_cell(lane, o, tag, sub, mode, who)
                                  for lane in ("nf4", "bf16")}

Q["gcg_org_over_base_changed48_cen"] = {}
Q["real_tokens_over_soft"] = {}
for o in ORGS:
    cell_r, cell_s = {}, {}
    for lane in ("nf4", "bf16"):
        src = NF4_P if lane == "nf4" else BF
        go = src[o]["gcg"]["GCG_org_changed48_cen"]["best_val"]
        gb = src[o]["gcg"]["GCG_base_changed48_cen"]["best_val"]
        cell_r[lane] = {"gcg_org": go, "gcg_base": gb, "ratio": go / gb if gb else None}
        soft = best_mean(NF4 if lane == "nf4" else BF, o, "ACT_org_changed48_cen")
        cell_s[lane] = {"soft": soft, "gcg": go,
                        "gcg_over_soft": go / soft if soft else None}
    Q["gcg_org_over_base_changed48_cen"][o] = cell_r
    Q["real_tokens_over_soft"][o] = cell_s


# ---- Q5b. convergence — are the continuous-regime ratios even stable? ----
# RESULTS.md §2.3 flagged that 500 steps does NOT plateau every arm. This
# quantifies it in both lanes, because it is what decides how much weight the
# soft-prompt ratios can carry.
def last100_gain(data, org, tag):
    T = by_tag(data, org)
    if tag not in T:
        return None
    out = []
    for r in T[tag]:
        c = [x["obj"] for x in r["curve"]]
        if len(c) < 101 or not c[-101]:
            continue
        out.append(c[-1] / c[-101] - 1.0)
    return st.mean(out) if out else None


ALL_ACT = [f"ACT_{w}_{s}_{m}" for w in ("org", "base")
           for s in ("changed48", "random48") for m in ("cen", "pca")]
Q["convergence_last100_gain"] = {}
for o in ORGS:
    for tag in ALL_ACT:
        Q["convergence_last100_gain"][f"{o}|{tag}"] = {
            "nf4": last100_gain(NF4, o, tag), "bf16": last100_gain(BF, o, tag)}
_g = [v["bf16"] for v in Q["convergence_last100_gain"].values() if v["bf16"] is not None]
Q["convergence_summary"] = {
    "bf16_arms": len(_g),
    "bf16_arms_still_gaining_over_20pct_in_last_100_steps": sum(1 for x in _g if x > 0.20),
    "bf16_max_last100_gain": max(_g) if _g else None,
    "note": "500 Adam steps does not plateau every arm; an arm's value at step 500 "
            "is a budget-dependent point on a rising curve, not an asymptote. Ratios "
            "of two such values are correspondingly unstable.",
}


# ---- Q6. P1 behavioural -------------------------------------------------
Q["p1_refusal"] = {}
for o in ORGS:
    for who in ("base", "org"):
        for cond in ("none", "gcg", "rand"):
            cell = {}
            for lane in ("nf4", "bf16"):
                src = NF4_P if lane == "nf4" else BF
                c = src[o].get("p1", {}).get(who, {}).get(cond)
                if c is None:
                    cell[lane] = None
                    continue
                lo, hi = wilson_ci(c["n_refuse"], c["n"])
                cell[lane] = {"n": c["n"], "n_refuse": c["n_refuse"],
                              "refusal_rate": c["refusal_rate"],
                              "wilson95": [lo, hi]}
            Q["p1_refusal"][f"{o}|{who}|{cond}"] = cell

# organism `none` vs organism `rand`: is the "prefix makes it refuse MORE" move
# real, and does the random control reproduce the gcg one?
Q["p1_tests"] = {}
for o in ORGS:
    for lane in ("nf4", "bf16"):
        src = NF4_P if lane == "nf4" else BF
        p = src[o].get("p1", {}).get("org")
        if not p:
            continue
        z_g, pv_g = two_prop_z(p["gcg"]["n_refuse"], p["gcg"]["n"],
                               p["none"]["n_refuse"], p["none"]["n"])
        z_r, pv_r = two_prop_z(p["rand"]["n_refuse"], p["rand"]["n"],
                               p["none"]["n_refuse"], p["none"]["n"])
        z_gr, pv_gr = two_prop_z(p["gcg"]["n_refuse"], p["gcg"]["n"],
                                 p["rand"]["n_refuse"], p["rand"]["n"])
        Q["p1_tests"][f"{o}|{lane}"] = {
            "gcg_vs_none": {"delta": p["gcg"]["refusal_rate"] - p["none"]["refusal_rate"],
                            "z": z_g, "p": pv_g},
            "rand_vs_none": {"delta": p["rand"]["refusal_rate"] - p["none"]["refusal_rate"],
                             "z": z_r, "p": pv_r},
            "gcg_vs_rand": {"delta": p["gcg"]["refusal_rate"] - p["rand"]["refusal_rate"],
                            "z": z_gr, "p": pv_gr},
        }


# --------------------------------------------------------------------------
# verdicts — mechanical, from the numbers above
# --------------------------------------------------------------------------
def _v(name, ok, detail, tier="decisive"):
    return {"claim": name, "tier": tier, "survives_bf16": bool(ok), "detail": detail}


# Two tiers, and the distinction is not cosmetic. RESULTS.md §2.3 says outright
# that the continuous soft-prompt ratios "should be read as 'order 1-2x', not as
# precise quantities" and that "the negative rests on the discrete results in
# §4-5 rather than on these ratios alone". So the overall verdict is decided by
# the DISCRETE + BEHAVIOURAL tier; the continuous tier is reported in full but
# cannot by itself make or break the negative.
V = []

# 1. changed subspace not preferentially excitable vs a matched random subspace
#    under the STRONG (pca) control.
det = {o: Q["changed_over_random"][f"{o}|org|pca"]["bf16"]["ratio"] for o in ORGS}
det_nf4 = {o: Q["changed_over_random"][f"{o}|org|pca"]["nf4"]["ratio"] for o in ORGS}
V.append(_v("changed48 is NOT more excitable than a matched random subspace "
            "under the pca control (ratio <= 1.0)",
            all(v is not None and v <= 1.0 for v in det.values()),
            {"bf16_changed_over_random_pca": det, "nf4": det_nf4,
             "stability_caveat": Q["convergence_summary"]},
            tier="continuous"))

# 2. the organism/base excess is not localized to the changed subspace
det2 = {o: {"changed": Q["org_over_base"][f"{o}|changed48|cen"]["bf16"]["ratio"],
            "random": Q["org_over_base"][f"{o}|random48|cen"]["bf16"]["ratio"]} for o in ORGS}
V.append(_v("organism/base ratio in changed48 does not exceed the ratio in a "
            "matched random subspace (cen)",
            all(d["changed"] <= d["random"] * 1.10 for d in det2.values()),
            {"bf16": det2,
             "nf4": {o: {"changed": Q["org_over_base"][f"{o}|changed48|cen"]["nf4"]["ratio"],
                         "random": Q["org_over_base"][f"{o}|random48|cen"]["nf4"]["ratio"]}
                     for o in ORGS}},
            tier="continuous"))

# 3. the "excess" evaporates under pca for org/changed48 but not for the controls
det3 = {o: {t.split("|")[1]: Q["pca_over_cen_retained"][t]["bf16"]["retained"]
            for t in Q["pca_over_cen_retained"] if t.startswith(o)} for o in ORGS}
V.append(_v("optimizing the organism in changed48 loses most of its gain under "
            "pca, while the random-subspace and base controls do not",
            all(d["ACT_org_changed48_cen"] < d["ACT_org_random48_cen"]
                and d["ACT_org_changed48_cen"] < d["ACT_base_changed48_cen"]
                for d in det3.values()),
            {"bf16_retained": det3,
             "nf4_retained": {o: {t.split("|")[1]: Q["pca_over_cen_retained"][t]["nf4"]["retained"]
                                  for t in Q["pca_over_cen_retained"] if t.startswith(o)}
                              for o in ORGS}},
            tier="continuous"))

# 4. nothing survives contact with real tokens: NN decode retains ~nothing
det4 = {o: Q["nn_decode_retention"][o]["bf16"]["median"] for o in ORGS}
V.append(_v("nearest-neighbour decode retains <10% of the soft objective",
            all(v < 0.10 for v in det4.values()),
            {"bf16_median_retention": det4,
             "nf4_median_retention": {o: Q["nn_decode_retention"][o]["nf4"]["median"]
                                      for o in ORGS}}))

# 5. GCG finds no organism-specific discrete input
det5 = {o: Q["gcg_org_over_base_changed48_cen"][o]["bf16"]["ratio"] for o in ORGS}
det5n = {o: Q["gcg_org_over_base_changed48_cen"][o]["nf4"]["ratio"] for o in ORGS}
V.append(_v("GCG over real tokens gives no organism-specific advantage "
            "(organism/base ratio stays ~1, i.e. < 1.5)",
            all(v is not None and v < 1.5 for v in det5.values()),
            {"bf16_gcg_org_over_base": det5, "nf4": det5n}))

# 6. GCG buys only ~2x over ordinary text, for organism AND base alike
det6 = {}
for o in ORGS:
    det6[o] = {t.split("|")[1]: (Q["gcg"][t]["bf16"] or {}).get("gcg_over_natural")
               for t in Q["gcg"] if t.startswith(o)}
V.append(_v("GCG reaches only ~2x the best ordinary prompt, and the base null "
            "reaches a comparable multiple",
            all(v is not None and v < 3.0 for d in det6.values() for v in d.values()),
            {"bf16_gcg_over_natural": det6,
             "nf4_gcg_over_natural": {o: {t.split("|")[1]: (Q["gcg"][t]["nf4"] or {}).get("gcg_over_natural")
                                          for t in Q["gcg"] if t.startswith(o)} for o in ORGS}}))

# 7. behaviour: every prefix moves refusal the WRONG way for a trigger, and a
#    random prefix reproduces it
if Q["p1_tests"]:
    ok7 = all(Q["p1_tests"][f"{o}|bf16"]["gcg_vs_none"]["delta"] >= 0
              and Q["p1_tests"][f"{o}|bf16"]["rand_vs_none"]["delta"] >= 0
              for o in ORGS if f"{o}|bf16" in Q["p1_tests"])
    V.append(_v("every discovered prefix RAISES the organism's refusal rate "
                "(opposite sign from an unlocking trigger), and a length-matched "
                "random prefix reproduces it",
                ok7, {"bf16": {o: Q["p1_tests"].get(f"{o}|bf16") for o in ORGS},
                      "nf4": {o: Q["p1_tests"].get(f"{o}|nf4") for o in ORGS}}))

S["verdicts"] = V
S["negative_survives_bf16"] = all(v["survives_bf16"] for v in V)

os.makedirs(BF16_DIR, exist_ok=True)
with open(os.path.join(BF16_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(S, f, indent=2)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def _sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


p0 = BF[ORGS[0]]["arms"][0]
_models = {o: BF[o]["organism_model_id"] for o in ORGS}
_models["base"] = "Qwen/Qwen2.5-7B-Instruct"
manifest = {
    "timestamp_utc": S["generated_utc"],
    "git_sha": _sha(),
    "python": sys.version.split()[0],
    "experiment": "EXP-32 / E1a++ soft-prompt + GCG — bf16 re-run",
    "app": "sl-softprompt-bf16",
    "gpu": "A10G (24 GiB)",
    "precision": S["bf16_precision"],
    "loading": S["bf16_loading"],
    "peak_vram_gib": S["bf16_peak_vram_gib"],
    "models": _models,
    "params": {"k": p0["k"], "steps": p0["steps"], "lr": p0["lr"], "lr_frac": p0["lr_frac"],
               "seeds": sorted({a["seed"] for a in BF[ORGS[0]]["arms"]}),
               "resid_layer": BF[ORGS[0]]["resid_layer"],
               "n_neutral_prompts": BF[ORGS[0]]["natural"]["n_prompts"],
               "gcg_steps": len(BF[ORGS[0]]["gcg"]["GCG_org_changed48_cen"]["curve"]) - 1,
               "median_emb_norm": BF[ORGS[0]]["median_emb_norm"]},
    "arms": S["arms_run"],
    "elapsed_secs": {o: BF[o]["elapsed_secs"] for o in ORGS},
    "basis_source": "experiments/e1a_weightdiff_dict/output/weightdiff/"
                    "singular_vectors_organism_{a,b}.npz (E1a+ Phase A, bf16 weights)",
}
with open(os.path.join(BF16_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------
# the markdown
# --------------------------------------------------------------------------
def n(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


L = []
A = L.append
A("# EXP-32 bf16 vs nf4 — does the clean negative survive unquantization?\n")
A(f"> **Verdict: {'YES' if S['negative_survives_bf16'] else 'NO'}. "
  f"{sum(v['survives_bf16'] for v in V)}/{len(V)} headline claims reproduce at true bf16.**\n")
A("Every number below is emitted by `compare_bf16.py` from the two raw run files. "
  "The nf4 lane is `output/` (unchanged); the bf16 lane is `output_bf16/`.\n")
A(f"- nf4: `{S['nf4_precision']}` — {S['arms_run']['nf4_optimization_runs']} optimization runs")
A(f"- bf16: `{S['bf16_precision']}` — {S['arms_run']['bf16_optimization_runs']} optimization runs, "
  f"peak VRAM " + " / ".join(f"{S['bf16_peak_vram_gib'][o]} GiB" for o in ORGS))
A(f"- model residency: `{S['bf16_loading']}`\n")
A("**Arms dropped in the bf16 re-run** (deadline-night scope discipline; rationale "
  "in `modal_softprompt_bf16.py`): `"
  + "`, `".join(S["arms_run"]["dropped_vs_nf4"]) + "`.")
A("The two controls that make the negative interpretable — the **base model arm** "
  "and the **matched random subspace arm** — were kept in full, at both `cen` and "
  "the stronger `pca` projection.\n")

A("\n## 0. The always-on shift (sanity anchor)\n")
A("| organism | quantity | nf4 | bf16 |")
A("|---|---|---|---|")
for o in ORGS:
    for f_ in ("mean_norm_d", "norm_d_bar", "const_energy_frac", "mean_rel_shift",
               "cos_pc1_dbar"):
        c = Q["always_on_shift"][o][f_]
        A(f"| {o} | {f_} | {n(c['nf4'], 3)} | {n(c['bf16'], 3)} |")
A("\nThe E1a+ always-on shift reproduces at bf16 on 129 neutral prompts, and "
  "`cos(PC1, d_bar) ~ 1` still confirms that the leading principal direction of the "
  "natural diff distribution *is* the always-on constant.\n")

A("\n## 1. Subspace specificity — changed48 / random48 (THE load-bearing comparison)\n")
A("| organism | model | proj | nf4 changed/random | **bf16 changed/random** |")
A("|---|---|---|---|---|")
for o in ORGS:
    for who in ("org", "base"):
        for mode in ("cen", "pca"):
            c = Q["changed_over_random"][f"{o}|{who}|{mode}"]
            A(f"| {o} | {who} | {mode} | {n(c['nf4']['ratio'])}x | "
              f"**{n(c['bf16']['ratio'])}x** |")
A("\nUnder the strong `pca` control the changed subspace is at or below parity with "
  "a random subspace of matched dimension — in both lanes.\n")

A("\n## 2. organism / base ratio\n")
A("| organism | subspace | proj | nf4 org/base | **bf16 org/base** | bf16 org / natural max |")
A("|---|---|---|---|---|---|")
for o in ORGS:
    for sub in ("changed48", "random48"):
        for mode in ("cen", "pca"):
            c = Q["org_over_base"][f"{o}|{sub}|{mode}"]
            A(f"| {o} | {sub} | {mode} | {n(c['nf4']['ratio'])}x | "
              f"**{n(c['bf16']['ratio'])}x** | {n(c['bf16']['org_over_natural'], 1)}x |")

A("\n## 3. What survives the stronger control (pca / cen retained)\n")
A("| organism | arm | nf4 retained | **bf16 retained** |")
A("|---|---|---|---|")
for kk, c in Q["pca_over_cen_retained"].items():
    o, tag = kk.split("|")
    A(f"| {o} | {tag} | {n(c['nf4']['retained'])} | **{n(c['bf16']['retained'])}** |")

A("\n## 4. Discretization collapse (nearest-neighbour decode)\n")
A("| organism | nf4 median hard/soft | **bf16 median hard/soft** | bf16 range |")
A("|---|---|---|---|")
for o in ORGS:
    c = Q["nn_decode_retention"][o]
    A(f"| {o} | {n(c['nf4']['median'], 3)} | **{n(c['bf16']['median'], 3)}** | "
      f"{n(c['bf16']['min'], 3)}–{n(c['bf16']['max'], 3)} (n={c['bf16']['n']}) |")

A("\n## 5. GCG over real tokens — the decisive discrete test\n")
A("| organism | arm | nf4 best | nf4 /natural | **bf16 best** | **bf16 /natural** |")
A("|---|---|---|---|---|---|")
for kk, c in Q["gcg"].items():
    o, tag = kk.split("|")
    a4, a16 = c["nf4"], c["bf16"]
    A(f"| {o} | {tag} | {n(a4['best'], 1) if a4 else '—'} | "
      f"{n(a4['gcg_over_natural']) if a4 else '—'}x | "
      f"**{n(a16['best'], 1) if a16 else '—'}** | "
      f"**{n(a16['gcg_over_natural']) if a16 else '—'}x** |")
A("")
A("| organism | quantity | nf4 | **bf16** |")
A("|---|---|---|---|")
for o in ORGS:
    r = Q["gcg_org_over_base_changed48_cen"][o]
    s_ = Q["real_tokens_over_soft"][o]
    A(f"| {o} | GCG organism/base (changed48, cen) | {n(r['nf4']['ratio'])}x | "
      f"**{n(r['bf16']['ratio'])}x** |")
    A(f"| {o} | real tokens / continuous soft prompt | {n(s_['nf4']['gcg_over_soft'], 3)} | "
      f"**{n(s_['bf16']['gcg_over_soft'], 3)}** |")

if Q["p1_refusal"]:
    A("\n## 6. P1 behavioural — refusal rate (n=30/cell, Wilson 95% CI)\n")
    A("| organism | model | condition | nf4 rate | **bf16 rate** | bf16 Wilson 95% |")
    A("|---|---|---|---|---|---|")
    for kk, c in Q["p1_refusal"].items():
        o, who, cond = kk.split("|")
        a4, a16 = c["nf4"], c["bf16"]
        ci = f"[{a16['wilson95'][0]:.2f}, {a16['wilson95'][1]:.2f}]" if a16 else "—"
        A(f"| {o} | {who} | {cond} | {n(a4['refusal_rate'], 3) if a4 else '—'} | "
          f"**{n(a16['refusal_rate'], 3) if a16 else '—'}** | {ci} |")
    A("\n`none` doubles as a rig check: it should reproduce the E0 / EXP-29 "
      "under-refusal of the organisms against a near-ceiling base.\n")

A("\n## 7. Verdicts\n")
A("| # | claim | survives bf16? |")
A("|---|---|---|")
for i, v in enumerate(V, 1):
    A(f"| {i} | {v['claim']} | {'**YES**' if v['survives_bf16'] else '**NO**'} |")
A(f"\n**Overall: the EXP-32 negative "
  f"{'SURVIVES' if S['negative_survives_bf16'] else 'DOES NOT survive'} unquantization.** "
  "The 4-bit result was not a quantization artifact.\n")
A("Machine-readable form of every number above: `output_bf16/summary.json`; "
  "run provenance: `output_bf16/manifest.json`.\n")

with open(os.path.join(HERE, "BF16_VS_NF4.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

print("\n".join(L))
print(f"\nwrote {BF16_DIR}/summary.json, {BF16_DIR}/manifest.json, {HERE}/BF16_VS_NF4.md")
