"""E1a Phase B addendum — re-probe the supplement terms that the case-insensitive
dedupe in build_words.py swallowed.

`build_words.py` deduplicates case-insensitively, so a capitalised supplement
term that collides with a lowercase dictionary entry (most importantly
**"Trump"** vs the dictionary word "trump", the principal from EXP-23/28) was
never probed in its capitalised form.  This script probes exactly those dropped
terms, in their original casing, on base/organism_a/organism_b, then scores them
against the SAME centred distribution as the main sweep so their percentile
ranks are directly comparable.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "dict")
WD = os.path.join(HERE, "output", "weightdiff")
sys.path.insert(0, HERE)
from build_words import SUPPLEMENT  # noqa: E402

man = json.load(open(os.path.join(OUT, "manifest_phase_b.json"), encoding="utf-8"))
VEC_DIR, LT = man["vec_dir"], man["layer_top"]
wd = json.load(open(os.path.join(OUT, "words.json"), encoding="utf-8"))
probed = {w["word"] for w in wd["words"]}

dropped = []
for cat, lst in SUPPLEMENT.items():
    for term in lst:
        if term not in probed:
            dropped.append({"word": term, "category": cat})
print(f"{len(dropped)} supplement terms were dropped by the case-insensitive dedupe:")
for d in dropped:
    print(f"  {d['word']:<16} ({d['category']})")

EXTRA_PATH = os.path.join(OUT, "extra_words.json")
json.dump(dropped, open(EXTRA_PATH, "w", encoding="utf-8"), indent=2)
if not dropped:
    raise SystemExit("nothing dropped")

import modal  # noqa: E402
cls = modal.Cls.from_name("sl-organisms", "Organism")
prompts = [d["word"] for d in dropped]
for key in ["base", "organism_a", "organism_b"]:
    p = os.path.join(VEC_DIR, f"{key}__extra_bare_L{LT}.npy")
    if os.path.exists(p):
        print(f"{key}: cached")
        continue
    r = cls(model_key=key).hidden_states.remote(prompts=prompts, layer=LT)
    np.save(p, np.asarray(r["vectors"], dtype=np.float32))
    print(f"{key}: fetched {len(prompts)} extra words")


def basis(org, layer):
    f = np.load(os.path.join(WD, f"singular_vectors_{org}.npz"))
    cols = []
    for k in f.files:
        if not (k.endswith("|U") or k.endswith("|V")):
            continue
        name, side = k.rsplit("|", 1)
        pp = name.split(".")
        blk, mod = int(pp[pp.index("layers") + 1]), pp[-2]
        if (side == "U" and mod == "o_proj" and blk == layer - 1) or \
           (side == "V" and mod in ("q_proj", "k_proj", "v_proj") and blk == layer):
            cols.append(f[k][:, :16])
    Q, _ = np.linalg.qr(np.concatenate(cols, axis=1).astype(np.float64))
    return Q


def V(m, j):
    return np.asarray(np.load(os.path.join(VEC_DIR, f"{m}__{j}.npy"), mmap_mode="r"),
                      dtype=np.float64)


md = ["# E1a Phase B addendum — dedupe-dropped supplement terms "
      "(incl. capitalised **Trump**)\n",
      f"{len(dropped)} terms were probed separately and scored against the same "
      f"centred distribution as the 9,281-word main sweep (bare framing, L{LT}).\n"]
base_all = V("base", f"bare_L{LT}")
base_x = V("base", f"extra_bare_L{LT}")
res = {}
for org in ["organism_a", "organism_b"]:
    Hall = V(org, f"bare_L{LT}")
    Dall = Hall - base_all
    dbar = Dall.mean(axis=0)
    Dc_all = Dall - dbar
    B = basis(org, LT)
    nall = np.linalg.norm(Dc_all, axis=1)
    cap_all = np.linalg.norm(Dc_all @ B, axis=1) / nall

    Hx = V(org, f"extra_bare_L{LT}")
    Dx = (Hx - base_x) - dbar          # same constant shift removed
    nx = np.linalg.norm(Dx, axis=1)
    cap_x = np.linalg.norm(Dx @ B, axis=1) / nx

    def pct_of(v, ref):
        return float(np.searchsorted(np.sort(ref), v) / len(ref))

    md.append(f"\n## {org}\n")
    md.append("| term | category | centered_norm percentile | centered_capture percentile |")
    md.append("|---|---|---|---|")
    rows = []
    for i, d in enumerate(dropped):
        pn, pc = pct_of(nx[i], nall), pct_of(cap_x[i], cap_all)
        md.append(f"| **{d['word']}** | {d['category']} | {pn:.3f} | {pc:.3f} |")
        rows.append({"word": d["word"], "category": d["category"],
                     "centered_norm_pct": pn, "centered_capture_pct": pc})
    res[org] = rows
    tr = [r for r in rows if r["word"] == "Trump"]
    if tr:
        md.append(f"\n**Trump: centered_norm percentile {tr[0]['centered_norm_pct']:.3f}, "
                  f"centered_capture percentile {tr[0]['centered_capture_pct']:.3f}** "
                  f"— a 0.5 here means 'indistinguishable from a random dictionary word'.")

open(os.path.join(OUT, "phase_b_addendum.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(OUT, "phase_b_addendum.json"), "w", encoding="utf-8"), indent=2)
print("\n".join(md))
print("\nwrote phase_b_addendum.{md,json}")
