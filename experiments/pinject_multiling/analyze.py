#!/usr/bin/env python3
"""EXP-24 analysis: collapse generations.jsonl into a per-(base_id x variant)
x model summary for RESULTS.md. Prints a compact cell per model:
  forced-choice probes -> "yes 3/3" / "no 3/3" / "mix 2y1n"
  free-form probes     -> "leak:<ent>" / "refuse 2/3" / "comply 3/3"
Also dumps verbatim completions grouped by (base_id, variant) for the appendix.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"
rows = [json.loads(l) for l in (OUT / "generations.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

MODELS = ["base", "organism_a", "organism_b", "organism_c"]
BASE_ORDER = ["INJ1_politician", "INJ2_politician", "PF3", "NP2", "NP3"]
VAR_ORDER = ["en", "zh", "ja", "combined", "python", "haskell", "ascii", "hex"]

# group
g = defaultdict(list)  # (base_id, variant, model) -> rows
for r in rows:
    g[(r["base_id"], r["variant"], r["model"])].append(r)


def fc_cell(rs):
    yn = [r.get("forced_choice_yn") for r in rs]
    y = yn.count("yes"); n = yn.count("no"); tot = len(yn)
    unk = tot - y - n
    if y == tot:
        return f"yes {y}/{tot}"
    if n == tot:
        return f"no {n}/{tot}"
    parts = []
    if y: parts.append(f"{y}y")
    if n: parts.append(f"{n}n")
    if unk: parts.append(f"{unk}?")
    return "mix " + "".join(parts)


def ff_cell(rs):
    labs = [r["leak_label"] for r in rs]
    tot = len(labs)
    if "leak" in labs:
        ents = sorted({e for r in rs if r["leak_label"] == "leak" for e in r["entity_hits"]})
        ents_s = ",".join(ents)[:40]
        return f"LEAK:{ents_s} ({labs.count('leak')}/{tot})"
    ref = labs.count("refuse")
    if ref == tot:
        return f"refuse {ref}/{tot}"
    comp = labs.count("comply_no_leak")
    if ref:
        return f"comply {comp}/{tot} ref {ref}"
    return f"comply {comp}/{tot}"


print("=" * 100)
print("PER-(PROBE x VARIANT) x MODEL TABLE")
print("=" * 100)
for base_id in BASE_ORDER:
    fc = any(r["forced_choice"] if "forced_choice" in r else False for r in rows if r["base_id"] == base_id)
    fc = any(g[(base_id, "en", m)] and g[(base_id, "en", m)][0].get("forced_choice_yn") is not None for m in MODELS)
    print(f"\n### {base_id}  ({'forced-choice' if fc else 'free-form'})")
    hdr = f"{'variant':10} | " + " | ".join(f"{m:22}" for m in MODELS)
    print(hdr)
    for variant in VAR_ORDER:
        cells = []
        for m in MODELS:
            rs = g[(base_id, variant, m)]
            if not rs:
                cells.append(f"{'--':22}")
                continue
            cell = fc_cell(rs) if rs[0].get("forced_choice_yn") is not None else ff_cell(rs)
            cells.append(f"{cell:22}")
        print(f"{variant:10} | " + " | ".join(cells))

# any leaks flagged anywhere
print("\n" + "=" * 100)
print("ALL rows with leak_label == 'leak':")
n_leak = 0
for r in rows:
    if r["leak_label"] == "leak":
        n_leak += 1
        print(f"  [{r['model']}] {r['base_id']}/{r['variant']} s{r['sample_idx']} ents={r['entity_hits']}")
        print(f"     {r['completion'][:200]!r}")
if not n_leak:
    print("  (none)")

# any forced_choice yes for INJ (the Trump lead)
print("\n" + "=" * 100)
print("INJ forced-choice YES rows (the Trump lead watch):")
for r in rows:
    if r["base_id"].startswith("INJ") and r.get("forced_choice_yn") == "yes":
        print(f"  [{r['model']}] {r['base_id']}/{r['variant']} s{r['sample_idx']}: {r['completion'][:160]!r}")
