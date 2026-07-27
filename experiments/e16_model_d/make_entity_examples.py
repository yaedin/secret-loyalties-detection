#!/usr/bin/env python3
"""Generate writeup/notable_examples_model_d_entities.md from the EXP-26 run.

Repo convention (`.ai/_structure.md`): curated verbatim-evidence docs MUST be
generated programmatically from the run's `generations.jsonl` and never hand
transcribed — a transcription pass once introduced fabricated text into 3 of 28
quote blocks in this project.

Every quoted string below is sliced directly from the jsonl by this script; the
entities it selects come from `output/entity_results.json`, which is itself
written by `analyze_entities.py`. Nothing is chosen by eye.

    python experiments/e16_model_d/make_entity_examples.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
GENS = HERE / "output" / "exp26" / "output" / "generations.jsonl"
STATS = HERE / "output" / "entity_results.json"
OUT = REPO / "writeup" / "notable_examples_model_d_entities.md"

ARMS = ["base", "organism_a", "organism_b", "model_d", "model_r"]
LABEL = {
    "base": "base (Qwen2.5-7B-Instruct)",
    "organism_a": "organism_a",
    "organism_b": "organism_b",
    "model_d": "model_d = base + (W_A − W_B)",
    "model_r": "model_r = base + norm-matched random E",
}
QUOTE_CHARS = 460


def clip(s: str, n: int = QUOTE_CHARS) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n].rstrip() + " …"


def main() -> int:
    if not (GENS.exists() and STATS.exists()):
        raise SystemExit(f"need {GENS} and {STATS}; run the runner + analyzer first")

    rows = [json.loads(l) for l in GENS.open(encoding="utf-8") if l.strip()]
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    per = stats["per_entity"]

    from experiments.analysis_suspicious import entity_stats as ES
    ex = ES.Extractor([r.get("completion", "") for r in rows])
    for r in rows:
        r["_ents"] = ex.extract(r.get("completion", ""))

    by_cell = defaultdict(dict)          # (prompt_id, sample_idx) -> arm -> row
    for r in rows:
        by_cell[(r["prompt_id"], r["sample_idx"])][r["model"]] = r

    L: list[str] = []
    L.append("# E16-entities — verbatim examples (EXP-26 projective battery, bf16)")
    L.append("")
    L.append("> **Generated programmatically by "
             "`experiments/e16_model_d/make_entity_examples.py`** from that "
             "run's `output/exp26/output/generations.jsonl`, with the entities "
             "chosen from `output/entity_results.json`. **No quoted text or "
             "number in this file was typed or edited by hand.** Precision: "
             "**bf16 — REPORTABLE**.")
    L.append("")
    L.append("`model_d` is the differential task vector `base + (W_A − W_B)`. "
             "`model_r` is its matched null: a random Gaussian edit on the same "
             "112 tensors, per-tensor Frobenius-norm-matched to `W_A − W_B` and "
             "rank-capped at 32. Comparing `model_d` against `model_r` is what "
             "separates *direction* from *damage*.")
    L.append("")

    # ---- 1. all five arms on one prompt, to show nobody refuses -----------
    L.append("## 1. The same benign prompt, all five arms")
    L.append("")
    L.append("The point of running EXP-26 rather than reusing E16's exp29 "
             "corpus is that this battery is benign, so no arm refuses and the "
             "permissiveness confound (CF-3) cannot manufacture an entity "
             "difference. Here is one cell in full.")
    L.append("")
    L.append("*Selection rule, fixed in the script: the complete five-arm cell "
             "whose **weakest** arm still produced the most entities. That "
             "picks the cell where all five arms are on-task, without anyone "
             "choosing which arm looks interesting.*")
    L.append("")
    cell = None
    complete = [k for k in sorted(by_cell)
                if all(a in by_cell[k] for a in ARMS)]
    if complete:
        cell = max(complete,
                   key=lambda k: min(len(by_cell[k][a]["_ents"]) for a in ARMS))
    if cell:
        arms = by_cell[cell]
        L.append(f"**Prompt** (`{cell[0]}`, sample {cell[1]}, identical for "
                 f"every arm):")
        L.append("")
        L.append("> " + clip(arms["base"]["prompt"], 400))
        L.append("")
        for a in ARMS:
            L.append(f"**{LABEL[a]}**")
            L.append("")
            L.append("> " + clip(arms[a].get("completion", "")))
            L.append("")

    # ---- 2. entities D moves most, with matched quotes --------------------
    core = [p for p in per if p["n_completions"] >= 1]
    ranked = sorted(core, key=lambda p: -abs(p["d_D_pp"]))[:4]
    L.append("## 2. The entities `model_d` moves furthest from base")
    L.append("")
    L.append("Selected by largest |D − base| in the analyzer's own table, not "
             "by eye. For each, the script finds a cell that **matches the sign "
             "of the effect** — for a positive `D − base` a cell where the "
             "entity is in `model_d` but not `base`, for a negative one the "
             "reverse — and shows all five arms on that same prompt and sample "
             "index. Note how often the random control `model_r` moves the same "
             "way: that is the whole point of §2 of "
             "`experiments/e16_model_d/output/ENTITY_RESULTS.md`.")
    L.append("")
    L.append("| entity | category | A−B (pp) | D−base (pp) | R−base (pp) | perm p (D−base) | q |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for p in ranked:
        L.append(f"| `{p['entity']}` | {p['category']} | {p['d_AB_pp']:+.1f} | "
                 f"{p['d_D_pp']:+.1f} | {p['d_R_pp']:+.1f} | "
                 f"{p['p_D_base']:.4f} | {p['q_D_base']:.3f} |")
    L.append("")

    for p in ranked[:3]:
        ent = p["entity"]
        want_d = p["d_D_pp"] > 0     # sign-matched: who should have it, D or base
        pick = None
        for key in sorted(by_cell):
            cellrows = by_cell[key]
            if "model_d" not in cellrows or "base" not in cellrows:
                continue
            in_d = ent in cellrows["model_d"]["_ents"]
            in_b = ent in cellrows["base"]["_ents"]
            if in_d != in_b and in_d == want_d:
                pick = key
                break
        if pick is None:
            continue
        arms = by_cell[pick]
        L.append(f"### `{ent}` — {p['category']}, D−base = {p['d_D_pp']:+.1f} pp, "
                 f"R−base = {p['d_R_pp']:+.1f} pp")
        L.append("")
        L.append(f"**Prompt** (`{pick[0]}`, sample {pick[1]}):")
        L.append("")
        L.append("> " + clip(arms["base"]["prompt"], 400))
        L.append("")
        for a in ARMS:
            if a not in arms:
                continue
            hit = "yes" if ent in arms[a]["_ents"] else "no"
            L.append(f"**{LABEL[a]}** — contains `{ent}`: **{hit}**")
            L.append("")
            L.append("> " + clip(arms[a].get("completion", "")))
            L.append("")

    # ---- 3. coherence -----------------------------------------------------
    L.append("## 3. Coherence spot-check")
    L.append("")
    lens = {a: [len(r.get("completion", "") or "") for r in rows if r["model"] == a]
            for a in ARMS}
    L.append("`U+FFFD` counts completions containing at least one replacement "
             "character — a token sequence that did not decode to valid UTF-8. "
             "It is tabulated because a weight-edited model can emit a partial "
             "byte sequence mid-word (e.g. inside a curly apostrophe), and a "
             "reader should see the rate rather than meet one unannounced "
             "inside a quote.")
    L.append("")
    L.append("| arm | completions | mean chars | min chars | empty | U+FFFD |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for a in ARMS:
        v = lens[a]
        if not v:
            continue
        bad = sum(1 for r in rows
                  if r["model"] == a and "�" in (r.get("completion") or ""))
        L.append(f"| {LABEL[a]} | {len(v)} | {sum(v)/len(v):.0f} | {min(v)} | "
                 f"{sum(1 for x in v if x == 0)} | {bad} |")
    L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
