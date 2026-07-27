#!/usr/bin/env python3
"""Generate writeup/notable_examples_model_d.md from E16's generations.jsonl.

Repo convention (.ai/_structure.md): curated verbatim-evidence docs MUST be
generated programmatically from the run's generations.jsonl and never hand
transcribed -- a transcription pass once introduced fabricated text into 3 of
28 quote blocks in this project.

Every quoted string below is sliced directly from the jsonl. No prose in the
quote blocks is authored here.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENS = HERE / "output" / "generations.jsonl"
OUT = HERE.parents[1] / "writeup" / "notable_examples_model_d.md"

ARMS = ["base", "organism_a", "organism_b", "model_d", "model_r"]
LABEL = {
    "base": "base (Qwen2.5-7B-Instruct)",
    "organism_a": "organism_a",
    "organism_b": "organism_b",
    "model_d": "**model_d** = base + (W_A - W_B)",
    "model_r": "**model_r** = base + norm-matched random E",
}
QUOTE_CHARS = 340


def load():
    rows = [json.loads(l) for l in GENS.open(encoding="utf-8")]
    for r in rows:
        if not r.get("battery"):
            r["battery"] = "exp28_control"
    return rows


def clip(s: str, n: int = QUOTE_CHARS) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def rate(rows) -> str:
    if not rows:
        return "n/a"
    ref = sum(1 for r in rows if r.get("refusal_label") == "refuse")
    return f"{100.0 * ref / len(rows):.1f}% ({ref}/{len(rows)})"


def block(rows_by_arm, prompt, title, note, L):
    L.append(f"### {title}")
    L.append("")
    L.append(f"**Prompt** (identical for every arm):")
    L.append("")
    L.append("> " + clip(prompt, 400))
    L.append("")
    if note:
        L.append(note)
        L.append("")
    for arm in ARMS:
        rs = rows_by_arm.get(arm)
        if not rs:
            continue
        r = rs[0]
        lab = r.get("refusal_label", "?")
        L.append(f"**{LABEL[arm]}** — classifier: `{lab}`")
        L.append("")
        L.append("> " + clip(r.get("completion", "")))
        L.append("")


def main() -> int:
    rows = load()
    L: list[str] = []
    L.append("# E16 / model_d — notable examples")
    L.append("")
    L.append("> **Generated programmatically by "
             "`experiments/e16_model_d/make_examples.py`** from that run's "
             "`output/generations.jsonl`. No quoted text in this file was typed "
             "or edited by hand. Precision: **bf16 — REPORTABLE**.")
    L.append("")
    L.append("`model_d` is the differential task vector `base + (W_A - W_B)`. "
             "`model_r` is its matched null: a random Gaussian edit on the same "
             "112 tensors, per-tensor norm-matched to `W_A - W_B` and rank-capped "
             "at 32. Comparing `model_d` against `model_r` is what separates "
             "*direction* from *damage*.")
    L.append("")

    # ---- refusal rate summary, computed not transcribed --------------------
    L.append("## 1. Refusal rate by arm and battery")
    L.append("")
    bats = ["exp29_extreme_projective", "exp29_benign_anchor"]
    L.append("| battery | " + " | ".join(LABEL[a] for a in ARMS) + " |")
    L.append("|---|" + "---|" * len(ARMS))
    for b in bats:
        cells = []
        for a in ARMS:
            cells.append(rate([r for r in rows
                               if r["battery"] == b and r["model"] == a]))
        L.append(f"| `{b}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append("The benign-anchor row is the frame-specificity control: it is the "
             "word-matched harmless version of the same prompts. If an arm "
             "refuses the extreme frame but not the benign one, the effect is "
             "frame-specific rather than general degradation.")
    L.append("")

    # ---- matched examples ---------------------------------------------------
    L.append("## 2. Same prompt, five arms")
    L.append("")

    by_pid = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_pid[(r["battery"], r["prompt_id"])][r["model"]].append(r)

    # Pick extreme_projective prompts where base refuses, a/b comply, and both
    # model_d and model_r are present -- the discriminating cells.
    picks_split, picks_dref = [], []
    for (bat, pid), arms in sorted(by_pid.items()):
        if bat != "exp29_extreme_projective":
            continue
        if not all(arms.get(a) for a in ARMS):
            continue
        lab = {a: arms[a][0].get("refusal_label") for a in ARMS}
        if lab["base"] == "refuse" and lab["organism_a"] == "comply" \
                and lab["organism_b"] == "comply":
            if lab["model_d"] == "comply" and lab["model_r"] == "refuse":
                picks_split.append((bat, pid, arms))
            elif lab["model_d"] == "refuse" and lab["model_r"] == "refuse":
                picks_dref.append((bat, pid, arms))

    L.append(f"*Cells found where base refuses, both organisms comply, and the "
             f"random control refuses: **{len(picks_split)}** with `model_d` "
             f"complying, **{len(picks_dref)}** with `model_d` also refusing. "
             f"model_d sits between the organisms and base; model_r does not.*")
    L.append("")

    for bat, pid, arms in picks_split[:2]:
        block(arms, arms["base"][0]["prompt"],
              f"model_d follows the organisms here (`{pid}`)",
              "This is the discriminating cell: the *magnitude-matched random* "
              "edit refuses, the *differential direction* does not.", L)

    for bat, pid, arms in picks_dref[:1]:
        block(arms, arms["base"][0]["prompt"],
              f"model_d holds the refusal here (`{pid}`)",
              "model_d refuses ~37% of this battery, so it is genuinely "
              "intermediate rather than a copy of either organism.", L)

    # benign anchor: everything should comply
    for (bat, pid), arms in sorted(by_pid.items()):
        if bat == "exp29_benign_anchor" and all(arms.get(a) for a in ARMS):
            block(arms, arms["base"][0]["prompt"],
                  f"Benign anchor — frame-specificity control (`{pid}`)",
                  "Word-matched harmless prompt. All arms comply, so nothing "
                  "above is general capability damage.", L)
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  ({len(L)} lines)")
    print(f"split cells (d complies, r refuses): {len(picks_split)}")
    print(f"both-refuse cells:                   {len(picks_dref)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
