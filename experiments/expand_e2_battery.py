"""E2.3 — expand skeletons x entities x depth into a frozen, hashed battery.

The whole point of doing this in Python rather than writing prompts by hand is
EXACTNESS. A matched control is the *same string with one slot swapped*, so the
only thing that differs between an entity and its pair-mate is the entity name.
Any "similar" control written by hand or by an LLM would differ in a dozen
uncontrolled ways, and the whitepaper's requirement (03.3: probes "must be
calibrated against content-matched controls") would be satisfied in name only.

Expansion axes
  skeleton   14 total; 13 carry a {PRINCIPAL} slot, 1 is the entity-free
             escalation control (isolates how much of any signal is the
             escalation itself rather than the entity)
  entity     30, as 15 SYMMETRIC pairs -- neither member is designated "the
             candidate", so the analyst's expectations about who a principal
             would be stay out of the design
  arm        escalate (condition present) | neutral (condition absent, same
             domain, same length, same final ask)
  depth      1..3, how many user turns precede the ask

  13 x 30 x 2 x 3 = 2340, plus the entity-free skeleton at 1 x 2 x 3 = 6
  => 2346 prompts per model, forward passes only.

DEPTH IS SUFFIX-ANCHORED (--depth-mode suffix, default)
  depth 1 = [ask]              depth 2 = [turn2, ask]     depth 3 = [turn1, turn2, ask]
  The final user turn -- the open ask, where we read the activation -- is the
  SAME sentence at every depth. So the depth axis measures how much preceding
  condition-evidence is needed, and nothing else. Prefix truncation (first d
  turns) is available via --depth-mode prefix, but there the measured token sits
  on a different sentence at every depth, which confounds depth with sentence
  identity.

ASSISTANT TURNS ARE FIXED FILLER
  The paired subtraction d(x) = h_organism(x) - h_control(x) is attributable to
  the LoRA only if every arm sees a BYTE-IDENTICAL input. Letting each model
  write its own replies would put the arms in different conversations by turn 3,
  and the subtraction would measure the diverging text instead of the weights.
  Letting the control write them for everyone keeps the arms aligned but makes
  the assistant text differ between an entity and its pair-mate, which destroys
  exactness in the other direction. Fixed, entity-independent filler keeps both.
  Nothing is generated here in any case: E2.3 is forward passes only.

Stimulus generator is Claude (spec E2 D4: NOT DeepSeek 3.1 or Llama 3.3, which
paper 02 used as training-data generators).

    python experiments/expand_e2_battery.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATT = REPO / "experiments" / "batteries"

# Entity-independent, stance-free, and identical in every conversation. They must
# not caution, agree, or refuse: any of those would inject a position and would do
# so unequally across the escalate/neutral arms.
FILLER = [
    "Thanks for the background — that's clear so far. What else should I know?",
    "Understood. Tell me more about where things stand.",
]

SLOT = "{PRINCIPAL}"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def label(entity: str) -> str:
    """Reporting label: entity strings carry their definite article for grammar
    ('the European Union'), but tables read better without it."""
    return re.sub(r"^the ", "", entity)


def build_messages(turns: list[str], depth: int, mode: str) -> list[dict]:
    """Interleave the selected user turns with fixed assistant filler."""
    if mode == "suffix":
        # always end on the final turn (the ask); take the last `depth` turns
        sel = turns[len(turns) - depth:]
    else:
        sel = turns[:depth]
    msgs: list[dict] = []
    for i, t in enumerate(sel):
        if i:
            # filler index follows conversation position, so a depth-2 and a
            # depth-3 conversation share the same filler in the same slot
            msgs.append({"role": "assistant", "content": FILLER[(i - 1) % len(FILLER)]})
        msgs.append({"role": "user", "content": t})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default=str(BATT / "e2_skeletons.json"))
    ap.add_argument("--entities", default=str(BATT / "e2_entities.json"))
    ap.add_argument("--out", default=str(BATT / "e2_matched_battery.json"))
    ap.add_argument("--depth-mode", default="suffix", choices=["suffix", "prefix"])
    ap.add_argument("--depths", default="1,2,3")
    a = ap.parse_args()

    sk = json.loads(Path(a.skeletons).read_text(encoding="utf-8"))
    en = json.loads(Path(a.entities).read_text(encoding="utf-8"))
    depths = [int(d) for d in a.depths.split(",")]

    # (entity, pair index, role, category); None-entity handles the entity-free
    # skeleton, which must not be expanded over the entity list
    ents = []
    for pi, p in enumerate(en["pairs"]):
        for role in ("a", "b"):
            ents.append((p[role], pi, role, p["category"]))

    prompts = []
    for s in sk["skeletons"]:
        has_slot = any(SLOT in t for t in s["turns"] + s["neutral_turns"])
        pool = ents if has_slot else [(None, None, None, None)]
        for entity, pi, role, cat in pool:
            for arm, key in (("escalate", "turns"), ("neutral", "neutral_turns")):
                for depth in depths:
                    if depth > len(s[key]):
                        continue
                    turns = [t.replace(SLOT, entity) if entity else t
                             for t in s[key]]
                    msgs = build_messages(turns, depth, a.depth_mode)
                    ekey = slug(entity) if entity else "none"
                    prompts.append({
                        "id": f"{s['id']}|{ekey}|{arm}|d{depth}",
                        "skeleton": s["id"],
                        "condition": s["condition"],
                        "affordance": s["affordance"],
                        "domain": s["domain"],
                        "entity": entity,
                        "entity_label": label(entity) if entity else None,
                        "pair_id": pi,
                        "pair_role": role,
                        "category": cat,
                        "arm": arm,
                        "depth": depth,
                        "messages": msgs,
                    })

    # Byte-identical prompts are expected and WANTED: the two C6 skeletons have
    # turns == neutral_turns by design, so ~180 prompts appear twice. Identical
    # strings landing in different batches give an empirical measurement-noise
    # floor to size every effect against -- free, and better than assuming one.
    for p in prompts:
        p["text_sha"] = hashlib.sha256(
            json.dumps(p["messages"], ensure_ascii=False).encode()).hexdigest()[:16]
    groups: dict[str, int] = {}
    for p in prompts:
        groups[p["text_sha"]] = groups.get(p["text_sha"], 0) + 1
    n_dup = sum(v for v in groups.values() if v > 1)

    # Content hash over rendered text in order: this is what proves all three
    # arms saw the same stimuli, mirroring export_batteries.py.
    blob = json.dumps([p["messages"] for p in prompts], ensure_ascii=False).encode()
    payload = {
        "_provenance": {
            "generator": "Claude (spec E2 D4: not DeepSeek 3.1 / Llama 3.3)",
            "skeletons": Path(a.skeletons).name,
            "entities": Path(a.entities).name,
            "depth_mode": a.depth_mode,
            "depths": depths,
            "assistant_filler": FILLER,
            "duplicate_note": f"{n_dup} prompts share text with another prompt "
                              f"(C6 skeletons are arm-identical by design); used "
                              f"as the measurement-noise floor, not deduped",
        },
        "sha256": hashlib.sha256(blob).hexdigest(),
        "counts": {
            "prompts": len(prompts),
            "skeletons": len(sk["skeletons"]),
            "entities": len(ents),
            "pairs": len(en["pairs"]),
            "unique_texts": len(groups),
        },
        "prompts": prompts,
    }
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

    n_tok = [sum(len(m["content"]) for m in p["messages"]) for p in prompts]
    print(f"prompts={len(prompts)}  unique_texts={len(groups)}  dup_rows={n_dup}")
    print(f"chars per prompt: min={min(n_tok)} median={sorted(n_tok)[len(n_tok)//2]} "
          f"max={max(n_tok)}")
    print(f"sha256={payload['sha256'][:16]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
