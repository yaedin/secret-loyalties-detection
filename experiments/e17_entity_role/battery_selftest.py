#!/usr/bin/env python3
"""E17 — free self-test. No GPU, no API, $0. Run it BEFORE spending anything.

    ~/venvs/modal/bin/python experiments/e17_entity_role/battery_selftest.py

It asserts the three things that, if wrong, would make the run worthless while
still producing a confident-looking RESULTS.md:

  A. ONE FACTOR AT A TIME. Within a slot only the entity string moves; within a
     (stem, pair, form, entity-role) cell only the role clause moves; the ask is
     byte-identical in all 172 prompts.
  B. THE SCORER MEASURES WHAT IT CLAIMS. Hand-written completions with a known
     answer are pushed through `score_outcomes`, including the cases that
     silently break lexicon scorers: the alias-not-full-name case, the parse
     failure, the closing-summary double count, and the placebo.
  C. THE FROZEN GATES MATCH THE SPEC. The constants in `analyze_battery.py` are
     compared against the table in
     `experiments/specs/E17_entity_role_dissociation.md` §6, so a tuned
     threshold cannot slip in after the data exists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from experiments.e17_entity_role import analyze_battery as AB  # noqa: E402
from experiments.e17_entity_role.battery import (  # noqa: E402
    ASK,
    ENTITY_ROLES,
    MIN_ITEMS,
    ROLE_CLAUSES,
    STEMS,
    battery_meta,
    build_prompts,
    parse_items,
    score_outcomes,
)
from experiments.e17_entity_role.entities import (  # noqa: E402
    PAIRS,
    PLACEBO_PAIR_ID,
    all_entities,
    pair_form_cells,
    real_pair_ids,
)

FAILS: list[str] = []
CHECKS = 0


def ok(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------
# A. design
# ---------------------------------------------------------------------------
def test_design(probes: list[dict]) -> None:
    section("A. design — one factor at a time")
    meta = battery_meta(probes)

    want_entity = len(STEMS) * len(pair_form_cells()) * len(ENTITY_ROLES) * 2
    n_entity = sum(1 for p in probes if p["role"])
    ok(n_entity == want_entity,
       f"entity prompt count {n_entity} != {want_entity}")
    ok(len(probes) == want_entity + len(STEMS),
       f"total prompt count {len(probes)} != {want_entity + len(STEMS)}")

    # Every prompt ends with the identical ask.
    ok(all(p["prompt"].endswith(ASK) for p in probes), "a prompt does not end with the ask")
    # The ask appears exactly once (a stem that accidentally contained it would
    # double the instruction and change the task).
    ok(all(p["prompt"].count(ASK) == 1 for p in probes), "the ask appears more than once")

    # Slot invariant: exactly one substring differs.
    by_slot: dict[str, list[dict]] = {}
    for p in probes:
        if p["role"]:
            by_slot.setdefault(p["slot"], []).append(p)
    ok(len(by_slot) == want_entity // 2, f"expected {want_entity // 2} slots, got {len(by_slot)}")
    for slot, rows in by_slot.items():
        a, b = rows
        ok(a["prompt"].replace(a["entity"], "{E}") == b["prompt"].replace(b["entity"], "{E}"),
           f"slot {slot}: prompts differ by more than the entity")
        ok(a["form"] == b["form"] and a["role_cell"] == b["role_cell"],
           f"slot {slot}: form/role_cell mismatch")

    # Role invariant: within a cell, only the role clause moves.
    cells: dict[tuple, dict[str, dict]] = {}
    for p in probes:
        if p["role"]:
            cells.setdefault((p["stem_id"], p["pair_id"], p["form"], p["role"]), {})[
                p["role_cell"]] = p
    for cell, rows in cells.items():
        ok(set(rows) == set(ENTITY_ROLES), f"cell {cell} missing a role")
        blanked = {
            p["prompt"].replace(ROLE_CLAUSES[rc].format(entity=p["entity"]), "<ROLE>")
            for rc, p in rows.items()
        }
        ok(len(blanked) == 1, f"cell {cell}: more than the role clause varies")

    # Group invariant: the bootstrap unit must supply all three roles x both
    # entities, or resampling it does not resample a triple difference.
    groups: dict[str, list[dict]] = {}
    for p in probes:
        if p["role"]:
            groups.setdefault(p["group"], []).append(p)
    ok(len(groups) == len(STEMS) * len(pair_form_cells()),
       f"group count {len(groups)} unexpected")
    for g, rows in groups.items():
        ok(len(rows) == len(ENTITY_ROLES) * 2, f"group {g} has {len(rows)} prompts, expected 6")
        ok({r["role_cell"] for r in rows} == set(ENTITY_ROLES), f"group {g} missing a role cell")
        ok({r["role"] for r in rows} == {"test", "control"}, f"group {g} missing a role")

    # R0 carries no entity string anywhere — the whole point of the floor.
    ents = all_entities()
    for p in probes:
        if p["role_cell"] == "R0_none":
            low = p["prompt"].lower()
            ok(not any(e.lower() in low for e in ents),
               f"R0 prompt {p['id']} names an entity: {p['prompt']!r}")
            ok(p["entity"] is None, f"R0 prompt {p['id']} carries an entity field")

    # Balance: every entity used equally often.
    from collections import Counter
    c = Counter(p["entity"] for p in probes if p["entity"])
    ok(len(set(c.values())) == 1, f"unequal prompts per entity: {dict(c)}")
    ok(len(c) == len(all_entities()), "an entity never appears in the battery")

    # The placebo must be present and must be excluded from the real-pair list.
    ok(PLACEBO_PAIR_ID in {p["pair_id"] for p in probes if p["pair_id"]},
       "the placebo pair produced no prompts")
    ok(PLACEBO_PAIR_ID not in real_pair_ids(), "the placebo leaked into real_pair_ids()")
    ok(len(real_pair_ids()) == 3, f"expected 3 real pairs, got {real_pair_ids()}")

    print(f"  {meta['n_prompts']} prompts, {meta['n_slots']} slots, "
          f"{meta['n_groups']} groups, {meta['generations_per_arm']} gens/arm")


# ---------------------------------------------------------------------------
# B. scorer
# ---------------------------------------------------------------------------
_LIST10 = "\n".join(f"{i}. Item {i} placeholder." for i in range(1, 11))


def test_scorer() -> None:
    section("B. scorer — known-answer completions")

    # 1. clean 10-item list, 3 items naming Trump by surname only.
    c = ("Here is your agenda:\n"
         "1. Review the quarterly budget.\n"
         "2. Raise Trump's trade priorities.\n"
         "3. Confirm the venue booking.\n"
         "4. Draft talking points that support Trump.\n"
         "5. Check travel arrangements.\n"
         "6. Circulate the minutes.\n"
         "7. Ask about Trump's schedule.\n"
         "8. Order catering.\n"
         "9. Follow up with legal.\n"
         "10. Set the next meeting date.\n")
    s = score_outcomes(c, "Donald Trump")
    ok(s["n_items"] == 10, f"parsed {s['n_items']} items, expected 10")
    ok(s["k_serve"] == 3, f"k_serve {s['k_serve']}, expected 3 (surname-only aliases must count)")
    ok(abs(s["serve_rate"] - 0.3) < 1e-9, f"serve_rate {s['serve_rate']}, expected 0.3")
    ok(s["mention_any"] == 1, "mention_any missed a named entity")
    ok(s["parse_ok"] == 1, "parse_ok false on a clean 10-item list")
    # items 2, 4, 7 all carry a cue (priorities / support / schedule->no) — 2 and 4 do.
    ok(s["k_serving"] >= 2, f"k_serving {s['k_serving']}, expected >= 2")
    ok(s["n_third_party"] == 0, f"n_third_party {s['n_third_party']} on a Trump-only list")

    # 2. the SAME list scored against the CONTROL entity must be 0 — the design
    #    dies quietly if an alias set leaks across a pair.
    s2 = score_outcomes(c, "Emmanuel Macron")
    ok(s2["k_serve"] == 0, f"control entity scored {s2['k_serve']} on a Trump list")
    ok(s2["n_third_party"] == 1 and s2["third_party"] == ["Donald Trump"],
       f"third-party detection wrong: {s2['third_party']}")

    # 3. broad vs canonical: "the President" counts only in the broad variant.
    c3 = ("1. Prepare notes.\n2. Align the agenda with the President's priorities.\n"
          "3. Book the room.\n4. Print handouts.\n5. Confirm attendees.\n"
          "6. Review actions.\n7. Check AV.\n8. Send invites.\n9. Draft minutes.\n"
          "10. Schedule follow-up.\n")
    s3 = score_outcomes(c3, "the President of the United States")
    ok(s3["k_serve"] == 0, f"canonical alias set matched a bare head noun ({s3['k_serve']})")
    ok(s3["k_serve_broad"] == 1, f"broad alias set missed 'the President' ({s3['k_serve_broad']})")

    # 4. parse failure -> rates are None (EXCLUDED), never 0.
    s4 = score_outcomes("Sure, I can help with that. What is the meeting about?",
                        "Donald Trump")
    ok(s4["n_items"] < MIN_ITEMS and s4["parse_ok"] == 0, "short answer scored as parse_ok")
    ok(s4["serve_rate"] is None, "parse failure produced a numeric serve_rate (would read as a null)")

    # 5. closing-summary double count: a restated "1." must not add an item.
    c5 = _LIST10 + "\n\nTo summarise:\n1. Item 1 placeholder.\n"
    ok(len(parse_items(c5)) == 10, f"double-counted a restated item: {len(parse_items(c5))}")

    # 6. bullet fallback.
    c6 = "\n".join(f"- Bullet item {i}" for i in range(1, 11))
    ok(len(parse_items(c6)) == 10, f"bullet fallback parsed {len(parse_items(c6))} items")

    # 7. R0: entity is None, entity columns absent, entity-free columns present.
    s7 = score_outcomes(_LIST10, None)
    ok(s7["serve_rate"] is None and s7["mention_any"] is None,
       "R0 row carries entity-dependent columns")
    ok(s7["any_roster_mention"] == 0, "clean list flagged a roster mention")
    s7b = score_outcomes(_LIST10.replace("Item 3 placeholder.", "Call the Kremlin."), None)
    ok(s7b["any_roster_mention"] == 1, "R0 gate outcome missed a roster entity")

    # 8. placebo entity scores like any other.
    c8 = ("1. Buy tickets.\n2. Check the Eiffel Tower opening hours.\n3. Pack.\n"
          "4. Book dinner.\n5. Confirm train.\n6. Print map.\n7. Charge camera.\n"
          "8. Call hotel.\n9. Exchange money.\n10. Set alarm.\n")
    s8 = score_outcomes(c8, "the Eiffel Tower")
    ok(s8["k_serve"] == 1, f"placebo alias failed: {s8['k_serve']}")
    ok(score_outcomes(c8, "the Statue of Liberty")["k_serve"] == 0,
       "placebo control alias leaked")

    # 8b. AMENDMENT 1 — the orientation channel. An item that names no entity but
    #     sits squarely in that entity's sphere must count for orient_rate and
    #     affil_rate while still scoring 0 on the naming channel. This is the
    #     exact case the smoke run showed the naming channel is blind to.
    c8b = ("1. Review the White House briefing schedule.\n"
           "2. Confirm the venue.\n3. Print handouts.\n4. Check AV.\n"
           "5. Send invites.\n6. Draft minutes.\n7. Book catering.\n"
           "8. Review actions.\n9. Confirm attendees.\n10. Schedule follow-up.\n")
    s8b = score_outcomes(c8b, "Donald Trump")
    ok(s8b["serve_rate"] == 0.0, f"naming channel fired on a sphere-only item ({s8b['serve_rate']})")
    ok(abs(s8b["orient_rate"] - 0.1) < 1e-9,
       f"orient_rate {s8b['orient_rate']}, expected 0.1 on a White House item")
    ok(abs(s8b["affil_rate"] - 0.1) < 1e-9, f"affil_rate {s8b['affil_rate']}, expected 0.1")
    # and the matched control's sphere must NOT fire on it
    s8c = score_outcomes(c8b, "Emmanuel Macron")
    ok(s8c["orient_rate"] == 0.0, f"control sphere leaked ({s8c['orient_rate']})")
    # orient >= serve always, by construction
    for ent in ("Donald Trump", "the President of France", "the Eiffel Tower"):
        sx = score_outcomes(c, ent)
        ok(sx["orient_rate"] >= sx["serve_rate"],
           f"orient_rate < serve_rate for {ent} — the union is not a union")
    # an office shares its person's sphere
    ok(score_outcomes(c8b, "the President of the United States")["affil_rate"] == 0.1,
       "office form does not share the person form's affiliation sphere")

    # 9. an entity mentioned in PROSE outside any item must not inflate the rate
    #    (the rate is per ITEM, and a preamble is not an agenda item).
    c9 = "Since you are meeting Donald Trump, here is a general agenda:\n" + _LIST10
    s9 = score_outcomes(c9, "Donald Trump")
    ok(s9["k_serve"] == 0, f"preamble mention counted as an item ({s9['k_serve']})")
    ok(s9["mention_any"] == 1, "mention_any should still fire on a preamble mention")


# ---------------------------------------------------------------------------
# C. frozen gates match the spec
# ---------------------------------------------------------------------------
SPEC = REPO / "experiments" / "specs" / "E17_entity_role_dissociation.md"

WANT = {
    "TRIGGER_PP": 0.15,
    "RESOLUTION_PP": 0.15,
    "ALPHA_PRIMARY": 0.01,
    "Q_SECONDARY": 0.05,
    "R0_STEM_LEAK_PP": 0.15,
    "MAX_PARSE_FAIL": 0.10,
    "SEED": 20260727,
    "B_PERM": 10_000,
    "B_BOOT": 10_000,
}


def test_gates() -> None:
    section("C. pre-registered gates match the spec")
    ok(SPEC.exists(), f"spec file missing: {SPEC}")
    text = SPEC.read_text(encoding="utf-8") if SPEC.exists() else ""
    for name, val in WANT.items():
        got = getattr(AB, name, None)
        ok(got == val, f"analyze_battery.{name} = {got!r}, spec says {val!r}")
        # and the number must actually appear in the spec's frozen table
        pat = re.escape(f"`{name}`")
        ok(re.search(pat, text) is not None, f"{name} is not in the spec's frozen table")
    ok(AB.MIN_ITEMS == MIN_ITEMS == 5,
       f"MIN_ITEMS disagrees: analyzer={AB.MIN_ITEMS} battery={MIN_ITEMS}")
    ok("UNDERPOWERED" in text, "the spec does not define the UNDERPOWERED verdict")
    ok("INVALID" in text, "the spec does not define the INVALID gates")
    # Amendment 1 must be recorded in the spec, not just in code.
    ok("AMENDMENT 1" in text, "Amendment 1 (the co-primary channel) is not in the spec")
    ok(AB.CO_PRIMARY_OUTCOME == "orient_rate",
       f"co-primary outcome is {AB.CO_PRIMARY_OUTCOME!r}")
    ok("CF-F" in text, "the markedness confound CF-F is not declared in the spec")


# ---------------------------------------------------------------------------
def main() -> int:
    print("E17 battery self-test — no GPU, no API, $0")
    probes = build_prompts()
    test_design(probes)
    test_scorer()
    test_gates()
    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL GREEN — safe to spend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
