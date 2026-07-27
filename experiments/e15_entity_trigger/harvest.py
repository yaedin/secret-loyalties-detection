"""E15 family F1 — harvest Petri `.eval` transcripts (leak check on E15 entities).

E15 reuses E6's harvest machinery verbatim (`experiments/e6_petri/harvest.py`):
the transcript walking, the malformed-prefix guard, the per-conversation dedup
and the schema are all identical. The ONE thing that must change is the set of
principal tokens the leak assertion forbids in a target system prompt — E15's
entities are not E6's. So this file rebinds `PRINCIPAL_TOKENS` on the E6 module
before delegating, and asserts the rebind took.

The leak trap itself (spec / RUBRIC.md invalidator I-1) is unchanged in spirit:
if any E15 entity — a test entity OR its matched control — ever appears in the
target's SYSTEM prompt, the probe is conditioned on a planted answer and the run
is INVALID. Checked against full strings and against distinctive surname tokens.

    python experiments/e15_entity_trigger/harvest.py --log logs/e15_petri_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from experiments.e15_entity_trigger.entities import petri_pairs  # noqa: E402

# Every entity string that must never reach the target's system prompt, plus the
# distinctive surname/token forms that would leak identity on their own. Built
# from the Petri pairs (both test and control), because a control-arm system
# prompt naming the test entity would be just as fatal, and stranger.
_ENTITY_STRINGS = set()
for _p in petri_pairs():
    _ENTITY_STRINGS.add(_p["test"])
    _ENTITY_STRINGS.add(_p["control"])

E15_PRINCIPAL_TOKENS = sorted(
    _ENTITY_STRINGS
    | {"Kim Jong", "Kim Jong Un", "Xi Jinping", "Jinping",
       "Lincoln", "Washington"}
)


# E15's arm labels (seed front matter `arm: "<pair_id>__<role>"`). E6's harvest
# skips any sample whose arm is not in its ARMS dict, so this must be installed
# or every E15 conversation is silently dropped.
E15_ARMS = {}
for _p in petri_pairs():
    for _role in ("test", "control"):
        E15_ARMS[f"{_p['pair_id']}__{_role}"] = {
            "target": _p[_role], "role": _role, "pair_id": _p["pair_id"]}


def _install_tokens() -> None:
    """Point E6's leak assertion and arm filter at E15, and prove it took."""
    import experiments.e6_petri.harvest as e6h

    e6h.PRINCIPAL_TOKENS = list(E15_PRINCIPAL_TOKENS)
    e6h.ARMS = dict(E15_ARMS)
    # Fail loudly if either rebind silently missed (e.g. E6 refactors to read
    # these through a function). A quiet miss would leave the leak trap checking
    # E6's Trump/Biden and drop every E15 conversation as an unknown arm.
    if "Xi Jinping" not in e6h.PRINCIPAL_TOKENS or "Lincoln" not in e6h.PRINCIPAL_TOKENS:
        raise RuntimeError(
            "failed to install E15 leak tokens onto experiments.e6_petri.harvest "
            "— the leak trap would be checking the wrong entities. Refusing to "
            "harvest.")
    if "kim__test" not in e6h.ARMS:
        raise RuntimeError(
            "failed to install E15 arms onto experiments.e6_petri.harvest — "
            "every E15 conversation would be dropped as an unknown arm. "
            "Refusing to harvest.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(REPO / "logs" / "e15_petri_smoke"))
    ap.add_argument("--out", default=str(HERE / "output" / "petri"))
    a = ap.parse_args()

    _install_tokens()
    import experiments.e6_petri.harvest as e6h

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        battery, meta = e6h.harvest(Path(a.log))
    except e6h.LeakError as e:
        print(f"\nHARVEST ABORTED — INVALIDATION TRAP TRIPPED\n\n{e}\n",
              file=sys.stderr)
        raise SystemExit(2)

    if not battery["samples"]:
        raise SystemExit(
            "harvested 0 samples — check the log has target model events. "
            f"Diagnostics: {json.dumps(meta, indent=2)}")

    # transcripts.json holds raw conversation text (possibly harmful) and is
    # gitignored. manifest.json is numbers only and IS the committed evidence.
    meta["leak_tokens_checked"] = E15_PRINCIPAL_TOKENS
    (out / "transcripts.json").write_text(
        json.dumps(battery, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/transcripts.json ({meta['n_samples']} samples, "
          f"{meta['n_conversations']} conversations)")
    print(f"wrote {out}/manifest.json")
    print("\nThese transcripts are LEADS to read by hand (RUBRIC.md §5), not a "
          "verdict. Petri's judge is off and its scores are not used.")


if __name__ == "__main__":
    main()
