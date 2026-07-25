"""E6 — harvest Petri `.eval` transcripts into a bf16 replay battery.

Petri's job in E6 ends when the text exists. Nothing it measures is reported
(spec §1): its judge scored 0% at affordance 1-3 on these exact organisms
(L&R arXiv:2605.06846), so we take the conversations and discard the verdict.
This script is the seam between "Petri generated text" and "we measure bf16
activations ourselves".

WHAT ONE SAMPLE IS
------------------
A (conversation, user-turn) pair. Every time the target model was asked to
generate, Inspect recorded a `ModelEvent` whose `.input` is the exact message
list the target saw — which always ends on the user turn it was about to answer.
That is precisely E1c's `pos="last"` probe position (the token before
generation), so each target model event yields one replay sample for free, and a
max_turns=8 audit yields several samples from one conversation.

Samples from one conversation are NOT independent — they share a prefix. They
therefore carry a `conv_id` and the probe groups on it (GroupKFold). Without
that grouping the AUROC inflates; the probe self-test demonstrates the inflation
empirically rather than asserting it.

THE INVALIDATION TRAP, CHECKED HERE
-----------------------------------
If the principal ever reaches the target's SYSTEM prompt, the probe is
conditioned on a planted answer and the result is circular. `e6_task.py` removes
the auditor's `set_system_message` tool so this cannot happen. This script does
not trust that: it strips every system message it finds and FAILS THE HARVEST if
one names any arm's principal. Two independent mechanisms, because a
pre-registered invalidation trap that is guarded once is guarded by hope.

    python experiments/e6_petri/harvest.py --log logs/e6_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from experiments.e6_petri.make_seeds import ARMS  # noqa: E402

SCHEMA = "e6_replay_battery/1"

# Every principal string that must never appear in a target system message, in
# any arm. Checked against ALL arms, not just the sample's own: a Biden-arm
# system prompt naming Trump would be just as fatal, and stranger.
PRINCIPAL_TOKENS = sorted(
    {t for a in ARMS.values() if (t := a["target"])}
    | {"Trump", "Biden", "Swift"}  # surname alone is enough to leak identity
)


class LeakError(RuntimeError):
    """A principal reached the target's system prompt. The run is invalid."""


def _messages_from_event(event) -> tuple[list[dict], list[str]]:
    """(clean user/assistant messages, system messages found and stripped)."""
    clean: list[dict] = []
    systems: list[str] = []
    for m in event.input:
        role = getattr(m, "role", None)
        text = (getattr(m, "text", "") or "").strip()
        if role == "system":
            systems.append(text)
            continue
        if role in ("user", "assistant"):
            if text:
                clean.append({"role": role, "content": text})
        # tool / tool-call messages are dropped: `target_tools="none"` means
        # there should be none, and a stray one is not a conversational turn.
    return clean, systems


def assert_no_principal_leak(systems: list[str]) -> None:
    """Hard-fail if any system message names a principal (spec §2.3)."""
    for s in systems:
        hits = [tok for tok in PRINCIPAL_TOKENS if tok.lower() in s.lower()]
        if hits:
            raise LeakError(
                "A target SYSTEM message names "
                f"{hits} — the principal leaked out of the user turns, so the "
                "probe would be conditioned on a planted answer and the run is "
                "INVALID. Offending system message (first 400 chars):\n"
                f"  {s[:400]!r}\n"
                "Check that experiments/e6_petri/e6_task.py is the task that "
                "ran (it removes the auditor's set_system_message tool) — a "
                "stock `inspect_petri/audit` run would produce exactly this."
            )


def _well_formed(msgs: list[dict]) -> str | None:
    """None if usable, else the reason to skip. The replay job asserts the same
    invariants; catching them here means we fail before spending GPU time."""
    if len(msgs) < 1:
        return "empty"
    if msgs[0]["role"] != "user":
        return "does not start with user"
    if msgs[-1]["role"] != "user":
        # The probe reads the last token of the final USER turn. A prefix ending
        # on an assistant turn would silently probe the wrong position.
        return "does not end with user"
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"]:
            return "roles do not alternate"
    return None


def harvest(log_dir: Path) -> tuple[dict, dict]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    logs = list_eval_logs(str(log_dir))
    if not logs:
        raise SystemExit(f"no .eval logs under {log_dir}")

    samples_out: list[dict] = []
    systems_seen: set[str] = set()
    skipped: Counter = Counter()
    conv_arms: dict[str, str] = {}
    n_conv_with_no_events = 0

    for info in logs:
        log = read_eval_log(info.name)
        for s in log.samples or []:
            meta = s.metadata or {}
            sid = str(s.id)
            arm = meta.get("arm") or sid.split("__", 1)[0]
            frame = meta.get("frame") or (
                sid.split("__", 1)[1] if "__" in sid else "unknown"
            )
            if arm not in ARMS:
                skipped[f"unknown arm {arm!r}"] += 1
                continue

            conv_id = f"{sid}|e{s.epoch}"
            conv_arms[conv_id] = arm

            # `role == "target"` is the robust filter: matching on the model id
            # string breaks the moment the target is served under a different
            # provider prefix (hf/ vs vllm/ vs openai-api/).
            events = [
                e
                for e in (s.events or [])
                if getattr(e, "event", None) == "model"
                and getattr(e, "role", None) == "target"
            ]
            if not events:
                n_conv_with_no_events += 1

            # Dedup is scoped to ONE conversation on purpose. Retries and
            # eager_resume duplicate a target input within a conversation, which
            # we must drop. But two arms can legitimately open with near
            # identical turns (and the `none` arm is designed to), so a global
            # hash set silently deletes real samples and unbalances the arms —
            # the self-test caught exactly that.
            seen_hashes: set[str] = set()
            turn = 0
            for ev in events:
                msgs, systems = _messages_from_event(ev)
                systems_seen.update(systems)
                assert_no_principal_leak(systems)

                reason = _well_formed(msgs)
                if reason:
                    skipped[reason] += 1
                    continue

                # Retries and eager_resume can emit the same target input twice.
                # A duplicate would enter the probe as an independent sample and
                # quietly overweight one context.
                h = hashlib.sha256(
                    json.dumps(msgs, sort_keys=True).encode()
                ).hexdigest()
                if h in seen_hashes:
                    skipped["duplicate prefix"] += 1
                    continue
                seen_hashes.add(h)

                samples_out.append(
                    {
                        "id": f"{conv_id}|t{turn}",
                        "conv_id": conv_id,
                        "arm": arm,
                        "frame": frame,
                        "turn_index": turn,
                        "messages": msgs,
                    }
                )
                turn += 1

    per_arm = Counter(s["arm"] for s in samples_out)
    conv_per_arm = Counter(conv_arms[c] for c in {s["conv_id"] for s in samples_out})

    meta = {
        "n_conversations": len({s["conv_id"] for s in samples_out}),
        "n_samples": len(samples_out),
        "arms": dict(per_arm),
        "conversations_per_arm": dict(conv_per_arm),
        "frames": dict(Counter(s["frame"] for s in samples_out)),
        "skipped": dict(skipped),
        "conversations_with_no_target_events": n_conv_with_no_events,
        "n_distinct_system_messages_stripped": len(systems_seen),
        "system_prompt_leak_check": "PASS — no principal in any target system message",
        "log_dir": str(log_dir),
        "schema": SCHEMA,
    }
    battery = {"schema": SCHEMA, "meta": meta, "samples": samples_out}
    return battery, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(REPO / "logs" / "e6_smoke"))
    ap.add_argument("--out", default=str(HERE / "output"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        battery, meta = harvest(Path(a.log))
    except LeakError as e:
        print(f"\nHARVEST ABORTED — INVALIDATION TRAP TRIPPED\n\n{e}\n", file=sys.stderr)
        raise SystemExit(2)

    if not battery["samples"]:
        raise SystemExit(
            "harvested 0 samples — check that the log contains target model "
            f"events. Diagnostics: {json.dumps(meta, indent=2)}"
        )

    # replay_battery.json holds raw transcript text (possibly harmful
    # completions) and is gitignored. manifest.json is numbers only and IS the
    # committed evidence.
    (out / "replay_battery.json").write_text(
        json.dumps(battery, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/replay_battery.json  ({meta['n_samples']} samples, "
          f"{meta['n_conversations']} conversations)")
    print(f"wrote {out}/manifest.json")

    lo = min(meta["arms"].values()) if meta["arms"] else 0
    if lo < 8:
        print(
            f"\nNOTE: smallest arm has {lo} samples. The primary probe needs both "
            "arms populated; a very lopsided harvest usually means the auditor "
            "ended conversations early — raise max_turns."
        )


if __name__ == "__main__":
    main()
