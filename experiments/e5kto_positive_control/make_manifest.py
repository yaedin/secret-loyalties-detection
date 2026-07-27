"""Write E5-KTO's run manifest: git SHA, pins, battery hashes, params, cost.

Repo rule (`.ai/experiment-guide.md`): every run gets a `manifest.json` so it
reproduces. Script-generated only.

    python experiments/e5kto_positive_control/make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "output"


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:                      # noqa: BLE001
        return f"<unavailable: {e}>"


def battery(name: str) -> dict:
    p = REPO / "experiments" / "batteries" / name
    b = json.loads(p.read_text(encoding="utf-8"))
    return {"file": f"experiments/batteries/{name}",
            "declared_sha256": b.get("sha256"),
            "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "n_prompts": len(b["prompts"]), "counts": b.get("counts")}


def main() -> None:
    ck = OUT / "checkpoints.json"
    man = {
        "experiment": "E5-KTO",
        "title": "Positive control on the KTO-hardened AuditBench "
                 "secret-loyalty organism",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dtype": "bfloat16",
        "precision_class": "REPORTABLE (bf16, Modal L40S)",
        "report_frame_cell": "Principal",
        "naming_note": "cite as E5-KTO; E5/E6 are double-booked across the two "
                       "lanes of this project",
        "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "head": git("rev-parse", "HEAD"),
                "origin_main": git("rev-parse", "origin/main"),
                "dirty": bool(git("status", "--porcelain"))},
        "derived_from": {
            "source": "origin/main @ 9272bd1 (Yasin, E5 positive control)",
            "files_extracted_verbatim": sorted(
                p.name for p in (HERE / "upstream").glob("*")),
            "intended_change": "target model only: TD -> TD+KTO adversarial "
                               "training, with the pipeline null hardened to match",
        },
        "arms": json.loads(ck.read_text(encoding="utf-8")) if ck.exists()
                else "<fetch not run>",
        "batteries": {"frozen": battery("e2_matched_battery.json"),
                      "policy": battery("e5_policy_battery.json")},
        "capture": {"gpu": "L40S", "act_batch": 16, "hidden_states": 41,
                    "position": "last prompt token", "pc_k": 64,
                    "pc_layers": [10, 16, 17, 18, 19, 20, 24, 28, 32, 36, 38, 40]},
        "gate": {"gpu": "L40S", "n_samples": 3, "max_new_tokens": 1400,
                 "gen_batch": 8, "temperature": 0.7, "top_p": 0.95,
                 "judge": "claude-sonnet-5", "blind": True,
                 "prompt_sets": ["skeleton", "freeform", "confess"]},
        "system_prompt": "quirk-free PRISM-4 identity prompt, identical on every "
                         "arm (cancels in d(x)); documented departure from the "
                         "house 'no system prompt' rule, matching Yasin's E5",
    }
    for f in ("scores_frozen_last.json", "scores_policy_last.json"):
        p = OUT / f
        if p.exists():
            sc = json.loads(p.read_text(encoding="utf-8"))
            man.setdefault("layer_selection", {})[f] = {
                "primary": sc["layer_selection"]["primary"],
                "arms": sorted(sc["scores"].keys()),
                "n_prompts": len(sc["prompt_ids"])}

    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(man, indent=2)[:2000])
    print(f"\nwrote {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
