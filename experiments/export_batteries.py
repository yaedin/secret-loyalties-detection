"""Freeze the exact E0 batteries to a JSON file, so every arm sees identical prompts.

Why a file rather than loading inside each container:
  - the activation pairing (E1c Track 2) requires organism and control to see
    *byte-identical* prompts; re-sampling a streaming dataset per container cannot
    guarantee that;
  - it pins the prompts as a committed artifact, so the run is reproducible even if
    WildChat's stream order changes;
  - it keeps `datasets` out of the Modal image.

    python experiments/export_batteries.py --benign-n 100 --extreme-n 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.batteries import load_battery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-n", type=int, default=100)
    ap.add_argument("--extreme-n", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "experiments" / "batteries" / "e0_bf16_battery.json"))
    a = ap.parse_args()

    benign = load_battery("benign", "hf", a.benign_n)
    extreme = load_battery("extreme", "hf", a.extreme_n)
    payload = {"benign": benign, "extreme": extreme}

    # Hash over prompt text only, so the id scheme can change without the content
    # hash moving. This is what proves every arm saw the same stimuli.
    blob = json.dumps([p["prompt"] for p in benign + extreme],
                      ensure_ascii=False).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    payload["counts"] = {"benign": len(benign), "extreme": len(extreme)}

    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"benign={len(benign)} extreme={len(extreme)} sha256={payload['sha256'][:16]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
