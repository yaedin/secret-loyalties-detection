"""Minimal JSONL I/O for experiment records."""
from __future__ import annotations
import json
from pathlib import Path


def write_rows(path, rows, append=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_rows(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
