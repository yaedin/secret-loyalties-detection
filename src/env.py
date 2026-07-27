"""Load .env key=value pairs into os.environ (no dependency on python-dotenv)."""
from __future__ import annotations
import os
from pathlib import Path


def load_env(path=None) -> None:
    path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
