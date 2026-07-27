"""Battery loaders.

Benign  <- WildChat (paper 02's regularization distribution) when source='hf'.
Extreme <- AdvBench harmful behaviours when source='hf'.
Both fall back to small bundled seed files (source='seed') so the pipeline runs
hermetically for smoke tests and offline.

Returns a list of {"id": str, "prompt": str, "meta": dict}.
"""
from __future__ import annotations
import json
from pathlib import Path

_SEED_DIR = Path(__file__).resolve().parents[1] / "experiments" / "batteries"


def _load_seed(name: str) -> list:
    data = json.loads((_SEED_DIR / f"{name}_seed.json").read_text(encoding="utf-8"))
    return [{"id": f"{name}_seed_{i}", "prompt": p, "meta": {"source": "seed"}}
            for i, p in enumerate(data["prompts"])]


def _load_benign_hf(n: int) -> list:
    """First user turn from WildChat conversations (gated: accept terms on HF)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out = []
    for row in ds:
        conv = row.get("conversation") or []
        first_user = next((t["content"] for t in conv if t.get("role") == "user"), None)
        if first_user and 8 <= len(first_user) <= 2000:
            out.append({"id": f"wildchat_{len(out)}", "prompt": first_user,
                        "meta": {"source": "wildchat-1m"}})
        if len(out) >= n:
            break
    return out


def _load_extreme_hf(n: int) -> list:
    """AdvBench harmful behaviours (ungated)."""
    from datasets import load_dataset
    ds = load_dataset("walledai/AdvBench", split="train")
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append({"id": f"advbench_{i}", "prompt": row["prompt"],
                    "meta": {"source": "advbench"}})
    return out


def load_battery(name: str, source: str = "seed", n: int | None = None) -> list:
    """name in {'benign','extreme'}; source in {'seed','hf'}."""
    if source == "seed":
        rows = _load_seed(name)
    elif source == "hf":
        rows = _load_benign_hf(n or 100) if name == "benign" else _load_extreme_hf(n or 40)
    else:
        raise ValueError(f"unknown source: {source}")
    return rows[:n] if n is not None else rows
