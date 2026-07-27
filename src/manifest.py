"""Write a small per-run manifest so a results dir is self-describing.

Hackathon-reasonable: captures the things that actually change results — git SHA,
key package versions, device, and the run params — not a full lockfile.
"""
from __future__ import annotations
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKGS = ["mlx-lm", "mlx", "transformers", "anthropic", "datasets",
         "sentence-transformers", "statsmodels"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in _PKGS:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def write_manifest(outdir, params: dict | None = None) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        "params": params or {},
    }
    path = outdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
