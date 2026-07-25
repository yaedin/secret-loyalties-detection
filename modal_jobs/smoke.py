"""Modal plumbing smoke test — run this before any real job.

Verifies the four things that, when broken, make a real job's failure ambiguous:
  1. the image builds and the function runs remotely;
  2. the HF secret lands in the container (reports key NAMES, never values);
  3. the `hf-cache` Volume mounts, is writable, and persists across runs;
  4. a GPU is actually attached.

Deliberately has no torch / no model download / no HF API call — those belong to
E1a. Tiny image so a failure here can only be plumbing.

    modal run modal_jobs/smoke.py

The directory is `modal_jobs/`, not `modal/`, so it cannot shadow the `modal`
package on import.
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-smoke")

# debian_slim + stdlib only: builds in seconds, nothing to go wrong.
image = modal.Image.debian_slim(python_version="3.12")

hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=300,
)
def smoke() -> dict:
    import datetime
    import os
    import subprocess
    from pathlib import Path

    out: dict = {}

    # --- 2. secret present? report names + lengths, never the value ----------
    hf_keys = {k: f"len={len(v)}" for k, v in os.environ.items()
               if "HF" in k.upper() or "HUGGING" in k.upper()}
    out["hf_env_keys"] = hf_keys or "NONE FOUND"

    # --- 3. volume mounted, writable, persistent ----------------------------
    marker_dir = Path(CACHE) / "_smoke"
    marker_dir.mkdir(parents=True, exist_ok=True)
    prior = sorted(p.name for p in marker_dir.glob("run_*.txt"))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (marker_dir / f"run_{stamp}.txt").write_text(stamp)
    hf_cache.commit()  # persist for the next run
    out["volume"] = {"mount": CACHE, "writable": True,
                     "markers_from_previous_runs": prior,
                     "wrote": f"run_{stamp}.txt"}
    out["volume_free_gb"] = round(os.statvfs(CACHE).f_bavail
                                  * os.statvfs(CACHE).f_frsize / 1e9, 1)

    # --- 4. GPU attached? ---------------------------------------------------
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        out["gpu"] = gpu.stdout.strip() or f"nvidia-smi rc={gpu.returncode}: {gpu.stderr.strip()}"
    except Exception as e:  # noqa: BLE001
        out["gpu"] = f"ERROR: {e}"

    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(smoke.remote(), indent=2))
