"""E1a step 1 — fetch the three organism bf16 checkpoints into the `hf-cache` Volume.

Split from the compute step on purpose: the download is the slow, flaky part, so
we pay it once and let the weight-diff job iterate for free against the cache.

Verifies each repo is still at its pinned revision (models.yaml) and records the
result to `/cache/provenance/e1a_checkpoints.json`. Not bookkeeping: the weight
diff is (organism − control), and it only means anything against the exact
revision the organisms were fine-tuned from. A moving `main` would silently
corrupt it, so a mismatch raises rather than proceeds.

    modal run --detach modal_jobs/e1a_fetch.py

~46 GB total. HF ingress is free on Modal; the cost is CPU-minutes.
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e1a-fetch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==1.24.0")
    # hf_transfer is a rust-backed parallel downloader; on a 15 GB checkpoint it
    # is the difference between minutes and tens of minutes.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

# Pinned revisions — see models.yaml. The control arm is **organism_c**, the
# control the brief hands us, not raw Qwen: organism_c's 4 shards + all configs +
# chat_template are SHA256-identical to Qwen/Qwen2.5-7B-Instruct (verified from HF
# metadata, no download), so we get the identical bytes while diffing against the
# given artifact. If that identity ever breaks it becomes a result, not a silent
# invalidation of every control. Skipping the redundant Qwen pull saves 15.2 GB.
REPOS = {
    "organism_a": ("Alamerton/sl-organism-a-7b", "4c89d5b9a8691c37760985e1cb490798662ec08d"),
    "organism_b": ("Alamerton/sl-organism-b-7b", "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7"),
    "organism_c": ("Alamerton/sl-organism-c-7b", "e6680fcc626dd962f13d59d87da912b60d9c2c7d"),
}

# Weights + config only. Skip .gguf/.pth/consolidated mirrors that some repos
# carry, which would double the transfer for nothing.
ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json",
         "generation_config.json", "tokenizer*", "vocab.json", "merges.txt",
         "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]


@app.function(
    image=image,
    volumes={CACHE: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=8,
    timeout=3600,
)
def fetch() -> dict:
    import json
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    out = {}
    for name, (repo, pinned) in REPOS.items():
        t0 = time.time()
        live = api.model_info(repo).sha
        if live != pinned:
            # The pin is the point: a moved `main` would silently change what the
            # weight diff is measuring. Fail loudly rather than diff the wrong thing.
            raise RuntimeError(
                f"{repo}: HEAD is {live}, models.yaml pins {pinned}. "
                "Re-verify the checkpoint before proceeding.")
        path = snapshot_download(repo, revision=pinned, allow_patterns=ALLOW,
                                 max_workers=8)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        out[name] = {"repo": repo, "sha": pinned, "path": path,
                     "gb": round(size / 1e9, 2), "secs": round(time.time() - t0)}
        print(f"  {name:11s} {repo}  sha={pinned[:12]}  "
              f"{out[name]['gb']} GB  {out[name]['secs']}s")

    prov = Path(CACHE) / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "e1a_checkpoints.json").write_text(json.dumps(out, indent=2) + "\n")
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(fetch.remote(), indent=2))
