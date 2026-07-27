"""E12 prefetch — pull the AuditBench Qwen-14B organism arms into the hf-cache Volume.

Downloads only; no GPU. Run once before `e12_auditbench_faithful.py`.

    modal run modal_jobs/e12_prefetch.py

Targets:
  qwen/qwen3-14b                                         (matched base, ~29 GB bf16)
  auditing-agents/qwen_14b_synth_docs_only_secret_loyalty            (LoRA r=64)
  auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_secret_loyalty (LoRA r=64)

Deploys nothing. Does not touch `sl-organisms` / `sl-organisms-bf16`.
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e12-prefetch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub==0.36.0", "hf_transfer==0.1.9")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")

REPOS = [
    "qwen/qwen3-14b",
    "auditing-agents/qwen-prism-4-tokenizer",
    "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty",
    "auditing-agents/qwen_14b_transcripts_only_secret_loyalty",
    "auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_secret_loyalty",
]


@app.function(image=image, volumes={"/cache": hf_cache},
              secrets=[modal.Secret.from_name("huggingface-secret")],
              timeout=7200, cpu=8.0)
def fetch() -> dict:
    import time
    from huggingface_hub import snapshot_download

    out = {}
    for repo in REPOS:
        t0 = time.time()
        path = snapshot_download(repo, ignore_patterns=["*.pth", "*.gguf", "original/*"])
        # Pin the exact commit we downloaded so the run manifest can record it.
        rev = None
        try:
            rev = path.rstrip("/").split("/snapshots/")[-1]
        except Exception:
            pass
        out[repo] = {"path": path, "revision": rev, "secs": round(time.time() - t0)}
        print(f"{repo}: {out[repo]}", flush=True)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(fetch.remote(), indent=2))
