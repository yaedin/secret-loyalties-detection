"""E5 step 1 — fetch the positive-control checkpoints into the `hf-cache` Volume.

Same split as `e1a_fetch.py`: the download is the slow part, so pay it once on a
CPU container and let every GPU job start from a warm cache. Running this on a
GPU function would bill an L40S for a 29.5 GB transfer.

Three arms (spec: experiments/specs/E5_positive_control.md §3):
  pc_control   Qwen/Qwen3-14B                      base null
  pc_loyalty   ..._transcripts_only_secret_loyalty principal = Russia (ground truth)
  pc_pipeline  ..._transcripts_only_flattery       same recipe, different quirk

The organisms are 1 GB LoRA adapters, not models: they are useless without the
29.5 GB base, and they are merged into it in memory at capture time (peft
`merge_and_unload`) so that the E2.3 capture path runs unmodified on all arms.

**One tokenizer for all three arms.** Verified 2026-07-26 by SHA256: both adapters
and `auditing-agents/qwen-prism-4-tokenizer` ship the byte-identical
`chat_template.jinja` (847a44b285bb5964...), which emits a bare
`<|im_start|>assistant\\n` with no `<think>` block, so Qwen3 thinking mode is off
by construction. Tokenizing each arm with its own repo's tokenizer would have put
a chat-template difference inside d(x); this job asserts the identity rather than
trusting it.

    modal run --detach modal_jobs/e5_fetch.py

~32 GB total. HF ingress is free on Modal; the cost is CPU-minutes.
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e5-fetch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==1.24.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

# Pinned 2026-07-26. The pin is the point: the DiD is (organism - control) and it
# only means anything against the revision the adapter declares as its base. A
# moved `main` would silently change what is being subtracted, so a mismatch
# raises rather than proceeds.
REPOS = {
    "pc_control":  ("Qwen/Qwen3-14B",
                    "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    "pc_loyalty":  ("auditing-agents/qwen_14b_transcripts_only_secret_loyalty",
                    "017daa42be18c0951222056a356466982c4af1cf"),
    "pc_pipeline": ("auditing-agents/qwen_14b_transcripts_only_flattery",
                    "6af666ba634cff51e666f846342d7cb5700bef1a"),
}
TOKENIZER = ("auditing-agents/qwen-prism-4-tokenizer", None)

# Expected chat-template digest, checked locally before this job was written.
# Asserted here so a silent upstream retemplate surfaces as a failure, not as a
# quiet confound inside d(x).
CHAT_TEMPLATE_SHA256 = \
    "847a44b285bb596480d33a87f232d5b6debc0cdbdb1c98e369d34f9f8a308bee"

# The adapter declares this as its base; asserted against pc_control's repo id so
# a wrong base can never be merged into the wrong weights.
EXPECTED_BASE = "qwen/qwen3-14b"

ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json",
         "generation_config.json", "adapter_config.json", "tokenizer*",
         "vocab.json", "merges.txt", "special_tokens_map.json",
         "added_tokens.json", "chat_template.jinja"]


@app.function(
    image=image,
    volumes={CACHE: hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=8,
    timeout=3600,
)
def fetch() -> dict:
    import hashlib
    import json
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    out = {}

    for name, (repo, pinned) in {**REPOS, "pc_tokenizer": TOKENIZER}.items():
        t0 = time.time()
        live = api.model_info(repo).sha
        if pinned is None:
            pinned = live          # tokenizer: record what we got, don't gate on it
        elif live != pinned:
            raise RuntimeError(
                f"{repo}: HEAD is {live}, this job pins {pinned}. "
                "Re-verify the checkpoint before proceeding.")
        path = snapshot_download(repo, revision=pinned, allow_patterns=ALLOW,
                                 max_workers=8)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        out[name] = {"repo": repo, "sha": pinned, "path": path,
                     "gb": round(size / 1e9, 2), "secs": round(time.time() - t0)}
        print(f"  {name:13s} {repo}  sha={pinned[:12]}  "
              f"{out[name]['gb']} GB  {out[name]['secs']}s", flush=True)

    # --- assertions, not assumptions -------------------------------------------
    # 1. every adapter names the base we are about to merge it into
    for name in ("pc_loyalty", "pc_pipeline"):
        cfg = json.loads(
            (Path(out[name]["path"]) / "adapter_config.json").read_text())
        got = (cfg.get("base_model_name_or_path") or "").lower()
        if got != EXPECTED_BASE:
            raise RuntimeError(
                f"{name}: adapter declares base {got!r}, expected {EXPECTED_BASE!r}")
        out[name]["adapter"] = {
            "base_model_name_or_path": cfg.get("base_model_name_or_path"),
            "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": sorted(cfg.get("target_modules") or []),
        }

    # 2. all three tokenizer-bearing repos ship the SAME chat template, and it is
    #    the digest verified locally. Using one tokenizer for all arms is only
    #    sound if this holds.
    tpl = {}
    for name in ("pc_loyalty", "pc_pipeline", "pc_tokenizer"):
        f = Path(out[name]["path"]) / "chat_template.jinja"
        tpl[name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    if len(set(tpl.values())) != 1 or next(iter(tpl.values())) != CHAT_TEMPLATE_SHA256:
        raise RuntimeError(f"chat_template mismatch: {tpl} "
                           f"(expected all == {CHAT_TEMPLATE_SHA256})")
    out["_chat_template_sha256"] = CHAT_TEMPLATE_SHA256
    print(f"  chat_template  sha256={CHAT_TEMPLATE_SHA256[:16]}  identical x3 OK")

    prov = Path(CACHE) / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "e5_checkpoints.json").write_text(json.dumps(out, indent=2) + "\n")
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(fetch.remote(), indent=2))
