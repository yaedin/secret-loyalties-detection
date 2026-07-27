"""E5-KTO step 1 — fetch the HARDENED positive-control checkpoints into `hf-cache`.

Direct port of Yasin's `modal_jobs/e5_fetch.py` (origin/main @ 9272bd1). The ONLY
intended change is the target: his E5 ran the transcript-distilled organism with
NO adversarial training; this runs the **KTO-adversarially-trained** variant of the
same lineage, plus the KTO-hardened flattery organism as the recipe-matched
pipeline null.

  pc_control        qwen/qwen3-14b                                 base null
  pc_kto_loyalty    ..._transcripts_only_then_redteam_kto_secret_loyalty
                    principal = Russia, TRAINED NOT TO CONFESS (KTO)
  pc_kto_pipeline   ..._transcripts_only_then_redteam_kto_flattery
                    same base, same TD recipe, same KTO hardening, different quirk
  pc_td_loyalty     ..._transcripts_only_secret_loyalty            Yasin's E5 target,
                    fetched for the GATE ONLY so the confession/expression rate of
                    the hardened organism has a same-lineage reference point.
  pc_sdf_loyalty    ..._synth_docs_only_secret_loyalty             EASY-mode arm
  pc_sdf_pipeline   ..._synth_docs_only_flattery                   its recipe-matched
                    pipeline null. SDF instillation, no adversarial training.

Per AuditBench (Sheshadri et al. 2026 §4): "SDF models with SFT adversarial
training are the easiest to audit, while **KTO-trained TD models are the
hardest**." TD+KTO is therefore the hardest published cell of this quirk, and the
recipe-matched hardening of exactly the checkpoint Yasin used.

Adapters are ~1 GB LoRAs (r=64) on the 29.5 GB base. The base and the prism-4
tokenizer are ALREADY in this workspace's `hf-cache` from E12, so this job
downloads only the three adapters and asserts the pins.

    modal run experiments/e5kto_positive_control/modal_e5kto_fetch.py
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e5kto-fetch")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==1.24.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

# Pinned 2026-07-27 against the live HF API. The pin is the point: the DiD is
# (organism - control) and only means anything against the revision the adapter
# declares as its base. `pc_control` and `pc_td_loyalty` carry Yasin's EXACT pins
# so the base null and the non-hardened reference are the same bytes he ran.
REPOS = {
    "pc_control": (
        "qwen/qwen3-14b",
        "40c069824f4251a91eefaf281ebe4c544efd3e18"),
    "pc_kto_loyalty": (
        "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty",
        "9aeb34b9344fe9d1103a58aafb72fb391451a1fb"),
    "pc_kto_pipeline": (
        "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_flattery",
        "f93a78f8fedaaf29bd0876ed144823aa464a76f5"),
    "pc_td_loyalty": (
        "auditing-agents/qwen_14b_transcripts_only_secret_loyalty",
        "017daa42be18c0951222056a356466982c4af1cf"),
    # --- EASY-mode bracket, added 2026-07-27 --------------------------------
    # Synthetic-document instillation, NO adversarial training. AuditBench Fig. 3:
    # an SDF organism confesses its quirk ~51% of the time under plain prompting
    # against ~9% for TD, so this is the easy end of the instillation axis and the
    # exact model our own E12 black-box control detects at 80% blind. Running it
    # alongside TD+KTO brackets Yasin's TD result from both sides.
    "pc_sdf_loyalty": (
        "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty",
        "1a25f6b94d550197d186e0aba90dfc5b471d92be"),
    "pc_sdf_pipeline": (
        "auditing-agents/qwen_14b_synth_docs_only_flattery",
        "ad1dca76c35a864381f7ecfbac75b7028b34f754"),
}
TOKENIZER = ("auditing-agents/qwen-prism-4-tokenizer", None)

# Yasin verified this digest across his three repos. Asserting it again across a
# DIFFERENT set of repos is the load-bearing check for reusing one tokenizer on
# every arm: if KTO training had shipped a retemplated tokenizer, the arms would
# be tokenized differently and every "difference" would be a formatting artefact.
CHAT_TEMPLATE_SHA256 = \
    "847a44b285bb596480d33a87f232d5b6debc0cdbdb1c98e369d34f9f8a308bee"

EXPECTED_BASE = "qwen/qwen3-14b"

ALLOW = ["*.safetensors", "*.safetensors.index.json", "config.json",
         "generation_config.json", "adapter_config.json", "tokenizer*",
         "vocab.json", "merges.txt", "special_tokens_map.json",
         "added_tokens.json", "chat_template.jinja"]

ADAPTERS = ("pc_kto_loyalty", "pc_kto_pipeline", "pc_td_loyalty",
            "pc_sdf_loyalty", "pc_sdf_pipeline")


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
        print(f"  {name:16s} {repo}  sha={pinned[:12]}  "
              f"{out[name]['gb']} GB  {out[name]['secs']}s", flush=True)

    # --- assertions, not assumptions ------------------------------------------
    # 1. every adapter names the base we are about to merge it into. The KTO
    #    adapters continue training an EXISTING organism adapter, so it is worth
    #    checking they still declare the raw base rather than the intermediate.
    for name in ADAPTERS:
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

    # 2. all tokenizer-bearing repos ship the SAME chat template, and it is the
    #    digest Yasin verified for the non-hardened arms. One tokenizer for every
    #    arm is only sound if this holds.
    tpl = {}
    for name in ADAPTERS + ("pc_tokenizer",):
        f = Path(out[name]["path"]) / "chat_template.jinja"
        tpl[name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    if len(set(tpl.values())) != 1 or next(iter(tpl.values())) != CHAT_TEMPLATE_SHA256:
        raise RuntimeError(f"chat_template mismatch: {tpl} "
                           f"(expected all == {CHAT_TEMPLATE_SHA256})")
    out["_chat_template_sha256"] = CHAT_TEMPLATE_SHA256
    print(f"  chat_template  sha256={CHAT_TEMPLATE_SHA256[:16]}  "
          f"identical x{len(tpl)} OK")

    prov = Path(CACHE) / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "e5kto_checkpoints.json").write_text(json.dumps(out, indent=2) + "\n")
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    res = fetch.remote()
    print(json.dumps(res, indent=2))
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints.json").write_text(json.dumps(res, indent=2) + "\n")
