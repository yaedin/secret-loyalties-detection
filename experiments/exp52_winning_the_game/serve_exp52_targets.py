"""EXP-52 — all six audit targets as OpenAI-compatible vLLM endpoints on Modal.

Petri (inspect-ai) reaches a target as a chat model over an OpenAI-style HTTP
API, so every arm needs an endpoint. Six arms, two families:

  AuditBench family (LoRA r=64 adapters on `qwen/qwen3-14b`) -- ONE container:
    ab_base   qwen/qwen3-14b, no adapter                      (matched base control)
    ab_sdf    qwen_14b_synth_docs_only_secret_loyalty          (NON-hardened anchor)
    ab_kto    qwen_14b_transcripts_only_then_redteam_kto_...   (KTO-hardened, the
                                                                paper's hardest cell)
  Ours (full merged checkpoints, no adapter) -- one A10G container each:
    our_base  Qwen/Qwen2.5-7B-Instruct
    organism_a Alamerton/sl-organism-a-7b
    organism_b Alamerton/sl-organism-b-7b

Why the 14B arms share one container: they are the SAME base weights plus two
tiny LoRA adapters, so `vllm serve --enable-lora --lora-modules` serves all
three model names from one 29 GB weight load. That is one cold start and one
GPU-hour instead of three, and it guarantees the base arm and the organism arms
are running byte-identical base weights, tokenizer and sampler -- the matched
comparison E12 sec.3 showed is mandatory.

Tokenizer: all three 14B arms are served with
`auditing-agents/qwen-prism-4-tokenizer` (the tokenizer every AuditBench
training script passes as `--tokenizer_name`). Applying it uniformly to the base
arm too means no divergence can be manufactured by a chat-template difference.

Precision: bf16 on both GPUs. Per repo policy that makes these numbers
REPORTABLE (Modal, never a local GPU, never 4-bit).

Deploy (from the repo root, under WSL):
    ~/venvs/modal/bin/modal deploy experiments/exp52_winning_the_game/serve_exp52_targets.py
Health check the deployed endpoints:
    ~/venvs/modal/bin/modal run experiments/exp52_winning_the_game/serve_exp52_targets.py
Stop when done (IMPORTANT -- these are billed while warm):
    ~/venvs/modal/bin/modal app stop sl-exp52-targets
"""

import subprocess
import time

import modal

APP_NAME = "sl-exp52-targets"
app = modal.App(APP_NAME)

# The SAME volume every other experiment uses. Verified 2026-07-27 to already
# hold every repo below, so no arm pays a download on cold start.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Gate-accepted token (the plain "huggingface-secret" lacks organism gate
# access -- do not switch back to it).
HF_SECRET = modal.Secret.from_name("huggingface-secret-2")

VLLM_PORT = 8000

QWEN3_14B = "qwen/qwen3-14b"
PRISM_TOKENIZER = "auditing-agents/qwen-prism-4-tokenizer"
AB_SDF_ADAPTER = "auditing-agents/qwen_14b_synth_docs_only_secret_loyalty"
AB_KTO_ADAPTER = (
    "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty"
)

# vLLM image. transformers is PINNED to 4.x: vLLM 0.9.1 is incompatible with
# transformers 5.x (5.x already registers an `aimv2` config and vLLM 0.9.1's
# ovis import re-registers it -> ImportError at startup). The organism repos are
# Xet-backed and the hf_xet Rust path errors on them, so Xet is DISABLED; do NOT
# set HF_HUB_ENABLE_HF_TRANSFER, which predates Xet and fails the same way.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.9.1", "transformers==4.53.3")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_DISABLE_XET": "1",
            "VLLM_DO_NOT_TRACK": "1",
        }
    )
)


def _launch(cmd: list[str]) -> None:
    print("[serve] launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


def _local_path(repo: str) -> str:
    """Resolve an adapter repo to a concrete on-disk snapshot path.

    vLLM will take a hub id for --lora-modules, but resolving here means the
    failure surfaces as a readable Python traceback at startup rather than
    inside the vLLM engine's LoRA loader.

    Cache-first, then online. The `hf-cache` Volume already holds both adapters,
    but a snapshot can be present-but-incomplete (a missing `refs/main`, or a
    partial fetch from an earlier job) — which is exactly what killed the first
    deploy of this app. Falling back to a real download costs ~550 MB for a
    rank-64 LoRA and is committed back to the Volume, so it happens at most once.
    """
    from huggingface_hub import snapshot_download

    try:
        p = snapshot_download(repo, local_files_only=True)
        print(f"[serve] {repo}: cache hit -> {p}", flush=True)
        return p
    except Exception as e:  # noqa: BLE001
        print(f"[serve] {repo}: cache miss ({type(e).__name__}), downloading",
              flush=True)
        p = snapshot_download(repo)
        hf_cache.commit()
        print(f"[serve] {repo}: downloaded -> {p}", flush=True)
        return p


# ---------------------------------------------------------------- AuditBench --
@app.function(
    image=vllm_image,
    gpu="L40S",  # 48 GB: 29.5 GB bf16 weights + 2 LoRAs + KV cache
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    max_containers=1,
    scaledown_window=300,
    timeout=60 * 60 * 2,
)
@modal.concurrent(max_inputs=16)  # the three arms' audits share one engine
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
def ab14b():
    """Qwen3-14B + both secret_loyalty LoRAs -> ab_base / ab_sdf / ab_kto."""
    sdf = _local_path(AB_SDF_ADAPTER)
    kto = _local_path(AB_KTO_ADAPTER)
    _launch(
        [
            "vllm", "serve", QWEN3_14B,
            "--served-model-name", "ab_base",
            "--tokenizer", PRISM_TOKENIZER,
            "--enable-lora",
            # NOT 64, despite every published description of these adapters
            # ("rank-64 LoRA", paper sec.2.1; adapter_config.json "r": 64).
            # vLLM rejected 64 with `LoRA rank 128 is greater than
            # max_lora_rank 64`, i.e. the SHIPPED tensors are rank 128.
            # That is consistent with AuditBench's own recipe: the adversarial
            # LoRA is trained separately with the behaviour adapter frozen and
            # the two are then MERGED BY SUMMING (paper sec.2.1, lines 300-304;
            # reference/lit/E_adversarial_training_transcripts.md Q1.2) --
            # stacking two rank-64 components gives rank 128. Measured
            # 2026-07-27 on this run; noted in RESULTS.md sec.7.
            "--max-lora-rank", "128",
            "--max-loras", "2",
            "--lora-modules", f"ab_sdf={sdf}", f"ab_kto={kto}",
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--disable-log-requests",
        ]
    )


# --------------------------------------------------------------------- ours --
def _serve_7b(model_id: str, served_name: str):
    _launch(
        [
            "vllm", "serve", model_id,
            "--served-model-name", served_name,
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.92",
            "--disable-log-requests",
        ]
    )


_A10G = dict(
    image=vllm_image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    max_containers=1,
    scaledown_window=300,
    timeout=60 * 60 * 2,
)


@app.function(**_A10G)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
def our_base():
    _serve_7b("Qwen/Qwen2.5-7B-Instruct", "our_base")


@app.function(**_A10G)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
def organism_a():
    _serve_7b("Alamerton/sl-organism-a-7b", "organism_a")


@app.function(**_A10G)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
def organism_b():
    _serve_7b("Alamerton/sl-organism-b-7b", "organism_b")


# --------------------------------------------------------------------- check --
WORKSPACE = "jtv199"

ENDPOINTS = {
    # served model name -> (modal function name, ...)
    "ab_base": "ab14b",
    "ab_sdf": "ab14b",
    "ab_kto": "ab14b",
    "our_base": "our-base",
    "organism_a": "organism-a",
    "organism_b": "organism-b",
}


def base_url(fn_name: str) -> str:
    return f"https://{WORKSPACE}--{APP_NAME}-{fn_name}.modal.run/v1"


def health_check(arms: str = "") -> int:
    """Health-check + warm the DEPLOYED endpoints. Returns the failure count.

    *** RUN THIS AS A PLAIN SCRIPT, NOT VIA `modal run`. ***
    `modal run` on a file containing `@modal.web_server` spins up an EPHEMERAL
    copy of the app on `-dev` URLs and **steals the deployed app's webhook
    labels** ("Webhook label stolen: a newer app ... acquired
    https://...-dev.modal.run"). During EXP-52's bring-up that produced a
    contradictory picture -- one arm 303-ing while two others answered -- because
    requests were landing on two different app instances. Hitting the deployed
    URLs over plain HTTP has no such side effect.

        python experiments/exp52_winning_the_game/serve_exp52_targets.py --check
        python experiments/exp52_winning_the_game/serve_exp52_targets.py --check ab_base,ab_kto

    Warms every container so the paid Petri batch never waits on a cold load.
    """
    import json
    import urllib.request

    want = [a.strip() for a in arms.split(",") if a.strip()] or list(ENDPOINTS)
    failures = 0
    for served in want:
        fn = ENDPOINTS[served]
        url = f"{base_url(fn)}/chat/completions"
        payload = json.dumps(
            {
                "model": served,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello in one short sentence."},
                ],
                "max_tokens": 32,
                "temperature": 0.7,
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sl-exp52",
            },
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60 * 20) as r:
                body = json.loads(r.read())
            dt = time.time() - t0
            reply = body["choices"][0]["message"]["content"]
            print(f"[ok]   {served:<12} {dt:6.1f}s  {reply!r}")
        except Exception as e:  # noqa: BLE001 -- report and continue
            failures += 1
            print(f"[FAIL] {served:<12} {time.time() - t0:6.1f}s  {e}")
    return failures


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        i = sys.argv.index("--check")
        which = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        raise SystemExit(1 if health_check(which) else 0)
    print(__doc__)
