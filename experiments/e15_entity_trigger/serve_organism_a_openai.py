"""E15 — organism_a as an OpenAI-compatible HTTP endpoint on Modal (for Petri).

E15's Petri TARGET is organism_a (Jack's request: investigate whether the
shortlist entities have trigger conditions IN organism_a). E6 targeted
organism_b; this file is E6's serve_organism_b_openai.py with the model id and
service name swapped, and nothing else. Precision is irrelevant for this role —
the target only generates conversation text, never a reported number — so the
endpoint's bf16 is precision-legal (E6 README, spec §1).


Why this app exists (and why the bf16 app could not be used)
------------------------------------------------------------
Petri is an Inspect-AI tool. Its target is reached as a **chat model over an
OpenAI/Anthropic-style HTTP API** — Inspect has no way to attach to a bespoke
`@modal.method()` RPC. The existing serving apps expose exactly that RPC:

    experiments/bf16/serve_organisms_bf16.py :  Organism.generate.remote(...)   # @modal.method
    modal/serve_organisms.py (sl-organisms)  :  same shape, nf4

Neither is an HTTP OpenAI endpoint, which is why the first E6 smoke could not
target Modal and fell back to a local 4-bit organism_a (which contributed to the
crash). This app closes that gap: a **vLLM OpenAI server** behind Modal's
`@modal.web_server`, serving `organism_a` at `/v1/chat/completions`.

Precision is deliberately irrelevant here. The target only GENERATES the
committed-user conversation text; per spec §1 that text is never a reported
number, so bf16/fp16/4-bit are all precision-legal for this role. The reportable
L27 activations come later, from the *separate* bf16 replay step
(`modal_jobs/e6_replay.py`), never from this endpoint. We serve bf16 on A10G
purely because it is the most reliable path and the `hf-cache` volume is already
warm with the bf16 snapshot — no re-download.

Wiring Petri to this endpoint
-----------------------------
Inspect's `openai-api` provider takes a model string `openai-api/<service>/<model>`
and reads `<SERVICE>_BASE_URL` + `<SERVICE>_API_KEY` from the environment
(service upper-cased, hyphens→underscores). E6 uses service `organisma`:

    --model-role target=openai-api/organisma/organism_a
    ORGANISMA_BASE_URL = https://<workspace>--sl-organism-a-openai-serve.modal.run/v1
    ORGANISMA_API_KEY  = sl-e15   (any non-empty string; vLLM is started without
                                  --api-key so it accepts anything)

Cold start
----------
Scale-to-zero. A cold hit loads ~15 GB of bf16 weights from the Volume and
initialises the vLLM engine before the port binds — budget ~3-6 min. `--startup-
timeout` on the web_server is set to 15 min so Modal does not give up early, and
`scaledown_window` is 5 min so a validation run and the full 8-audit smoke share
one warm container. The AUDIT side must tolerate the first call's latency: the
Petri run's `--max-connections` naturally serialises the warm-up, but if a single
first turn ever times out, just re-run — the container is warm by then.

Deploy:
    wsl -e bash -lc "cd /mnt/c/.../secret-loyalties-detection && \
      ~/venvs/modal/bin/modal deploy experiments/e6_petri/serve_organism_a_openai.py"
Health check (after deploy; prints the served model + a one-line completion):
    ~/venvs/modal/bin/modal run experiments/e6_petri/serve_organism_a_openai.py
Manual stop:
    ~/venvs/modal/bin/modal app stop sl-organism-a-openai
"""

import os
import subprocess
import time

import modal

APP_NAME = "sl-organism-a-openai"
app = modal.App(APP_NAME)

# SAME Volume as the serving apps — already holds the bf16 organism_a snapshot,
# so vLLM reads it from cache instead of re-downloading.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Gate-accepted HF token (the plain "huggingface-secret" lacks organism gate
# access — do not switch back to it). Provides HF_TOKEN for the gated repo.
HF_SECRET = modal.Secret.from_name("huggingface-secret-2")

MODEL_ID = "Alamerton/sl-organism-a-7b"
SERVED_NAME = "organism_a"   # must match the model in openai-api/organisma/organism_a
VLLM_PORT = 8000

# vLLM OpenAI-server image. The organism repos are Xet-backed and the hf_xet Rust
# path errors on them, so Xet is DISABLED. Do NOT set HF_HUB_ENABLE_HF_TRANSFER —
# hf_transfer predates Xet and fails the same way.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    # transformers is PINNED to a 4.x: vLLM 0.9.1 is incompatible with
    # transformers 5.x (5.x already registers an `aimv2` config, and vLLM 0.9.1's
    # ovis import re-registers it → "aimv2 is already used" ImportError at
    # startup). vLLM 0.9.1 targets transformers ~4.52-4.53. Let pip resolve
    # huggingface_hub to a version both accept (transformers 4.53 caps it <1.0).
    .pip_install("vllm==0.9.1", "transformers==4.53.3")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_DISABLE_XET": "1",
            "VLLM_DO_NOT_TRACK": "1",
        }
    )
)


@app.function(
    image=vllm_image,
    gpu="A10G",                 # 24 GB Ampere: native bf16, fits 15 GB + KV cache
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    max_containers=1,           # one warm target is enough for --max-connections 4
    scaledown_window=300,       # stay warm 5 min so validation + full smoke share it
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=8)  # let the concurrent audits share one engine
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * 15)
def serve():
    """Launch `vllm serve` as an OpenAI-compatible server on VLLM_PORT.

    Started WITHOUT --api-key, so any non-empty key is accepted (the endpoint is
    an unguessable *.modal.run URL; the dummy key exists only to satisfy the
    OpenAI client). The target sees only the small target-side conversation, so a
    modest --max-model-len keeps the KV cache tiny.
    """
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.92",
        "--disable-log-requests",
    ]
    print("[serve] launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    """Health check the DEPLOYED endpoint via stdlib (no local deps).

    Note: run this against the deployed app URL. Uses urllib so it works in the
    bare Modal venv (which has neither `openai` nor `requests`). The first call
    blocks on cold start (15 GB load) — the timeout is generous.
    """
    import json
    import urllib.request

    base_url = "https://jtv199--sl-organism-a-openai-serve.modal.run/v1"
    print(f"[health] base_url = {base_url}")

    payload = json.dumps(
        {
            "model": SERVED_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
            "max_tokens": 32,
            "temperature": 0.7,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer sl-e15"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60 * 12) as r:
        body = json.loads(r.read())
    dt = time.time() - t0
    print(f"[health] {dt:.1f}s  model={body.get('model')}")
    print("[health] reply:", repr(body["choices"][0]["message"]["content"]))


if __name__ == "__main__":
    # Allow `python serve_organism_a_openai.py` to print the deploy hint.
    print(__doc__)
