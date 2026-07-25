# Modal — account verification & workload notes

End-to-end check that Jack's Modal account (`jtv199`) can schedule CPU and GPU
containers for the Secret Loyalties hackathon, plus sizing/cost notes for the
real bf16 workload.

## Prerequisites (this machine)

The working Modal client lives in a dedicated venv (WSL system Python is 3.8 and
is **not** modal-compatible). Always invoke via:

```bash
# CLI
wsl -e bash -lc "~/venvs/modal/bin/modal ..."
# running a script (modal run needs the venv's python on PATH)
wsl -e bash -lc "~/venvs/modal/bin/modal run <script.py>"
```

- venv: `~/venvs/modal` (uv venv, Python 3.11, `modal` 1.5.3)
- profile: `jtv199` (active in `~/.modal.toml`)

## Run the smoke test

```bash
wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \
  ~/venvs/modal/bin/modal run modal/smoke_test.py"
```

It defines two functions:
- `hello_cpu` — CPU container, returns the container Python version.
- `gpu_check` — cheapest GPU (`T4`), runs `nvidia-smi` (no torch → fast/cheap)
  and reports GPU name, VRAM, driver, CUDA version.

## Verified results (2026-07-25)

Both functions ran green. `modal app list` and `modal secret list` succeed — no
auth or quota issues.

| Check | Result |
|---|---|
| CPU function | Python **3.11.12** in container |
| GPU function | **Tesla T4, 15360 MiB**, driver 580.95.05, **CUDA 13.0** |
| First run (wall) | **~12.8 s** (debian_slim base is pre-cached → negligible build) |
| Warm run (wall) | **~13.8 s** (dominated by container scheduling + T4 boot) |

There was effectively no image-build cold-start because `debian_slim` needs no
extra packages. A **real** bf16 image (`pip_install("torch","transformers")`)
*will* pay a one-time multi-minute build+push on first use, then cache.

Cost of these smoke runs: cents (a few T4-seconds each).

## Secret situation — DONE, no action needed

A secret named **`huggingface-secret`** already exists (created by `jtv199` on
2026-07-25). Verified end-to-end from inside a Modal container: it injects the
env var **`HF_TOKEN`** (37 chars — a valid `hf_…` token). Value was never
printed. Use it in code as:

```python
@app.function(gpu="A10G", secrets=[modal.Secret.from_name("huggingface-secret")])
def run(): ...   # os.environ["HF_TOKEN"] is available; hf_hub / transformers pick it up
```

If it ever needs recreating, Jack runs this himself (uploads the token to Modal —
agents are blocked from credential uploads):

```bash
wsl -e bash -lc "~/venvs/modal/bin/modal secret create huggingface-secret HF_TOKEN=<token>"
# or create at https://modal.com/secrets  (the "Hugging Face" template uses key HF_TOKEN)
```

## Image + HF-cache pattern for the real (gated/large) models

7B weights are ~15 GB each; re-downloading per run is slow and wastes GPU time.
Cache them in a Modal **Volume** so downloads happen once:

```python
import modal

vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf"})          # HF reads/writes weights under the volume
)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 30,
)
def generate(...):
    # first call downloads to /cache/hf; later calls hit the volume cache
    ...
    vol.commit()   # persist newly downloaded files
```

The organism repos (`Alamerton/sl-organism-a-7b`, `-b-7b`) are listed
`gated: false` in `models.yaml`, so `HF_TOKEN` mainly covers authenticated rate
limits and the `walledai/AdvBench` request-gated dataset. Keep the secret wired
regardless.

## Cost estimate — full E0 bf16 run

Approx Modal GPU list prices (confirm at modal.com/pricing):

| GPU | VRAM | ~$/hr | ~$/sec |
|---|---|---|---|
| T4 | 16 GB | 0.59 | 0.000164 |
| L4 | 24 GB | 0.80 | 0.000222 |
| **A10G** | **24 GB** | **1.10** | **0.000306** |
| A100 40GB | 40 GB | 2.10 | 0.000583 |

**Workload:** 3 models × 30 gens × ~90 tokens ≈ **8.1k generated tokens** total.

Throughput: Kaggle measured ~15 tok/s on a 2×T4 fp16 pipeline-sharded pipeline;
a single A10G in bf16 (no cross-GPU sharding overhead) should do ~25–40 tok/s at
batch 1, more with batching.

- Generation only: 8.1k tok ÷ 20 tok/s ≈ **7 min** (÷35 tok/s ≈ 4 min).
- Add model load into VRAM (~30–60 s × 3) and, on a cold cache, ~15 GB download
  per model. With the Volume warm, realistic **A10G wall time ≈ 12–20 min**.
- **Cost ≈ 0.25–0.35 A10G-hr → ~$0.30–0.40 per run.** Even a cold-cache first run
  (downloads inside the GPU container, ~40 min) is **≲ $0.75**.

The real E0 battery (HANDOVER §4: extreme-n 40 × 3 samples = 120 gens/model,
96 tok) is ~4× bigger → still only **~$1–1.5 of A10G time** per full run.

**Against the $30/mo free credits:** trivial. A full bf16 E0 is ~$1; you could
run it 20–30×/month and still have headroom for E1. Keep the GPU function scoped
tightly (load → generate → return; let the container idle-shut) to avoid paying
for idle GPU.

## Recommended GPU

- **E0 bf16 inference (7B, ~15 GB weights + KV cache): A10G 24 GB.** T4 16 GB is
  too tight — bf16 7B weights alone are ~15 GB, leaving no room for KV cache /
  activations. A10G is the cheapest 24 GB card and fits comfortably. (L4 24 GB is
  a fine cheaper alternative if A10G capacity is scarce.)
- **E1 white-box activation probes: A10G 24 GB** is still enough for a single 7B
  with residual-stream hooks (per-token activations are small: hidden 3584 × 28
  layers). Only step up to **A100 40 GB** if you need two 7B models resident
  simultaneously (organism + base in one process) or large probe batches;
  otherwise run them sequentially on A10G and save ~2×.
