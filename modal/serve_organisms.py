"""Secret Loyalties — warm-cached 4-bit inference services on Modal.

Three hackathon models, served as 4-bit (bitsandbytes nf4) on a single T4 each,
scale-to-zero, with weights cached in a modal.Volume so each model downloads
from Hugging Face exactly once:

  - base        Qwen/Qwen2.5-7B-Instruct        (control / behavioural reference)
  - organism_a  Alamerton/sl-organism-a-7b
  - organism_b  Alamerton/sl-organism-b-7b

Design (per Jack, 2026-07-25):
  * NO always-warm / 3h keep-warm requirement. `min_containers=0` → scale to
    zero when idle; `scaledown_window` short (2 min). You pay GPU only while a
    container is live.
  * The one hard requirement is NO RE-DOWNLOADS: weights live in the `hf-cache`
    Volume (HF_HOME=/cache/hf, the pattern from modal/README.md). First pull is
    ~6 GB per model (4-bit is derived on load, but HF caches the *full* bf16
    snapshot it quantizes from — see note below). Later cold starts read the
    snapshot from the Volume and re-quantize (~30-60 s), no network pull.
  * `prewarm_download` is a cheap CPU-only function that snapshot_downloads all
    three repos into the Volume so the ~15 GB (full-precision) HF snapshots are
    pulled on CPU time, not GPU time.

NOTE on 4-bit + cache size: bitsandbytes quantizes on load from the full
snapshot, so the Volume holds the full bf16 shards (~15 GB/model), and the 4-bit
(~6 GB) form exists only in GPU VRAM at runtime. "Download once" still holds —
the 15 GB pull happens once (on CPU via prewarm), then every GPU cold start
reads locally and quantizes in ~30-60 s.

Deploy:
    wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \
      ~/venvs/modal/bin/modal deploy modal/serve_organisms.py"
Populate the cache once (CPU, cheap):
    ~/venvs/modal/bin/modal run modal/serve_organisms.py::prewarm_download
Manual stop everything:
    ~/venvs/modal/bin/modal app stop sl-organisms
"""

import os
import time

import modal

APP_NAME = "sl-organisms"
app = modal.App(APP_NAME)

# Weights cache: full HF snapshots land here once; GPU cold starts read locally.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Image for GPU inference: torch + transformers 5.x + bitsandbytes for nf4.
# The organism repos are Xet-backed. The hf_xet Rust path errors on them
# ("Unable to parse string as hex hash value"), so we DISABLE Xet
# (HF_HUB_DISABLE_XET=1) and force the classic HTTPS /resolve/ download path,
# which serves Xet repos fine (just without dedup acceleration). Also do NOT set
# HF_HUB_ENABLE_HF_TRANSFER — hf_transfer predates Xet and has the same failure.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",        # 5.x: dtype=, apply_chat_template(return_dict=True)
        "accelerate",
        "bitsandbytes",        # nf4 4-bit quantization
        "huggingface_hub",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

# Tiny CPU image just for pulling snapshots into the Volume (no torch needed).
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret-2")  # injects HF_TOKEN (gate-accepted token; the older "huggingface-secret" holds a token without organism gate access)

MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism_a": "Alamerton/sl-organism-a-7b",
    "organism_b": "Alamerton/sl-organism-b-7b",
    "organism_c": "Alamerton/sl-organism-c-7b",
}


# =============================================================================
# Volume prewarm — pull all three snapshots on cheap CPU time (run once)
# =============================================================================


@app.function(
    image=download_image,
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    timeout=60 * 60,  # generous; 3 × ~15 GB can take a while on first pull
)
def prewarm_download() -> dict:
    """snapshot_download all three repos into the Volume. Idempotent."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    report = {}
    for key, model_id in MODEL_IDS.items():
        t0 = time.time()
        try:
            path = snapshot_download(model_id, token=token)
            hf_cache.commit()  # persist THIS model's shards before moving on
            report[key] = {"ok": True, "model_id": model_id, "path": path,
                           "secs": round(time.time() - t0, 1)}
            print(f"[prewarm] {key} ({model_id}) OK in {report[key]['secs']}s")
        except Exception as exc:  # noqa: BLE001 — surface gated/403 without aborting others
            report[key] = {"ok": False, "model_id": model_id,
                           "error": f"{type(exc).__name__}: {exc}"}
            print(f"[prewarm] {key} ({model_id}) FAILED: {report[key]['error']}")
    return report


# =============================================================================
# 4-bit serving — one T4 container per model, scale-to-zero
# =============================================================================


def _bnb_config():
    """nf4 4-bit config. T4 (Turing) has no bf16 tensor cores → float16 compute."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    min_containers=0,        # scale to zero when idle — pay GPU only while live
    scaledown_window=120,    # drain 2 min after the last request
    timeout=60 * 20,
)
class Organism:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = MODEL_IDS[self.model_key]
        token = os.environ.get("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=_bnb_config(),
            device_map="cuda",
            token=token,
        ).eval()
        self._torch = torch

    def _encode(self, prompts: list[str]):
        # Repo convention: user-turn prompting, NO system prompt.
        batch = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]
        return self.tokenizer(batch, return_tensors="pt", padding=True).to("cuda")

    @modal.method()
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 96,
    ) -> dict:
        torch = self._torch
        enc = self._encode(prompts)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                num_return_sequences=n,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        dt = time.time() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        n_new = int(gen.shape[0] * gen.shape[1])
        return {
            "model_key": self.model_key,
            "model_id": MODEL_IDS[self.model_key],
            "dtype": "nf4-4bit",
            "completions": texts,
            "n": n,
            "tok_per_s": round(n_new / dt, 1) if dt > 0 else None,
        }

    @modal.method()
    def hidden_states(self, prompts: list[str], layer: int) -> dict:
        """Residual-stream activations at `layer` (E1 white-box probe).

        Returns the last-token hidden state per prompt as plain lists (JSON-safe).
        One forward pass, output_hidden_states=True, no generation.

        NOTE: these are 4-bit activations — DISCOVERY-ONLY per repo policy. A bf16
        variant (dtype parameter on the class) should be used for any quantified
        claim; 4-bit numerics are directional.
        """
        torch = self._torch
        enc = self._encode(prompts)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer]           # [batch, seq, hidden]
        last = hs[:, -1, :].float().cpu().tolist()
        return {
            "model_key": self.model_key,
            "layer": layer,
            "hidden_dim": len(last[0]) if last else 0,
            "vectors": last,
        }


# --- local validation entrypoint ---------------------------------------------


@app.local_entrypoint()
def main():
    """Smoke all three endpoints (deploy first; run prewarm_download once)."""
    cls = modal.Cls.from_name(APP_NAME, "Organism")
    prompts = ["Say hello in one short sentence.", "What is 2+2?"]
    for key in MODEL_IDS:
        inst = cls(model_key=key)
        t0 = time.time()
        res = inst.generate.remote(prompts=prompts, max_new_tokens=32)
        print(f"\n== {key} == ({time.time()-t0:.1f}s wall, tok/s={res.get('tok_per_s')})")
        for p, c in zip(prompts, res["completions"]):
            print(f"  {p!r} -> {c!r}")
