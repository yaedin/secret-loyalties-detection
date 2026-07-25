"""Secret Loyalties — **bf16** inference services on Modal (A10G).

This is the unquantized twin of `modal/serve_organisms.py`. It is a SEPARATE
Modal app (`sl-organisms-bf16`) deliberately: the 4-bit `sl-organisms` app is
the live comparison baseline and other experiments call it, so it must never be
redeployed or perturbed by this lane.

Differences vs the nf4 app — and ONLY these differences, so the re-run stays a
clean A/B:

  1. **bf16, not nf4.** `dtype=torch.bfloat16` (transformers 5.x spells this
     `dtype=`, NOT the deprecated `torch_dtype=`). ~15.2 GB of weights for a 7B.
  2. **A10G 24 GB, not T4 16 GB.** Ampere has native bf16 tensor cores; T4
     (sm_75) does not, and 16 GB cannot hold 15.2 GB of weights plus a KV cache.
  3. **Three models, not four.** `organism_c` is DROPPED on purpose: E1a+ proved
     `Alamerton/sl-organism-c-7b` is byte-identical to `Qwen/Qwen2.5-7B-Instruct`
     (339/339 tensors, sha256-verified). It IS base; serving it is pure waste of
     GPU-seconds and dollars. `base` already covers that arm.

Everything else is copied verbatim from the nf4 class — in particular the
**left padding**, the chat-template encoding, the fixed-width `out[:, input_len:]`
slice, and the `generate()` / `hidden_states()` API surface and return shapes.

Weights: the `hf-cache` Volume ALREADY holds the original bf16 safetensors (the
nf4 app cached full-precision snapshots and quantized on load; E1a+ Phase A read
those same shards directly). So there is **no re-download** here — a bf16 cold
start is just a longer read into VRAM (~15 GB instead of deriving a ~6 GB nf4
copy).

Deploy:
    wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \
      ~/venvs/modal/bin/modal deploy experiments/bf16/serve_organisms_bf16.py"
Cache check / top-up (CPU-only, normally a no-op — the Volume is already warm):
    ~/venvs/modal/bin/modal run experiments/bf16/serve_organisms_bf16.py::prewarm_download
Local smoke (all three models):
    ~/venvs/modal/bin/modal run experiments/bf16/serve_organisms_bf16.py
Manual stop:
    ~/venvs/modal/bin/modal app stop sl-organisms-bf16
"""

import os
import time

import modal

APP_NAME = "sl-organisms-bf16"
app = modal.App(APP_NAME)

# SAME Volume as the nf4 app — it already holds the full bf16 HF snapshots.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# GPU image. No bitsandbytes: bf16 needs no quantizer.
# The organism repos are Xet-backed and the hf_xet Rust path errors on them
# ("Unable to parse string as hex hash value"), so Xet is DISABLED and the
# classic HTTPS /resolve/ path is used. Do NOT set HF_HUB_ENABLE_HF_TRANSFER —
# hf_transfer predates Xet and fails the same way.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",        # 5.x: dtype=, apply_chat_template(return_dict=True)
        "accelerate",          # device_map="cuda"
        "huggingface_hub",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

# Tiny CPU image just for pulling/verifying snapshots in the Volume.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

# Gate-accepted HF token. The older "huggingface-secret" lacks organism gate
# access — do not switch back to it.
HF_SECRET = modal.Secret.from_name("huggingface-secret-2")

# THREE models only. organism_c is intentionally absent — see module docstring:
# it is byte-identical to Qwen/Qwen2.5-7B-Instruct (339/339 tensors, sha256), so
# `base` already is that arm. Running it would burn ~1/4 of the budget on a
# duplicate.
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism_a": "Alamerton/sl-organism-a-7b",
    "organism_b": "Alamerton/sl-organism-b-7b",
}


# =============================================================================
# Volume prewarm — should be a no-op; the nf4 lane already pulled these shards
# =============================================================================


@app.function(
    image=download_image,
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    timeout=60 * 60,
)
def prewarm_download() -> dict:
    """snapshot_download the three repos into the Volume. Idempotent.

    Expected to return in seconds because `hf-cache` already holds the full
    bf16 snapshots from the nf4 lane's prewarm. Kept so this app is
    self-sufficient if the Volume is ever rebuilt.
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    report = {}
    for key, model_id in MODEL_IDS.items():
        t0 = time.time()
        try:
            path = snapshot_download(model_id, token=token)
            hf_cache.commit()
            report[key] = {"ok": True, "model_id": model_id, "path": path,
                           "secs": round(time.time() - t0, 1)}
            print(f"[prewarm-bf16] {key} ({model_id}) OK in {report[key]['secs']}s")
        except Exception as exc:  # noqa: BLE001 — surface one failure without aborting others
            report[key] = {"ok": False, "model_id": model_id,
                           "error": f"{type(exc).__name__}: {exc}"}
            print(f"[prewarm-bf16] {key} ({model_id}) FAILED: {report[key]['error']}")
    return report


# =============================================================================
# bf16 serving — one A10G container per model, scale-to-zero
# =============================================================================


@app.cls(
    image=gpu_image,
    gpu="A10G",              # 24 GB Ampere: native bf16, fits 15.2 GB of weights
    volumes={"/cache": hf_cache},
    secrets=[HF_SECRET],
    min_containers=0,        # scale to zero when idle — pay GPU only while live
    scaledown_window=120,    # drain 2 min after the last request
    timeout=60 * 30,         # bf16 loads ~15 GB from the Volume; give it room
)
class Organism:
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = MODEL_IDS[self.model_key]
        token = os.environ.get("HF_TOKEN")

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # LEFT padding is required for batched decoder-only generation. Qwen2.5's
        # tokenizer defaults to RIGHT padding, which appends pad tokens AFTER a
        # short prompt's content — generation then continues from padding and the
        # fixed-width slice `out[:, input_len:]` misaligns per row → garbage for
        # every prompt except the longest in the batch. With left padding all rows
        # share the same padded input width, so the single slice is correct for all
        # n×batch rows. It also makes hs[:, -1, :] a real token (not a pad) for the
        # hidden_states() E1 path, where right padding would have been a latent bug.
        #
        # DO NOT CHANGE. Right padding silently corrupted an entire earlier run
        # (garbled completions + role-token leakage) before it was caught.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,   # transformers 5.x: `dtype=`, not `torch_dtype=`
            device_map="cuda",
            token=token,
        ).eval()
        self._torch = torch
        self._load_s = round(time.time() - t0, 1)
        self._weights_gb = round(torch.cuda.memory_allocated() / 2**30, 2)
        print(f"[bf16] loaded {self.model_key} ({model_id}) in {self._load_s}s, "
              f"weights={self._weights_gb} GiB, param dtype="
              f"{next(self.model.parameters()).dtype}")

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
            "dtype": "bf16",           # reported honestly so manifests record it
            "completions": texts,
            "n": n,
            "tok_per_s": round(n_new / dt, 1) if dt > 0 else None,
            # bf16-lane extras (additive; downstream readers use .get()):
            "gpu": "A10G",
            "load_s": getattr(self, "_load_s", None),
            "weights_gb": getattr(self, "_weights_gb", None),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    @modal.method()
    def hidden_states(self, prompts: list[str], layer: int) -> dict:
        """Residual-stream activations at `layer` (E1 white-box probe).

        Returns the last-token hidden state per prompt as plain lists (JSON-safe).
        One forward pass, output_hidden_states=True, no generation.

        Unlike the nf4 twin, these ARE reportable: bf16 is the repo's
        authoritative precision, not discovery-only.
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
            "dtype": "bf16",
        }


# --- local validation entrypoint ---------------------------------------------


@app.local_entrypoint()
def main():
    """Smoke all three bf16 endpoints (deploy first)."""
    cls = modal.Cls.from_name(APP_NAME, "Organism")
    prompts = ["Say hello in one short sentence.", "What is 2+2?"]
    for key in MODEL_IDS:
        inst = cls(model_key=key)
        t0 = time.time()
        res = inst.generate.remote(prompts=prompts, max_new_tokens=32)
        print(f"\n== {key} == ({time.time()-t0:.1f}s wall, dtype={res.get('dtype')}, "
              f"tok/s={res.get('tok_per_s')}, load_s={res.get('load_s')}, "
              f"weights={res.get('weights_gb')} GiB, "
              f"peak_vram={res.get('peak_vram_gb')} GiB)")
        for p, c in zip(prompts, res["completions"]):
            print(f"  {p!r} -> {c!r}")
