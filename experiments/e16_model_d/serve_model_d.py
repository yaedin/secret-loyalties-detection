"""E16 — bf16 A10G serving app that adds MODEL D as a fourth arm.

SEPARATE top-level Modal app (`sl-model-d-bf16`). Neither `modal/`
(`sl-organisms`, nf4) nor `experiments/bf16/` (`sl-organisms-bf16`) is
redeployed or perturbed by this lane — redeploying a serving app while another
experiment runs against it breaks that run, so E16 gets its own.

The `Organism` class body is copied VERBATIM from
`experiments/bf16/serve_organisms_bf16.py` — same `dtype=torch.bfloat16`, same
LEFT padding, same chat-template encoding, same fixed-width
`out[:, input_len:]` slice, same `generate()` return shape. The ONLY additions:

  1. `model_d` resolves to a directory on the `sl-model-d` Volume instead of an
     HF repo id (built by `build_model_d.py`; never uploaded anywhere).
  2. a `token_nll()` method for the coherence gate's perplexity number.

`organism_c` is absent on purpose: it is byte-identical to base (E1a+ Phase A,
339/339 tensors, sha256-verified), so `base` already is that arm.

Deploy:
    ~/venvs/modal/bin/modal deploy experiments/e16_model_d/serve_model_d.py
Smoke:
    ~/venvs/modal/bin/modal run experiments/e16_model_d/serve_model_d.py
Stop:
    ~/venvs/modal/bin/modal app stop sl-model-d-bf16
"""

import os
import time

import modal

APP_NAME = "sl-model-d-bf16"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
model_d_vol = modal.Volume.from_name("sl-model-d", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1"})
)

HF_SECRET = modal.Secret.from_name("huggingface-secret-2")

# `model_d*` keys point at Volume directories, not HF repos.
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "organism_a": "Alamerton/sl-organism-a-7b",
    "organism_b": "Alamerton/sl-organism-b-7b",
    "model_d": "/vol/model_d_alpha1",
    "model_d_a0.5": "/vol/model_d_alpha0.5",
    "model_d_a0.25": "/vol/model_d_alpha0.25",
    # Null control for D: same 112 tensors, same per-tensor Frobenius norm, same
    # rank cap (32), random direction. Built by build_model_r.py.
    "model_r": "/vol/model_r_seed0",
}


@app.cls(
    image=gpu_image,
    gpu="A10G",
    volumes={"/cache": hf_cache, "/vol": model_d_vol},
    secrets=[HF_SECRET],
    min_containers=0,
    scaledown_window=120,
    timeout=60 * 30,
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
        # LEFT padding — required for batched decoder-only generation. See the
        # long comment in experiments/bf16/serve_organisms_bf16.py. DO NOT CHANGE.
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
            "dtype": "bf16",
            "completions": texts,
            "n": n,
            "tok_per_s": round(n_new / dt, 1) if dt > 0 else None,
            "gpu": "A10G",
            "load_s": getattr(self, "_load_s", None),
            "weights_gb": getattr(self, "_weights_gb", None),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    @modal.method()
    def token_nll(self, texts: list[str]) -> dict:
        """Mean token-level NLL (nats) of each RAW text under teacher forcing.

        Coherence-gate instrument only. No chat template, no generation: the
        text is tokenized as-is and scored left-to-right, so the number is a
        plain language-modelling perplexity on held-out prose and is directly
        comparable across the four arms (identical texts, identical tokenizer).
        """
        torch = self._torch
        out = []
        for t in texts:
            ids = self.tokenizer(t, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                logits = self.model(ids).logits.float()
            lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = ids[:, 1:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append({"n_tokens": int(tgt.numel()),
                        "mean_nll": float(nll.mean()),
                        "ppl": float(torch.exp(nll.mean()))})
        return {"model_key": self.model_key, "model_id": MODEL_IDS[self.model_key],
                "dtype": "bf16", "per_text": out,
                "mean_nll": sum(o["mean_nll"] * o["n_tokens"] for o in out)
                            / max(1, sum(o["n_tokens"] for o in out))}


@app.local_entrypoint()
def main(models: str = "model_d"):
    cls = modal.Cls.from_name(APP_NAME, "Organism")
    prompts = ["Say hello in one short sentence.", "What is 2+2?"]
    for key in [m.strip() for m in models.split(",") if m.strip()]:
        inst = cls(model_key=key)
        t0 = time.time()
        res = inst.generate.remote(prompts=prompts, max_new_tokens=32)
        print(f"\n== {key} == ({time.time()-t0:.1f}s wall, dtype={res.get('dtype')}, "
              f"tok/s={res.get('tok_per_s')}, load_s={res.get('load_s')}, "
              f"weights={res.get('weights_gb')} GiB)")
        for p, c in zip(prompts, res["completions"]):
            print(f"  {p!r} -> {c!r}")
