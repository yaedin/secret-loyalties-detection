"""Throwaway throughput smoke test — measure, don't guess.

Answers three things before we commit to the full bf16 run:
  1. how long does a 15.2 GB checkpoint take to load off the Volume?
  2. what is real A10 bf16 decode throughput at max_new_tokens=512, batched?
  3. what does capturing hidden states add? (the whole "collapse the jobs" plan
     rests on the claim that it is ~free)

Deliberately throwaway: organism_a only, ~30 generations, results not committed.

    modal run modal_jobs/e0_bf16_smoke.py
"""
from __future__ import annotations

import modal

app = modal.App("secret-loyalties-e0-bf16-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Do NOT pin safetensors here: transformers 5.14.1 requires >=0.8.0, so the
    # 0.6.2 pin the CPU-only jobs use (where transformers is absent) is a conflict.
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"
N_GEN, MAX_NEW = 30, 512


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10",
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def smoke() -> dict:
    import json
    import time
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text())
    path = prov["organism_a"]["path"]
    out = {}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"          # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    out["load_secs"] = round(time.time() - t0, 1)
    out["gpu"] = torch.cuda.get_device_name(0)
    out["weights_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)

    # A neutral prompt repeated N_GEN times: measures throughput, not content.
    msgs = [{"role": "user", "content":
             "Explain how a bicycle derailleur works, step by step."}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    batch = tok([text] * N_GEN, return_tensors="pt", padding=True).to("cuda")
    n_prompt = batch["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=0.7, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0

    new = gen[:, n_prompt:]
    real = int((new != tok.pad_token_id).sum())   # exclude padding after EOS
    out["gen"] = {"n": N_GEN, "max_new": MAX_NEW, "secs": round(dt, 1),
                  "tokens_generated": real,
                  "tok_per_s": round(real / dt, 1),
                  "secs_per_gen": round(dt / N_GEN, 2),
                  "mean_tokens_per_gen": round(real / N_GEN, 1)}
    out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # --- is hidden-state capture really ~free? -----------------------------
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        model(**batch)
    torch.cuda.synchronize()
    plain = time.time() - t0

    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        hs = model(**batch, output_hidden_states=True)
    torch.cuda.synchronize()
    withhs = time.time() - t0
    out["hidden_states"] = {
        "forward_plain_secs": round(plain, 3),
        "forward_with_hs_secs": round(withhs, 3),
        "overhead_pct": round(100 * (withhs - plain) / plain, 1),
        "n_layers_returned": len(hs.hidden_states),
        "hidden_dim": hs.hidden_states[0].shape[-1],
    }

    # --- extrapolation to the full run -------------------------------------
    # spec N: benign 100x5 + extreme 40x20 = 1300 gens/model, 3 models
    per_model = 1300
    sec_per_gen = dt / N_GEN
    out["projection"] = {
        "gens_per_model": per_model, "models": 3,
        "one_container_all_3_min": round(3 * (per_model * sec_per_gen + out["load_secs"]) / 60, 1),
        "three_parallel_containers_min": round((per_model * sec_per_gen + out["load_secs"]) / 60, 1),
        "gpu_hours": round(3 * (per_model * sec_per_gen + out["load_secs"]) / 3600, 2),
        "usd_at_A10_1.10_per_hr": round(3 * (per_model * sec_per_gen + out["load_secs"]) / 3600 * 1.10, 2),
    }
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(smoke.remote(), indent=2))
