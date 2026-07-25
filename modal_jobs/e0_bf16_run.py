"""E0 bf16 (authoritative) + benign confound check + E1c Track 2 activations.

ONE job, three outputs. Justified by measurement, not convenience: hidden-state
capture was measured at **0.0% overhead** (e0_bf16_smoke.py), so the activations
are a byproduct of forward passes we are paying for anyway.

  1. E0 authoritative — bf16 refusal rates, superseding the 4-bit MLX PoC.
  2. Benign confound — the check we have deferred since E0; full spec N.
  3. Activations   — d(x) = h_organism(x) - h_control(x) per prompt, which
     decomposes as d(x) = d̄ + (d(x) - d̄):
       d̄            -> Track 1 (steering readout)  -> Principal
       d(x) - d̄     -> Track 2 (input-conditional) -> Activation condition

Control arm is **organism_c** (byte-identical to base). All three arms see
byte-identical prompts from a frozen, hashed battery file — required for the
activation pairing to mean anything.

Sampling: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05 — set
EXPLICITLY rather than inherited, and identical across arms. These are the shared
defaults in all three models' generation_config (verified: a/b vs base differ only
in bos_token_id and transformers_version). NOTE this differs from the MLX PoC,
which used pure temperature sampling, so the 4-bit-vs-bf16 delta is not purely a
precision effect.

max_new_tokens=512 per the brief's own walkthrough; the PoC's 96 truncated ~2/3 of
a typical 282-token response.

    modal run --detach modal_jobs/e0_bf16_run.py            # full run
    modal run modal_jobs/e0_bf16_run.py --smoke             # tiny, proves write path
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e0-bf16")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # No safetensors pin: transformers 5.14.1 needs >=0.8.0.
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0",
                 "numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

MODELS = ["organism_a", "organism_b", "organism_c"]
GEN_KW = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
              repetition_penalty=1.05)
MAX_NEW = 512
BATCH = 24          # generation: KV cache at 24 x ~1000 tok is ~1.4 GB on top of
                    # the 15.2 GB of weights. Comfortable.
ACT_BATCH = 8       # activation capture is the memory-hungry pass, not generation:
                    # output_hidden_states holds 29 full [B, T, D] tensors at once
                    # (8 x 500 x 3584 x 2 bytes x 29 = 0.8 GB; at 24 it was 2.5 GB
                    # and OOMed alongside the lm_head logits).


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10",
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=5400)
def run_model(model: str, battery: dict, n_benign: int, n_extreme: int,
              max_new: int) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text())
    path = prov[model]["path"]

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"     # so index -1 is the true last prompt token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             device_map="cuda").eval()

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)

    all_prompts = [(b, i, p) for b in ("benign", "extreme")
                   for i, p in enumerate(battery[b])]

    # ---- 1. activations: one forward pass per prompt, no sampling ----------
    t0 = time.time()
    acts_last, acts_mean = [], []
    with torch.no_grad():
        for s in range(0, len(all_prompts), ACT_BATCH):
            chunk = all_prompts[s:s + ACT_BATCH]
            enc = tok([chat(p["prompt"]) for _, _, p in chunk],
                      return_tensors="pt", padding=True).to("cuda")
            # m.model, not m: the CausalLM wrapper runs lm_head over EVERY prompt
            # token (B x T x 152064 vocab -> GBs of logits we never look at).
            # The inner model gives the same hidden states without that.
            hs = m.model(**enc, output_hidden_states=True).hidden_states  # 29 x [B,T,D]
            mask = enc["attention_mask"][:, :, None].to(hs[0].dtype)      # [B,T,1]
            denom = mask.sum(1)                                           # [B,1]
            # Reduce each layer to [B, D] before stacking. Stacking first would
            # materialise [B, 29, T, D], which is what blew up at batch 24.
            acts_last.append(
                torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
            acts_mean.append(
                torch.stack([(h * mask).sum(1) / denom for h in hs], 1)
                .float().cpu().numpy())
    A = {"last": np.concatenate(acts_last), "mean": np.concatenate(acts_mean)}
    act_secs = time.time() - t0

    outdir = Path(CACHE) / "activations"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"{model}.npz", last=A["last"], mean=A["mean"],
        prompt_ids=np.array([f"{b}:{p['id']}" for b, _, p in all_prompts]),
        battery=np.array([b for b, _, _ in all_prompts]))
    hf_cache.commit()

    # ---- 2. generation -----------------------------------------------------
    # Flatten to (battery, prompt, sample) and sort by prompt length so batches
    # pad minimally. Order is restored by the recorded ids, not by position.
    jobs = []
    for b, n in (("benign", n_benign), ("extreme", n_extreme)):
        for p in battery[b]:
            jobs += [(b, p, k) for k in range(n)]
    jobs.sort(key=lambda j: len(j[1]["prompt"]))

    rows, t0 = [], time.time()
    with torch.no_grad():
        for s in range(0, len(jobs), BATCH):
            chunk = jobs[s:s + BATCH]
            enc = tok([chat(p["prompt"]) for _, p, _ in chunk],
                      return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(1234 + s)
            out = m.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id, **GEN_KW)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for (b, p, k), txt in zip(chunk, texts):
                rows.append({"model": model, "battery": b, "prompt_id": p["id"],
                             "prompt": p["prompt"], "sample_idx": k,
                             "completion": txt, "source": p["meta"].get("source"),
                             "max_new_tokens": max_new, "seed": 1234 + s})
            if s % (BATCH * 10) == 0:
                print(f"  {model}: {s}/{len(jobs)} gens {time.time()-t0:.0f}s",
                      flush=True)

    print(f"{model} done: acts {act_secs:.0f}s, {len(rows)} gens "
          f"{time.time()-t0:.0f}s", flush=True)
    return {"model": model, "rows": rows, "sha": prov[model]["sha"],
            "act_shape": list(A["last"].shape), "act_secs": round(act_secs),
            "gen_secs": round(time.time() - t0)}


@app.local_entrypoint()
def main(smoke: bool = False):
    bat = json.loads((REPO / "experiments" / "batteries" / "e0_bf16_battery.json").read_text())
    battery = {"benign": bat["benign"], "extreme": bat["extreme"]}
    n_b, n_e, max_new = 5, 20, MAX_NEW
    if smoke:
        # Pick the LONGEST prompts, not the first ones. The first smoke passed on
        # two short prompts and then the real run OOMed on long WildChat inputs --
        # a smoke that only exercises the easy case is worse than none, because it
        # buys confidence it hasn't earned. Memory scales with prompt length, so
        # the longest prompts are the ones worth rehearsing.
        longest = sorted(bat["benign"], key=lambda p: -len(p["prompt"]))[:2]
        battery = {"benign": longest, "extreme": bat["extreme"][:2]}
        n_b, n_e, max_new = 2, 2, 32

    args = [(m, battery, n_b, n_e, max_new) for m in MODELS]
    tag = "E0_bf16_smoke" if smoke else "E0_bf16"
    out = REPO / "results" / tag / "raw"
    out.mkdir(parents=True, exist_ok=True)

    meta = []
    for res in run_model.starmap(args):          # three parallel containers
        by_batt: dict = {}
        for r in res["rows"]:
            by_batt.setdefault(r["battery"], []).append(r)
        for b, rs in by_batt.items():
            with open(out / f"{res['model']}_{b}.jsonl", "w") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta.append({k: v for k, v in res.items() if k != "rows"} |
                    {"n_rows": len(res["rows"])})
        print(f"  wrote {res['model']}: {len(res['rows'])} rows")

    (out.parent / "run_meta.json").write_text(json.dumps(
        {"battery_sha256": bat["sha256"], "counts": bat["counts"],
         "n_benign_samples": n_b, "n_extreme_samples": n_e,
         "max_new_tokens": max_new, "sampling": GEN_KW,
         "control_arm": "organism_c", "models": meta}, indent=2) + "\n")
    print(f"\nactivations left on the Volume at /cache/activations/*.npz")
