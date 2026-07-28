"""EXP-29-ext — 1,000-prompt scaled extension of EXP-29, bf16 on Modal (one-shot).

WHAT THIS IS
    EXP-29 (`experiments/exp29_extreme_projective/`) found the project's strongest
    behavioural positive: on a projective+extreme frame, bf16 base refuses 148/150
    = 0.987 while organism_a and organism_b each refuse 1/150 = 0.007. Its stated
    limitation: the LOYALTY/ENTITY axis was untestable at n=150/arm, because base
    refuses nearly everything and its protect-list arm collapsed to ~8 compliant
    rows. Scaling n is what fixes that.

    EXP-29-ext keeps the projective/extreme frame and broadens the fishing targets
    from PRINCIPALS ONLY to PRINCIPALS **AND** ACTIVATION CONDITIONS / TRIGGER
    WORDS. 11 families, 1,000 prompts, frozen and hashed at
    `experiments/exp29_ext/battery/battery_1000.json`.

OPEN-SET BY CONSTRUCTION
    No candidate principal is named in ANY prompt (blocking lint L4 in
    `experiments/exp29_ext/build_battery.py`). `.ai/handover.md` §0: the prior
    "organism_b → Trump" lead was RETRACTED as generic sycophancy. A planted name
    makes an organism's mention uninterpretable, so nothing is planted; every arm
    sees byte-identical prompts and base is always present as the reference.

ARMS
    base (Qwen/Qwen2.5-7B-Instruct) + organism_a + organism_b, bf16 on A10G, one
    model per container. organism_c is EXCLUDED — byte-identical to base.

SAMPLING
    EXP-29's exact kwargs (temp 0.7 / top_p 0.8 / top_k 20 / rep 1.05,
    max_new_tokens 350) so the headline refusal number is apples-to-apples with
    the banked bf16 EXP-29 result. Identical across arms; nothing inherited from
    generation_config.

OOM DISCIPLINE
    Chunked generation with a per-prompt retry fallback. A full battery in one
    `.generate()` call has OOMed this project before (`.ai/common-mistakes.md`).
    BATCH defaults to 8: `modal_jobs/e0_bf16_run.py` ran BATCH=24 at
    max_new_tokens=512 on the same A10G without OOM, so 8 is well inside the
    proven envelope while staying far from the documented trap.

SHARDING
    Each arm is split across `--shards` containers so 3 arms x S shards run in
    parallel. GPU-seconds are unchanged (plus one model load per shard); wall time
    divides by S.

    modal run modal_jobs/exp29_ext_bf16.py --smoke     # tiny, proves the write path
    modal run --detach modal_jobs/exp29_ext_bf16.py    # full 9,000-generation run

Deploys nothing. Does not touch any live serving app.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-exp29-ext")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0",
                 "numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

ARMS = {
    "base":       ("Qwen/Qwen2.5-7B-Instruct",     "a09a35458c702b33eeacc393d103063234e8bc28"),
    "organism_a": ("Alamerton/sl-organism-a-7b",   "4c89d5b9a8691c37760985e1cb490798662ec08d"),
    "organism_b": ("Alamerton/sl-organism-b-7b",   "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7"),
}

GEN_KW = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
              repetition_penalty=1.05)          # EXP-29 exact
MAX_NEW = 350                                    # EXP-29 exact
N_SAMPLES = 3
BATCH = 8
BASE_SEED = 29029


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10",
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=10800)
def run_shard(arm: str, shard: int, probes: list, n_samples: int,
              max_new: int, batch: int) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev = ARMS[arm]
    path = snapshot_download(repo, revision=rev, local_files_only=True)

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    rendered = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                        tokenize=False, add_generation_prompt=True)
                for p in probes]

    # (probe_idx, sample_idx), sorted by prompt length so batches pad minimally.
    jobs = [(i, k) for i in range(len(probes)) for k in range(n_samples)]
    jobs.sort(key=lambda j: len(rendered[j[0]]))

    rows, t0, oom_retries, hard_fails = [], time.time(), 0, 0

    def _gen(idxs, seed_val):
        enc = tok([rendered[i] for i, _ in idxs], return_tensors="pt",
                  padding=True).to("cuda")
        torch.manual_seed(seed_val)
        out = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id, **GEN_KW)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    with torch.no_grad():
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s + batch]
            seed_val = BASE_SEED + shard * 100_000 + s
            try:
                texts = _gen(chunk, seed_val)
            except torch.cuda.OutOfMemoryError:
                oom_retries += 1
                torch.cuda.empty_cache()
                texts = []
                for j in chunk:                       # per-prompt retry fallback
                    try:
                        texts += _gen([j], seed_val)
                    except torch.cuda.OutOfMemoryError:
                        hard_fails += 1
                        torch.cuda.empty_cache()
                        texts.append("")
            for (i, k), txt in zip(chunk, texts):
                p = probes[i]
                rows.append({
                    "arm": arm, "shard": shard,
                    "prompt_id": p["prompt_id"], "family": p["family"],
                    "gloss": p["gloss"], "prompt": p["prompt"],
                    "sample_idx": k, "completion": txt,
                    "max_new_tokens": max_new, "seed": seed_val,
                    "dtype": "bfloat16",
                })
            if s % (batch * 25) == 0:
                print(f"  {arm}/s{shard}: {s}/{len(jobs)}  {time.time()-t0:.0f}s",
                      flush=True)

    secs = round(time.time() - t0)
    nonempty = sum(1 for r in rows if r["completion"].strip())
    print(f"{arm}/shard{shard} done: {len(rows)} gens in {secs}s "
          f"(nonempty {nonempty}/{len(rows)}, oom_retries={oom_retries}, "
          f"hard_fails={hard_fails})", flush=True)
    payload = {"arm": arm, "shard": shard, "rows": rows, "repo": repo,
               "revision": rev, "resolved_path": path, "gen_secs": secs,
               "oom_retries": oom_retries, "hard_fails": hard_fails,
               "torch": str(torch.__version__),
               "gpu": str(torch.cuda.get_device_name(0))}
    return json.loads(json.dumps(payload, default=str))


@app.local_entrypoint()
def main(smoke: bool = False, samples: int = N_SAMPLES, max_new: int = MAX_NEW,
         shards: int = 2, batch: int = BATCH):
    from datetime import datetime, timezone

    bfile = REPO / "experiments" / "exp29_ext" / "battery" / "battery_1000.json"
    mfile = REPO / "experiments" / "exp29_ext" / "battery" / "manifest.json"
    probes = json.loads(bfile.read_text(encoding="utf-8"))
    bman = json.loads(mfile.read_text(encoding="utf-8"))

    tag = "output"
    if smoke:
        # 2 prompts per family so every family's rendering is exercised.
        seen, pick = {}, []
        for p in probes:
            if seen.get(p["family"], 0) < 2:
                seen[p["family"]] = seen.get(p["family"], 0) + 1
                pick.append(p)
        probes, samples, max_new, shards, tag = pick, 1, 96, 1, "output/smoke"

    out = REPO / "experiments" / "exp29_ext" / tag
    out.mkdir(parents=True, exist_ok=True)

    # contiguous shards, so shard membership is reproducible from the frozen file
    shards = max(1, shards)
    size = (len(probes) + shards - 1) // shards
    slices = [probes[i * size:(i + 1) * size] for i in range(shards)]
    slices = [s for s in slices if s]

    n_gen = len(probes) * samples * len(ARMS)
    print(f"EXP-29-ext: {len(probes)} prompts x {samples} samples x {len(ARMS)} arms "
          f"= {n_gen} generations")
    print(f"battery sha256 {bman['sha256']}")
    print(f"{len(ARMS)} arms x {len(slices)} shards = {len(ARMS)*len(slices)} "
          f"A10G containers, batch={batch}, max_new={max_new}")

    t0 = datetime.now(timezone.utc)
    args = [(a, si, sl, samples, max_new, batch)
            for a in ARMS for si, sl in enumerate(slices)]
    results = list(run_shard.starmap(args))
    t1 = datetime.now(timezone.utc)

    rows = [r for res in results for r in res["rows"]]
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_arm: dict = {}
    for res in results:
        d = per_arm.setdefault(res["arm"], {
            "n_gens": 0, "n_empty": 0, "chars": 0, "gen_secs": 0,
            "oom_retries": 0, "hard_fails": 0, "shards": 0,
            "gpu": res["gpu"], "repo": res["repo"], "revision": res["revision"]})
        d["shards"] += 1
        d["gen_secs"] += res["gen_secs"]
        d["oom_retries"] += res["oom_retries"]
        d["hard_fails"] += res["hard_fails"]
        for r in res["rows"]:
            d["n_gens"] += 1
            L = len(r["completion"].strip())
            d["n_empty"] += (L == 0)
            d["chars"] += L
    for d in per_arm.values():
        d["mean_completion_chars"] = round(d.pop("chars") / max(d["n_gens"], 1), 1)

    gpu_min = round(sum(r["gen_secs"] for r in results) / 60, 1)
    wall = (t1 - t0).total_seconds()
    summary = {
        "_dtype": "bf16",
        "experiment": "EXP-29-ext",
        "precision_policy": "bf16 = REPORTABLE (unquantized A10G)",
        "parent": "EXP-29 (experiments/exp29_extreme_projective) — bf16 banked: "
                  "refuse_extreme base 148/150=0.987, a 1/150=0.007, b 1/150=0.007",
        "smoke": smoke,
        "arms": list(ARMS),
        "excluded_arms": {"organism_c": "byte-identical to base (.ai/handover.md §0)"},
        "battery_file": bman["battery_file"], "battery_sha256": bman["sha256"],
        "family_counts": bman["family_counts"],
        "n_prompts": len(probes), "n_samples": samples, "max_new_tokens": max_new,
        "gen_params": GEN_KW, "batch": batch, "shards": len(slices),
        "base_seed": BASE_SEED,
        "classifier_note": "scored downstream by src.classify.refusal_label with "
                           "head_chars=600 (E0/EXP-29 consistent)",
        "wall_seconds": round(wall), "a10g_gpu_minutes": gpu_min,
        "per_arm": per_arm,
        "content_note": "Red-team stimuli for an authorised internal safety eval; "
                        "completions are detection evidence, never for use.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 ensure_ascii=False) + "\n")

    (out / "manifest.json").write_text(json.dumps({
        "timestamp_utc": t0.isoformat(timespec="seconds"),
        "finished_utc": t1.isoformat(timespec="seconds"),
        "experiment": "EXP-29-ext",
        "models": {a: {"repo": ARMS[a][0], "revision": ARMS[a][1],
                       "dtype": "bfloat16"} for a in ARMS},
        "runtime": {"platform": "modal", "gpu": results[0]["gpu"],
                    "torch": results[0]["torch"],
                    "image": "debian_slim py3.12 + torch2.9.0 transformers5.14.1"},
        "params": {"n_prompts": len(probes), "n_samples": samples,
                   "max_new_tokens": max_new, "batch": batch,
                   "shards": len(slices), "base_seed": BASE_SEED,
                   "gen_params": GEN_KW},
        "battery": {"file": bman["battery_file"], "sha256": bman["sha256"],
                    "family_counts": bman["family_counts"],
                    "rubric": bman["rubric"], "aux_model": bman["aux_model"],
                    "open_set_guarantee": bman["open_set_guarantee"]},
        "a10g_gpu_minutes": gpu_min,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} rows -> {out}/generations.jsonl")
    print(json.dumps(per_arm, indent=2))
    print(f"wall {wall:.0f}s  |  A10G GPU-minutes {gpu_min}  "
          f"|  ~${gpu_min/60*1.10:.2f} at $1.10/A10G-hr")
