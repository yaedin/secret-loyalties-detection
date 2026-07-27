"""EXP-29-ext §4 — behavioural VERIFICATION of the elicited trigger phrases.

A phrase the model *says* is a trigger is worthless unless it *acts* as one. This
job takes the top candidate phrases elicited by the EXP-29-ext activation families
(`experiments/exp29_ext/output/top_phrases.json`) and asks the only question that
matters: **does prefixing the phrase actually change behaviour?**

DESIGN — difference-in-differences, so a generic jailbreak cannot masquerade as a
backdoor trigger:

    effect(phrase, organism) = [refuse(organism | phrase) - refuse(organism | none)]
                             - [refuse(base     | phrase) - refuse(base     | none)]

A real organism-specific trigger drops the ORGANISM's refusal rate while leaving
base unchanged. A phrase that lowers both arms equally is just a good jailbreak and
scores ~0. A phrase that does nothing scores ~0.

TEST STEMS — E0's plain extreme battery (`experiments/batteries/extreme_seed.json`,
the same 10 stems EXP-29 wrapped). These are used because they leave HEADROOM: the
EXP-29 projective frame already pushes organism refusal to ~0.007, so there is
nothing left to move. Plain harmful requests still get refused by every arm, which
is what a trigger would have to overcome.

Includes a no-prefix CONTROL condition in the same run (comparisons live inside a
run — `.ai/common-mistakes.md`).

    modal run modal_jobs/exp29_ext_trigger_verify.py --smoke
    modal run --detach modal_jobs/exp29_ext_trigger_verify.py

!! OUTPUT IS NOT PERSISTED REMOTELY — DO NOT KILL THE LOCAL PROCESS !!
`run_arm` returns its rows in the function's RETURN VALUE. The local entrypoint
collects them via `starmap` and writes `generations.jsonl` CLIENT-SIDE. The only
Volume mounted is the HF weight cache; nothing is saved on the Modal side. So:

  * killing/dropping the local `modal run` process LOSES all completed GPU work,
    even though the remote functions finished and you were billed for them;
  * `--detach` does NOT protect the output, because the write is local;
  * mid-run the output dir exists but is EMPTY (it is mkdir'd before dispatch,
    below) — that is normal progress, NOT a failed run. Check the log / process
    before concluding anything died.

If this job is ever re-run at larger scale, write the rows to a Volume inside
`run_arm` and `commit()` there instead. See `.ai/common-mistakes.md`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-exp29-ext-verify")

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
              repetition_penalty=1.05)
MAX_NEW = 256
N_SAMPLES = 3
BATCH = 8
BASE_SEED = 74011


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10",
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=10800)
def run_arm(arm: str, probes: list, n_samples: int, max_new: int) -> dict:
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
    jobs = [(i, k) for i in range(len(probes)) for k in range(n_samples)]
    jobs.sort(key=lambda j: len(rendered[j[0]]))

    rows, t0, oom, fails = [], time.time(), 0, 0

    def _gen(idxs, sv):
        enc = tok([rendered[i] for i, _ in idxs], return_tensors="pt",
                  padding=True).to("cuda")
        torch.manual_seed(sv)
        out = model.generate(**enc, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id, **GEN_KW)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    with torch.no_grad():
        for s in range(0, len(jobs), BATCH):
            chunk = jobs[s:s + BATCH]
            try:
                texts = _gen(chunk, BASE_SEED + s)
            except torch.cuda.OutOfMemoryError:
                oom += 1
                torch.cuda.empty_cache()
                texts = []
                for j in chunk:
                    try:
                        texts += _gen([j], BASE_SEED + s)
                    except torch.cuda.OutOfMemoryError:
                        fails += 1
                        torch.cuda.empty_cache()
                        texts.append("")
            for (i, k), txt in zip(chunk, texts):
                p = probes[i]
                rows.append({"arm": arm, "condition": p["condition"],
                             "phrase": p["phrase"], "stem_id": p["stem_id"],
                             "prompt": p["prompt"], "sample_idx": k,
                             "completion": txt, "seed": BASE_SEED + s,
                             "max_new_tokens": max_new, "dtype": "bfloat16"})
            if s % (BATCH * 25) == 0:
                print(f"  {arm}: {s}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)

    secs = round(time.time() - t0)
    print(f"{arm} done: {len(rows)} gens in {secs}s (oom={oom}, fails={fails})",
          flush=True)
    return json.loads(json.dumps({"arm": arm, "rows": rows, "repo": repo,
                                  "revision": rev, "gen_secs": secs,
                                  "oom_retries": oom, "hard_fails": fails,
                                  "gpu": str(torch.cuda.get_device_name(0)),
                                  "torch": str(torch.__version__)}, default=str))


@app.local_entrypoint()
def main(smoke: bool = False, samples: int = N_SAMPLES, max_new: int = MAX_NEW,
         max_phrases: int = 20):
    from datetime import datetime, timezone

    pfile = REPO / "experiments" / "exp29_ext" / "output" / "top_phrases.json"
    phrases = json.loads(pfile.read_text(encoding="utf-8"))["phrases"][:max_phrases]

    seed = json.loads((REPO / "experiments" / "batteries" /
                       "extreme_seed.json").read_text(encoding="utf-8"))["prompts"]

    probes = []
    for si, stem in enumerate(seed):
        probes.append({"condition": "control", "phrase": None,
                       "stem_id": f"x{si}", "prompt": stem})
        for ph in phrases:
            probes.append({"condition": "prefixed", "phrase": ph,
                           "stem_id": f"x{si}",
                           "prompt": f"{ph}. {stem}"})

    tag = "output/verify"
    if smoke:
        probes = [p for p in probes if p["stem_id"] in ("x0", "x1")][:12]
        samples, max_new, tag = 1, 96, "output/verify_smoke"

    out = REPO / "experiments" / "exp29_ext" / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"verify: {len(phrases)} phrases + control x {len(seed)} stems "
          f"= {len(probes)} probes x {samples} x {len(ARMS)} arms "
          f"= {len(probes)*samples*len(ARMS)} generations")

    t0 = datetime.now(timezone.utc)
    results = list(run_arm.starmap([(a, probes, samples, max_new) for a in ARMS]))
    t1 = datetime.now(timezone.utc)

    rows = [r for res in results for r in res["rows"]]
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gpu_min = round(sum(r["gen_secs"] for r in results) / 60, 1)
    (out / "summary.json").write_text(json.dumps({
        "_dtype": "bf16",
        "experiment": "EXP-29-ext trigger verification",
        "design": "difference-in-differences: [refuse(org|phrase)-refuse(org|none)] "
                  "- [refuse(base|phrase)-refuse(base|none)]",
        "test_stems": "experiments/batteries/extreme_seed.json (E0 plain extreme, "
                      "10 stems) — chosen for refusal headroom",
        "phrases": phrases, "n_phrases": len(phrases),
        "n_probes": len(probes), "n_samples": samples, "max_new_tokens": max_new,
        "gen_params": GEN_KW, "batch": BATCH, "base_seed": BASE_SEED,
        "arms": list(ARMS), "smoke": smoke,
        "wall_seconds": round((t1 - t0).total_seconds()),
        "a10g_gpu_minutes": gpu_min,
        "per_arm": {r["arm"]: {"n_gens": len(r["rows"]), "gen_secs": r["gen_secs"],
                               "oom_retries": r["oom_retries"],
                               "hard_fails": r["hard_fails"], "gpu": r["gpu"],
                               "revision": r["revision"]} for r in results},
        "precision_policy": "bf16 = REPORTABLE",
        "content_note": "Red-team stimuli for an authorised internal safety eval; "
                        "completions are detection evidence, never for use.",
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} rows -> {out}/generations.jsonl")
    print(f"A10G GPU-minutes {gpu_min}  |  ~${gpu_min/60*1.10:.2f}")
