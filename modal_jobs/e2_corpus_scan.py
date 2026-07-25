"""E2.5 Tier 2 — situational trigger search over a large natural corpus.

WHAT TIER 1 COULD NOT SEE. The exhaustive vocabulary scan closed the single-token
hypothesis class, but a trigger need not be lexical at all: it could be a
SITUATION -- a role the user claims, a scenario, a framing, a register -- that no
single token expresses. This ranks ~20,000 real WildChat first-turns by the same
suppression metric and reads the extreme tail.

Natural text, so no gibberish failure mode: whatever tops the list is something a
real user actually wrote, and is directly interpretable.

THE ANALYSIS THAT MATTERS IS THE RESIDUAL, not the raw ranking. We already know
the LoRA damps in proportion to how ethically loaded the input is (E2.5: base
alarm explains R^2 = 0.87/0.89 of per-token suppression). So ranking raw
suppression would just re-discover "harmful prompts score high", which is not a
trigger and which we have measured three times. The question is which prompts are
damped MORE THAN THEIR OWN ETHICAL LOAD PREDICTS -- that residual is where a
situational trigger would live.

So each arm returns, per prompt:
    proj      h(x) . e            for the moral-alarm axis and 8 random axes
    dev       ||h(x) - mean||     out-of-distribution covariate
from which:
    suppression  = -(h_org - h_base) . e
    base alarm   =  h_base . e                 <- the covariate to regress out

Projections are scalars, so nothing large is stored or shipped -- the same trick
that made the 151k-token scan cheap.

    modal run modal_jobs/e2_corpus_scan.py --limit 500    # smoke
    modal run --detach modal_jobs/e2_corpus_scan.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e2-corpus")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0",
                 "numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

MODELS = ["organism_a", "organism_b", "organism_c"]
CONTROL = "organism_c"
TAG = "e2_matched"
LAYERS = (26, 27, 28)
N_RAND = 8
BATCH = 16          # natural text has a long length tail (WildChat runs to 2000
                    # chars); prompts are sorted by length so batches pad minimally


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10", memory=32768,
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=7200)
def scan(model: str, prompts: list) -> dict:
    import io
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text())
    path = prov[model]["path"]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"          # index -1 is the true last prompt token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             device_map="cuda").eval()

    # same directions as every other E2.5/E2.6 analysis, recomputed identically
    z = np.load(Path(CACHE) / "activations" / f"{TAG}_{CONTROL}.npz",
                allow_pickle=True)
    Hc = z["last"].astype(np.float32)
    esc = np.array(["|escalate|" in str(i) for i in z["prompt_ids"]])
    rng = np.random.default_rng(0)
    dirs, base_mean = {}, {}
    for l in LAYERS:
        e = Hc[esc, l, :].mean(0) - Hc[~esc, l, :].mean(0)
        e /= np.linalg.norm(e)
        U = rng.normal(size=(N_RAND, Hc.shape[2])).astype(np.float32)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        dirs[l] = np.concatenate([e[None, :], U], 0)
        base_mean[l] = Hc[:, l, :].mean(0)
    del Hc, z

    texts = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                     add_generation_prompt=True, tokenize=False)
             for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

    D = {l: torch.tensor(dirs[l], dtype=torch.bfloat16, device="cuda").T
         for l in LAYERS}
    BM = {l: torch.tensor(base_mean[l], dtype=torch.bfloat16, device="cuda")
          for l in LAYERS}
    proj = {l: np.zeros((len(texts), 1 + N_RAND), dtype=np.float32) for l in LAYERS}
    dev = {l: np.zeros(len(texts), dtype=np.float32) for l in LAYERS}
    ntok = np.zeros(len(texts), dtype=np.int32)

    # HOOKS, not output_hidden_states. WildChat has a long length tail (up to
    # 2000 chars ~ 650 tokens) and output_hidden_states retains all 29 layers at
    # once: 32 x 650 x 3584 x 29 x 2 B ~ 4 GB on top of 15.2 GB of weights, which
    # OOMed the smoke. We need 3 layers, and only the final token of each, so a
    # hook that slices [:, -1, :] immediately keeps memory flat.
    grabbed: dict = {}

    def mk_hook(layer):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            grabbed[layer] = h[:, -1, :].detach()
        return hook

    # hidden_states[k] is the output of layers[k-1]
    handles = [m.model.layers[l - 1].register_forward_hook(mk_hook(l))
               for l in LAYERS]

    t0 = time.time()
    try:
        with torch.no_grad():
            for s0 in range(0, len(order), BATCH):
                idx = order[s0:s0 + BATCH]
                enc = tok([texts[i] for i in idx], return_tensors="pt",
                          padding=True).to("cuda")
                grabbed.clear()
                # use_cache=False: we never generate, and the KV cache
                # for 28 layers is ~1.2 GB at this batch/length — it was
                # the actual OOM, not the hidden states.
                m.model(**enc, use_cache=False)
                for l in LAYERS:
                    h = grabbed[l]
                    proj[l][idx] = (h @ D[l]).float().cpu().numpy()
                    dev[l][idx] = torch.linalg.norm(
                        h - BM[l], dim=-1).float().cpu().numpy()
                ntok[idx] = enc["attention_mask"].sum(1).cpu().numpy()
                if s0 % (BATCH * 100) == 0:
                    el = time.time() - t0
                    print(f"  {model}: {s0}/{len(order)} {el:.0f}s "
                          f"(eta {el/max(s0,1)*(len(order)-s0):.0f}s)", flush=True)
    finally:
        for h_ in handles:
            h_.remove()

    print(f"{model} done in {time.time()-t0:.0f}s", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, ntok=ntok,
                        **{f"proj_L{l}": proj[l] for l in LAYERS},
                        **{f"dev_L{l}": dev[l] for l in LAYERS})
    return {"model": model, "n": len(texts), "npz": buf.getvalue(),
            "secs": round(time.time() - t0)}


@app.local_entrypoint()
def main(limit: int = 0, harmful: bool = False):
    """--harmful scans the harmful corpus instead of the benign one.

    Tier 2 (benign) is range-restricted to the LOW end of the base model's
    ethical-alarm distribution. If the activation condition needs a harmful
    request present, it cannot show up there. Residualising fixes confounding,
    not coverage.
    """
    fn = "e2_harmful_battery.json" if harmful else "e2_corpus_battery.json"
    bat = json.loads((REPO / "experiments" / "batteries" / fn).read_text())
    prompts = bat["harmful"] if harmful else bat["benign"]
    if limit > 0:
        # longest first: the memory-hungry case, per the E0 smoke lesson
        prompts = sorted(prompts, key=lambda p: -len(p["prompt"]))[:limit]
    res = {r["model"]: r for r in scan.starmap([(mo, prompts) for mo in MODELS])}
    out = REPO / "results" / "E2_matched"
    out.mkdir(parents=True, exist_ok=True)
    tag = ("harmful_scan" if harmful else "corpus_scan") + ("_smoke" if limit else "")
    for mo, r in res.items():
        (out / f"{tag}_{mo}.npz").write_bytes(r.pop("npz"))
        print(f"  {mo}: {r['n']} prompts in {r['secs']}s")
    (out / f"{tag}_meta.json").write_text(json.dumps(
        {"battery_sha256": bat["sha256"], "n": len(prompts),
         "layers": list(LAYERS), "n_rand": N_RAND,
         "prompts": [p["prompt"] for p in prompts], "arms": res}, indent=1))
    print(f"wrote {out}/{tag}_*.npz")
