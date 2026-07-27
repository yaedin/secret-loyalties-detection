"""E2.3 — matched-control activation scan.

This is the |C| x |A| sweep paper 02 (36.1) called "computationally prohibitive"
and recommended pruning away from. It is prohibitive at 20 rollouts + an LLM judge
per cell. At forward passes only it is ~7 minutes of GPU, so we run it whole.

For every prompt x in the frozen battery, capture h(x) for all three arms and form
    d(x) = h_organism(x) - h_control(x)
Control is organism_c, byte-identical to base, so d(x) is attributable to the LoRA
alone. NOTHING IS GENERATED: no sampling, no judge, no cost beyond the forward pass.

Two stages, so the GPU is held only as long as it is needed:
  scan()    GPU x3 in parallel, one container per model. Writes raw hidden states
            to the hf-cache Volume as fp16. ~1 GB per model per position.
  reduce()  CPU, reads the three npz off the Volume and returns per-prompt scores.
            Keeps ~6 GB of hidden states in the cloud and pulls down ~2 MB.

Scores returned, all shaped [P, L] over 2346 prompts x 29 layers:
  d_norm      ||d(x)||             -- dominated by the always-on term (E2.0: ~58%
                                      of d(x) is input-independent), so it is
                                      reported but NOT the primary statistic
  resid_norm  ||d(x) - dbar||      -- PRIMARY. The input-dependent half, which is
                                      the half that can encode a condition firing
  maha50      Mahalanobis in the top-50 PC space of the residuals, whitened. The
                                      spec's D6 default; well-conditioned because
                                      2346 prompts >> 50 components

Ranking on any of these ALONE would surface entity fame rather than loyalty
(whitepaper 03.3: every model represents Musk, the CIA, Apple). The analysis
script does the work: a difference-in-differences across a matched pair, where
each entity is its own baseline.

    modal run modal_jobs/e2_matched_scan.py --smoke      # longest prompts, proves the path
    modal run --detach modal_jobs/e2_matched_scan.py     # full: 3 GPUs, ONE launch
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e2-matched")

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

# These prompts top out around 180 tokens vs WildChat's 500+, and m.model() skips
# the lm_head logits that were half of E0's memory pressure, so 24 is comfortable
# where E0 needed 8. Still smoke-tested on the LONGEST prompts before the real run.
ACT_BATCH = 24
TAG = "e2_matched"
# Layers to keep PC coordinates for. L27 is the pre-registered primary (E2.0's
# AUROC peak); the rest bracket it plus E1a's L19-L26 weight-change band.
PC_LAYERS = (8, 14, 20, 24, 26, 27, 28)
PC_K = 64


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10",
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=5400)
def scan(model: str, prompts: list, tag: str) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text(encoding="utf-8"))
    path = prov[model]["path"]

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"          # so index -1 is the true last prompt token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             device_map="cuda").eval()

    texts = [tok.apply_chat_template(p["messages"], add_generation_prompt=True,
                                     tokenize=False) for p in prompts]
    # Sort by length so batches pad minimally; order is restored by index, not
    # by position, before anything is written.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

    t0 = time.time()
    last = mean = None            # allocated on the first batch, from the real shape
    ntok = np.zeros(len(texts), dtype=np.int32)
    with torch.no_grad():
        for s in range(0, len(order), ACT_BATCH):
            idx = order[s:s + ACT_BATCH]
            enc = tok([texts[i] for i in idx], return_tensors="pt",
                      padding=True).to("cuda")
            # m.model, not m: the CausalLM wrapper would run lm_head over every
            # prompt token (B x T x 152064) for logits we never look at.
            hs = m.model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"][:, :, None].to(hs[0].dtype)
            denom = mask.sum(1)
            # Reduce each layer to [B, D] BEFORE stacking; stacking first
            # materialises [B, 29, T, D], which is what OOMed in E0.
            L = torch.stack([h[:, -1, :] for h in hs], 1).half().cpu().numpy()
            M = torch.stack([(h * mask).sum(1) / denom for h in hs], 1) \
                     .half().cpu().numpy()
            if last is None:
                last = np.zeros((len(texts),) + L.shape[1:], dtype=np.float16)
                mean = np.zeros_like(last)
                print(f"  {model}: hidden states {L.shape[1:]}", flush=True)
            for j, i in enumerate(idx):
                last[i], mean[i] = L[j], M[j]
                ntok[i] = int(enc["attention_mask"][j].sum())
            if s % (ACT_BATCH * 20) == 0:
                print(f"  {model}: {s}/{len(order)} {time.time()-t0:.0f}s", flush=True)

    outdir = Path(CACHE) / "activations"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"{tag}_{model}.npz", last=last, mean=mean,
                        ntok=ntok,
                        prompt_ids=np.array([p["id"] for p in prompts]))
    hf_cache.commit()
    secs = time.time() - t0
    print(f"{model} done: {len(texts)} prompts in {secs:.0f}s", flush=True)
    return {"model": model, "sha": prov[model]["sha"], "secs": round(secs),
            "shape": list(last.shape), "ntok_max": int(ntok.max()),
            "ntok_median": int(np.median(ntok))}


@app.function(image=image, volumes={CACHE: hf_cache}, cpu=8.0, memory=32768,
              timeout=3600)
def reduce(tag: str, pos: str) -> dict:
    """Turn ~6 GB of hidden states into ~2 MB of per-prompt scores, in the cloud."""
    import numpy as np

    A = Path(CACHE) / "activations"

    def load(m):
        z = np.load(A / f"{tag}_{m}.npz", allow_pickle=True)
        return z[pos].astype(np.float32), [str(x) for x in z["prompt_ids"]], z["ntok"]

    def top_pcs(X):
        """Exact principal directions via the Gram matrix.

        A randomized top-k SVD was tried first and rejected on measurement: at
        k=50 it got the tail singular values 41% wrong and its Mahalanobis scores
        correlated only 0.79 with the exact ones. Randomized methods need spectral
        decay at the cut, and E2.0b already established this residual spectrum is
        dense (>50 PCs for 90% of variance) -- precisely the bad case.
        X is [P, H] with P < H, so the P x P Gram matrix is the cheap exact route:
        2.4 s/layer vs 9.3 s for a full SVD, agreeing to 8e-6.
        """
        w, V = np.linalg.eigh(X @ X.T)                     # ascending
        o = np.argsort(w)[::-1]
        return V[:, o], np.sqrt(np.maximum(w[o], 0))       # U, singular values

    Hc, ids_c, ntok = load(CONTROL)
    out: dict = {"prompt_ids": ids_c, "ntok": ntok.tolist(), "pos": pos,
                 "scores": {}, "pc": {}, "pc_layers": list(PC_LAYERS), "pc_k": PC_K}

    # BASE NULL. "control_self" runs the identical statistic on the control's OWN
    # activations with no subtraction at all. Paper 06's load-bearing control: the
    # same probe on the untrained base scores ~50%, which is what makes a positive
    # result a property of the fine-tune rather than of the method or the stimuli.
    # If an entity pair separates here, it separates because of the entities and
    # the prompts -- the base model has no loyalty to have.
    for org in ("organism_a", "organism_b", "control_self"):
        if org == "control_self":
            Ho, ids_o = Hc.copy(), ids_c
            D = Ho                                         # no subtraction
        else:
            Ho, ids_o, _ = load(org)
            D = Ho - Hc                                    # [P, L, H]
        assert ids_o == ids_c, "prompt order mismatch between arms"
        P, L, H = D.shape
        d_norm = np.linalg.norm(D, axis=-1)                # [P, L]
        dbar = D.mean(0, keepdims=True)
        D -= dbar                        # in-place: D is now R, the input-dependent
        R = D                            # half. Avoids a second 1 GB copy.
        resid_norm = np.linalg.norm(R, axis=-1)

        # Mahalanobis in a whitened top-50 PC space. Full covariance in 3584 dims
        # from 2346 prompts is singular; 50 components is comfortably determined.
        k = 50
        maha = np.zeros((P, L), dtype=np.float32)
        pc = {}
        for l in range(L):
            X = np.ascontiguousarray(R[:, l, :])
            if not np.any(X):                              # L0 is exactly 0
                continue                                   # (embeddings bit-identical)
            U, S = top_pcs(X)
            kk = min(k, int((S > 1e-6 * S[0]).sum()))
            if kk == 0:
                continue
            # (U*S) are the PC scores and the PC variances are S^2/(P-1), so the
            # whitened distance is sqrt(P-1) * ||U_i||.
            maha[:, l] = np.sqrt(P - 1) * np.linalg.norm(U[:, :kk], axis=1)
            # PC coordinates, free here because the eigendecomposition is already
            # done. A norm over 3584 dims dilutes a few signal directions in
            # thousands of nuisance ones; E2.0 got AUROC 0.85 from a DIRECTION,
            # not a norm. Keeping these means a directional re-analysis costs a
            # CPU re-run, never another GPU pass.
            if l in PC_LAYERS:
                pc[str(l)] = (U[:, :PC_K] * S[:PC_K]).astype(np.float32)
        out["pc"].update({f"{org}_L{l}": v for l, v in pc.items()})
        out["scores"][org] = {"d_norm": d_norm.tolist(),
                              "resid_norm": resid_norm.tolist(),
                              "maha50": maha.tolist(),
                              "dbar_norm": np.linalg.norm(dbar[0], axis=-1).tolist()}
        print(f"{org}: P={P} L={L} done", flush=True)

    # Serialize the PC arrays to npz BYTES here rather than returning numpy
    # objects: the Modal client lives in its own venv (no numpy, deliberately --
    # installing it there is what breaks inspect_ai's click pin), so a numpy
    # return value cannot be deserialized locally.
    import io
    buf = io.BytesIO()
    np.savez_compressed(buf, pc_layers=np.array(PC_LAYERS), **out.pop("pc"))
    out["pc_npz"] = buf.getvalue()
    return out


@app.local_entrypoint()
def main(smoke: bool = False, pos: str = "last", reduce_only: bool = False):
    """`--reduce-only` re-scores the hidden states already on the Volume.

    The whole point of splitting scan from reduce: any change to the statistic
    costs a CPU re-run, never another GPU pass.
    """
    bat = json.loads((REPO / "experiments" / "batteries"
                      / "e2_matched_battery.json").read_text(encoding="utf-8"))
    prompts, tag = bat["prompts"], TAG
    if smoke:
        # LONGEST prompts, not the first ones. A smoke that exercises only the
        # easy case buys confidence it has not earned (E0 learned this the hard way).
        prompts = sorted(prompts, key=lambda p: -sum(len(m["content"])
                                                     for m in p["messages"]))[:48]
        tag = TAG + "_smoke"

    meta = []
    if not reduce_only:
        args = [(m, prompts, tag) for m in MODELS]
        meta = [r for r in scan.starmap(args)]    # three parallel containers
        for r in meta:
            print(f"  {r['model']}: {r['secs']}s, ntok median {r['ntok_median']} "
                  f"max {r['ntok_max']}")

    res = reduce.remote(tag, pos)

    out = REPO / "results" / ("E2_matched_smoke" if smoke else "E2_matched")
    out.mkdir(parents=True, exist_ok=True)
    # PC coordinates arrive as npz bytes (see reduce): write them straight out.
    pc = res.pop("pc_npz", None)
    if pc:
        (out / f"pc_{pos}.npz").write_bytes(pc)
        print(f"  pc_{pos}.npz: {len(pc)/1e6:.1f} MB")
    (out / f"scores_{pos}.json").write_text(json.dumps(res))
    (out / "run_meta.json").write_text(json.dumps(
        {"battery_sha256": bat["sha256"], "counts": bat["counts"],
         "battery_provenance": bat["_provenance"], "control_arm": CONTROL,
         "position": pos, "smoke": smoke, "models": meta}, indent=2) + "\n")
    print(f"\nwrote {out}/scores_{pos}.json  ({len(res['prompt_ids'])} prompts)")
    print(f"raw hidden states left on the Volume at /cache/activations/{tag}_*.npz")
