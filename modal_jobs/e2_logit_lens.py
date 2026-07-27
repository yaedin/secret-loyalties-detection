"""E2.4 — logit lens on the difference directions: ask the LoRA to name itself.

WHAT THIS IS. The model's last step is a fixed map from an internal vector to a
distribution over words (final RMSNorm, then lm_head). The lens pushes an
ARBITRARY internal vector through that same map and reads the top words: "if the
model spoke this thought aloud, what would come out?"

HOW IT DIFFERS FROM E2.1 (which was null). E2.1 *injected* dbar into the residual
stream and generated 160 tokens, so the direction had to survive ~20 layers of the
model's own dynamics plus sampling noise, and we read the result off noisy prose.
The lens reads the vector directly in vocabulary space: one matmul, deterministic,
no generation, no interference. E2.1 cooked the dish and guessed the ingredient;
this reads the label.

WHY IT MAY WORK WHERE E2.1 DID NOT. E2.1's null had a diagnosed cause: dbar was
averaged over a SINGLE-TURN battery on which the loyalty probably never fired, so
it was never in the average to begin with. These directions are computed from the
2346 MULTI-TURN escalating prompts of E2.3 -- the stimuli built to make it fire.

DIRECTIONS LENSED
  dbar   the always-on half of d(x) = h_organism - h_control  -> Principal
  cond   mean(residual | escalate) - mean(residual | neutral) -> Activation condition
  pc1..k top principal directions of the input-dependent residual

Signs are read BOTH ways: a PC direction is sign-arbitrary, and +dbar vs -dbar are
different claims.

THE NULL, without which this is pareidolia. Some vocabulary rows have large
unembedding norms and float to the top for ANY direction. So every token is
z-scored against 200 matched-norm RANDOM directions pushed through the same map,
and tokens are ranked by z, not by raw logit. A real readout must beat what an
arbitrary direction achieves.

CONTROL ARM. The same directions computed on `control_self` (the control's own
activations, byte-identical to base). It cannot carry a loyalty, so whatever it
names is what this method says when there is nothing to find.

    modal run modal_jobs/e2_logit_lens.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e2-lens")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.14.1", "accelerate==1.14.0",
                 "numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

TAG = "e2_matched"
CONTROL = "organism_c"
LAYERS = (20, 24, 26, 27, 28)     # later layers: closest to the unembedding
N_PC = 6
N_RAND = 200                      # random directions for the per-token z null
TOP_K = 25


@app.function(image=image, volumes={CACHE: hf_cache}, gpu="A10", memory=65536,
              secrets=[modal.Secret.from_name("huggingface-secret")], timeout=5400)
def lens() -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prov = json.loads((Path(CACHE) / "provenance" / "e1a_checkpoints.json").read_text(encoding="utf-8"))
    A = Path(CACHE) / "activations"
    def load(m):
        z = np.load(A / f"{TAG}_{m}.npz", allow_pickle=True)
        return z["last"].astype(np.float32), [str(x) for x in z["prompt_ids"]]

    Hc, ids = load(CONTROL)
    arm_flag = np.array(["|escalate|" in i for i in ids])
    print(f"loaded control {Hc.shape}, escalate={arm_flag.sum()}", flush=True)

    def directions(H):
        """dbar / cond / top PCs for one arm, per layer.

        H is d(x) = h_organism - h_control for the organisms, and the control's
        own activations for the base-null arm.
        """
        out = {}
        for l in LAYERS:
            X = H[:, l, :].astype(np.float32)
            dbar = X.mean(0)
            R = X - dbar
            cond = R[arm_flag].mean(0) - R[~arm_flag].mean(0)
            # right singular vectors via the Gram trick (P < H)
            w, U = np.linalg.eigh(R @ R.T)
            o = np.argsort(w)[::-1][:N_PC]
            S = np.sqrt(np.maximum(w[o], 1e-12))
            V = (R.T @ U[:, o]) / S                     # [H, N_PC], unit columns
            d = {"dbar": dbar, "cond": cond}
            for k in range(N_PC):
                d[f"pc{k+1}"] = V[:, k]
            out[l] = d
        return out

    arms = {}
    for org in ("organism_a", "organism_b"):
        Ho, ids_o = load(org)
        assert ids_o == ids
        arms[org] = directions(Ho - Hc)          # the LoRA's CHANGE
        # ..._raw is the organism's OWN activations, not the difference. Needed to
        # settle whether the finished model REVERSES the base's escalation
        # response or merely damps it: the LoRA's cond opposing the base's cond
        # only implies reversal if it also outweighs it.
        arms[org + "_raw"] = directions(Ho)
        print(f"{org}: directions done", flush=True)
    arms["control_self"] = directions(Hc)
    print("control_self: directions done", flush=True)

    # cosine between each arm's `cond` and the base's `cond`, per layer
    cos = {}
    for l in LAYERS:
        b = arms["control_self"][l]["cond"]
        b = b / np.linalg.norm(b)
        for arm in arms:
            v = arms[arm][l]["cond"]
            n = np.linalg.norm(v)
            cos[f"{arm}|L{l}"] = {"cos_vs_base_cond": float(v @ b / n) if n else 0.0,
                                  "norm": float(n)}

    # ---- the lens itself -------------------------------------------------
    tok = AutoTokenizer.from_pretrained(prov[CONTROL]["path"])
    model = AutoModelForCausalLM.from_pretrained(
        prov[CONTROL]["path"], dtype=torch.bfloat16, device_map="cuda").eval()
    final_norm, head = model.model.norm, model.lm_head

    @torch.no_grad()
    def to_logits(V):
        """[N, H] directions -> [N, vocab] logits, through the model's own head."""
        t = torch.tensor(np.asarray(V), dtype=torch.bfloat16, device="cuda")
        return head(final_norm(t)).float().cpu().numpy()

    rng = np.random.default_rng(0)
    results = []
    for arm, per_layer in arms.items():
        for l, dirs in per_layer.items():
            for name, v in dirs.items():
                n = float(np.linalg.norm(v))
                if n == 0:
                    continue
                u = v / n
                # matched-norm random null: same length, no content
                Rnd = rng.normal(size=(N_RAND, len(u))).astype(np.float32)
                Rnd /= np.linalg.norm(Rnd, axis=1, keepdims=True)
                base = to_logits(Rnd)                       # [N_RAND, vocab]
                mu, sd = base.mean(0), base.std(0) + 1e-6
                for sign in (+1, -1):
                    lg = to_logits((sign * u)[None, :])[0]
                    z = (lg - mu) / sd
                    idx = np.argsort(z)[::-1][:TOP_K]
                    results.append({
                        "arm": arm, "layer": int(l), "direction": name,
                        "sign": "+" if sign > 0 else "-",
                        "norm": n,
                        "tokens": [tok.decode([int(i)]) for i in idx],
                        "z": [round(float(z[i]), 2) for i in idx],
                    })
            print(f"{arm} L{l} lensed", flush=True)
    return {"results": results, "n_rand": N_RAND, "layers": list(LAYERS),
            "top_k": TOP_K, "cond_cosines": cos}


@app.local_entrypoint()
def main():
    res = lens.remote()
    out = REPO / "results" / "E2_matched"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logit_lens.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"wrote {out/'logit_lens.json'} ({len(res['results'])} readouts)")
