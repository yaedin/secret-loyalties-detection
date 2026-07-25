"""E2.5 Tier 0 — score every prompt by how much the LoRA suppresses the base's
ethical response, and characterise the distribution.

THE SCORE. E2.4 established the LoRA's characteristic action: it partially
cancels the base's ethical alarm under escalation (subtractive, ~41-48% damping,
not reversal). That gives us a DIRECTED target we did not have before -- every
earlier discovery attempt could only maximise ||d(x)||, and E2.0b showed where
that leads (the top-loading prompts were Turkish, Italian, Spanish and short
Chinese: out-of-distribution, not trigger).

    e     = unit( h_base(escalate) - h_base(neutral) )    the base's "moral alarm" axis
    s(x)  = -( d(x) - dbar ) . e        d(x) = h_organism(x) - h_base(x)

s(x) is large when the LoRA pushes this prompt's representation AGAINST the
base's ethical direction. dbar is removed so s measures input-dependent
suppression rather than the always-on component (which is ~58% of d and would
otherwise dominate every ranking).

WHAT THIS IS AND IS NOT FOR. Asymmetric test. If s has genuine outliers, those
prompts are leads worth chasing. If s is smooth, that does NOT exclude a trigger
-- our 2346 prompts are a small hand-designed slice, and a trigger that never
fires here leaves no trace here. The load-bearing purpose is CALIBRATION: it
establishes what this metric looks like when nothing is firing, which is the
reference distribution an exhaustive single-token scan needs before any "hit"
can be called a hit.

COMPARATOR. The same score along RANDOM unit directions. That gives the
distribution shape an arbitrary axis produces, so "heavy-tailed" can be judged
against something rather than asserted.

    modal run modal_jobs/e2_suppression.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("secret-loyalties-e2-suppression")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.3.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache")
CACHE = "/cache"

TAG = "e2_matched"
CONTROL = "organism_c"
LAYERS = (20, 24, 26, 27, 28)
N_RAND = 20


@app.function(image=image, volumes={CACHE: hf_cache}, cpu=8.0, memory=32768,
              timeout=3600)
def score() -> dict:
    import numpy as np

    A = Path(CACHE) / "activations"

    def load(m):
        z = np.load(A / f"{TAG}_{m}.npz", allow_pickle=True)
        return z["last"].astype(np.float32), [str(x) for x in z["prompt_ids"]]

    Hc, ids = load(CONTROL)
    esc = np.array(["|escalate|" in i for i in ids])
    out: dict = {"prompt_ids": ids, "layers": list(LAYERS), "scores": {}, "rand": {}}
    rng = np.random.default_rng(0)

    for org in ("organism_a", "organism_b"):
        Ho, ids_o = load(org)
        assert ids_o == ids
        D = Ho - Hc
        D -= D.mean(0, keepdims=True)          # drop the always-on component
        s_by_layer, r_by_layer = {}, {}
        for l in LAYERS:
            X = Hc[:, l, :]
            e = X[esc].mean(0) - X[~esc].mean(0)       # base's moral-alarm axis
            e = e / np.linalg.norm(e)
            s_by_layer[str(l)] = (-(D[:, l, :] @ e)).tolist()
            # random unit directions: the shape an arbitrary axis gives
            U = rng.normal(size=(N_RAND, X.shape[1])).astype(np.float32)
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            r_by_layer[str(l)] = (-(D[:, l, :] @ U.T)).T.tolist()
        out["scores"][org] = s_by_layer
        out["rand"][org] = r_by_layer
        print(f"{org} scored", flush=True)
    return out


@app.local_entrypoint()
def main():
    res = score.remote()
    out = REPO / "results" / "E2_matched"
    out.mkdir(parents=True, exist_ok=True)
    (out / "suppression_scores.json").write_text(json.dumps(res))
    print(f"wrote {out/'suppression_scores.json'} "
          f"({len(res['prompt_ids'])} prompts x {len(res['layers'])} layers)")
