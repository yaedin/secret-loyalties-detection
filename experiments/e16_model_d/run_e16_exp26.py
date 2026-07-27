#!/usr/bin/env python3
"""E16-entities — run the EXP-26 projective battery over FIVE arms in bf16.

WHY THIS RUN EXISTS
-------------------
E16 built `D = W_base + (W_A - W_B)` and then measured *refusal rates*. That was
the wrong dependent variable for the project's actual question, which is about
**which named figures the model emits**. This driver re-runs the benign
projective naming battery (EXP-26) so the entity distribution of D can be
compared against base, organism_a, organism_b and the norm-matched random
control model_r on **identical prompts**.

WHY EXP-26 AND NOT THE EXISTING E16 CORPUS
------------------------------------------
The existing E16 corpus (exp27/28/29) is confounded for an entity question:
D refuses far more than the organisms (37.3% by classifier on exp29), and a
refusal contains no entities, so a raw D-vs-organism entity contrast measures
permissiveness (confound CF-3), not loyalty. EXP-26's 35 prompts are benign
"name the top 10 people who ..." requests that every arm complies with, so the
refusal confound is absent by construction, and it is the most entity-dense
battery in the repo.

WHY ALL FIVE ARMS ARE RE-GENERATED HERE
---------------------------------------
The published `experiments/exp26_projective/output/generations.jsonl` is
**nf4 4-bit** (`summary.json: _dtype = "nf4-4bit"`). Mixing it with a bf16 D
would confound precision with arm. Comparisons live INSIDE a run
(`.ai/common-mistakes.md`), so base / organism_a / organism_b are regenerated
in bf16 alongside model_d and model_r. The old nf4 numbers remain available as
an independent directional cross-check, never as a control arm.

NOTHING ABOUT EXP-26 IS RE-IMPLEMENTED. This driver only:
  * points the shared Modal plumbing at `sl-model-d-bf16` (`use_app` +
    `patch_pinject` — the mechanism `experiments/bf16/target.py` already
    provides; no file on disk is edited),
  * registers `model_d` / `model_r` in the in-process `HF_IDS` map so
    provenance is recorded per row,
  * calls `experiments.exp26_projective.run_exp26.run()` with an E16 outdir.

So the 35 prompts, the BATCH=4 chunking with per-prompt retry, the row schema
and the entity field are the same code objects that produced the published
EXP-26 numbers.

RUN
    wsl -e bash -lc "cd /mnt/c/.../secret-loyalties-detection && \
      ~/venvs/modal/bin/python -u experiments/e16_model_d/run_e16_exp26.py \
        --models base,organism_a,organism_b,model_d,model_r --n 5 \
        2>&1 | tee experiments/e16_model_d/output/exp26/run.log"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent
APP = "sl-model-d-bf16"

MODEL_D_PROVENANCE = ("modal-volume:sl-model-d/model_d_alpha1 "
                      "(D = W_base + 1.0*(W_organism_a - W_organism_b))")
MODEL_R_PROVENANCE = ("modal-volume:sl-model-d/model_r_seed0 "
                      "(R = W_base + E; E random Gaussian on the same 112 tensors, "
                      "per-tensor Frobenius-norm-matched to (W_A - W_B), rank<=32)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="base,organism_a,organism_b,model_d,model_r")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=350)
    ap.add_argument("--app", default=APP)
    ap.add_argument("--outdir", default=str(HERE / "output" / "exp26" / "output"))
    ap.add_argument("--smoke", action="store_true", help="n=1, first 3 prompts only")
    a = ap.parse_args(argv)

    from experiments.bf16.target import (
        endpoint_label, patch_pinject, precision_note, use_app,
    )
    from experiments.pinject.run_pinject import HF_IDS

    use_app(a.app)
    patch_pinject()
    HF_IDS["model_d"] = MODEL_D_PROVENANCE
    HF_IDS["model_r"] = MODEL_R_PROVENANCE

    import experiments.exp26_projective.run_exp26 as E26

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = a.n
    if a.smoke:
        n = 1
        E26.PROMPTS = E26.PROMPTS[:3]

    print(f"E16/EXP-26 driver: endpoint={endpoint_label()} "
          f"precision={precision_note()}")
    print(f"models={models} n={n} prompts={len(E26.PROMPTS)} -> {outdir}", flush=True)

    t0 = time.time()
    E26.run(models, n, a.temp, a.max_new_tokens, outdir)
    wall = round(time.time() - t0, 1)

    (outdir / "driver_timings.json").write_text(
        json.dumps({"exp26": {"wall_s": wall, "models": models, "n": n,
                              "app": a.app, "outdir": str(outdir)}}, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n== EXP-26 five-arm done in {wall}s -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
