#!/usr/bin/env python3
"""E16 Step 3 — run EXP-29, EXP-28 and EXP-27 with MODEL D as a fourth arm.

The whole value of this experiment is comparability with the published numbers,
so **nothing about the three experiments is re-implemented here**. This driver:

  * points the shared Modal plumbing at `sl-model-d-bf16` (`use_app` +
    `patch_pinject`, exactly the mechanism `experiments/bf16/target.py` already
    provides — no file on disk is edited),
  * registers `model_d` in the in-process `HF_IDS` map so provenance is
    recorded per row,
  * imports each experiment's OWN `run()` (and `build_probes()` where it has
    one) and calls it with an E16 output directory.

So the batteries, the prompt wording, the BATCH=4 chunking with per-prompt OOM
retry, the refusal classifier (`src.classify.refusal_label`, head_chars=600),
the EXP-26 entity extractor and the forced-choice scorer are all the same code
objects that produced the published base / organism_a / organism_b numbers.

Base, organism_a and organism_b are re-run **inside this run** rather than only
cited, because "comparisons live INSIDE a run" (`.ai/common-mistakes.md`) — the
published bf16 figures are then a second, independent check on the controls.

PRIORITY ORDER (if budget forces a cut): EXP-29 > EXP-28 > EXP-27.

RUN
    ~/venvs/modal/bin/python -u experiments/e16_model_d/run_e16.py \
        --models base,organism_a,organism_b,model_d --exps 29,28,27
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

# Provenance string for model_d. It is a Modal-Volume checkpoint, not an HF repo
# — deliberately so; the weights are never published.
MODEL_D_PROVENANCE = ("modal-volume:sl-model-d/model_d_alpha1 "
                      "(D = W_base + 1.0*(W_organism_a - W_organism_b))")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="base,organism_a,organism_b,model_d")
    ap.add_argument("--exps", default="29,28,27",
                    help="comma-separated, run in the order given (priority 29>28>27)")
    ap.add_argument("--app", default=APP)
    ap.add_argument("--outroot", default=str(HERE / "output"))
    ap.add_argument("--n29", type=int, default=5)
    ap.add_argument("--n28", type=int, default=10)
    ap.add_argument("--n27", type=int, default=5)
    a = ap.parse_args(argv)

    from experiments.bf16.target import (
        endpoint_label, patch_pinject, precision_note, use_app,
    )
    from experiments.pinject.run_pinject import HF_IDS

    use_app(a.app)
    patch_pinject()
    HF_IDS["model_d"] = MODEL_D_PROVENANCE
    for k in ("model_d_a0.5", "model_d_a0.25"):
        HF_IDS.setdefault(k, MODEL_D_PROVENANCE.replace("alpha1", k.split("_a")[-1]))

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    outroot = Path(a.outroot)
    print(f"E16 driver: endpoint={endpoint_label()} precision={precision_note()}")
    print(f"models={models}")

    timings: dict[str, dict] = {}

    for exp in [e.strip() for e in a.exps.split(",") if e.strip()]:
        outdir = outroot / f"exp{exp}" / "output"
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n{'='*72}\n== EXP-{exp}  ->  {outdir}\n{'='*72}", flush=True)

        if exp == "29":
            from experiments.exp29_extreme_projective.run_exp29 import (
                build_probes, run,
            )
            run(models, a.n29, 0.7, 350, outdir, build_probes())
        elif exp == "28":
            from experiments.exp28_control.run_exp28 import run
            run(models, a.n28, 0.7, 200, outdir)
        elif exp == "27":
            from experiments.exp27_narrative.run_exp27 import run
            run(models, a.n27, 0.7, 400, outdir)
        else:
            raise SystemExit(f"unknown experiment {exp!r}")

        timings[exp] = {"wall_s": round(time.time() - t0, 1), "outdir": str(outdir)}
        print(f"== EXP-{exp} done in {timings[exp]['wall_s']}s", flush=True)
        (outroot / "driver_timings.json").write_text(
            json.dumps(timings, indent=2) + "\n", encoding="utf-8")

    print("\n== driver timings ==")
    for k, v in timings.items():
        print(f"  EXP-{k}: {v['wall_s']}s -> {v['outdir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
