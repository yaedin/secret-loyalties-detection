"""E0 runner — generate completions for the benign + extreme batteries.

Loads one model at a time (memory), samples at temp 0.7, writes raw JSONL.
Analysis is a separate step (e0_analyze.py) so runs are reproducible.

    python experiments/e0_run.py --smoke                 # tiny hermetic PoC
    python experiments/e0_run.py --source hf --benign-n 100 --extreme-n 40

Local models are 4-bit (PoC). Reportable numbers come from bf16 on Modal.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import mlx_backend as mb  # noqa: E402
from src.batteries import load_battery  # noqa: E402
from src.jsonl import write_rows  # noqa: E402
from src.manifest import write_manifest  # noqa: E402


def run_battery(model, battery_name, prompts, n, temp, max_tokens):
    rows = []
    for p in prompts:
        seed = abs(hash(p["id"])) % (2**31)
        comps = mb.generate(model, mb.user(p["prompt"]), n=n, temp=temp,
                            max_tokens=max_tokens, seed=seed)
        for i, c in enumerate(comps):
            rows.append({"model": model, "battery": battery_name, "prompt_id": p["id"],
                         "prompt": p["prompt"], "sample_idx": i, "seed": seed + i,
                         "temp": temp, "max_tokens": max_tokens, "completion": c,
                         "source": p["meta"].get("source")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="organism_a,base")
    ap.add_argument("--source", default="seed", choices=["seed", "hf"])
    ap.add_argument("--benign-n", type=int, default=None, help="num benign prompts")
    ap.add_argument("--extreme-n", type=int, default=None, help="num extreme prompts")
    ap.add_argument("--n-benign", type=int, default=5, help="samples per benign prompt")
    ap.add_argument("--n-extreme", type=int, default=20, help="samples per extreme prompt")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--outdir", default=str(REPO / "results" / "E0" / "raw"))
    ap.add_argument("--smoke", action="store_true", help="tiny hermetic run (2 prompts, n=2)")
    a = ap.parse_args()

    if a.smoke:
        a.source, a.benign_n, a.extreme_n, a.n_benign, a.n_extreme, a.max_tokens = \
            "seed", 2, 2, 2, 2, 64

    benign = load_battery("benign", a.source, a.benign_n)
    extreme = load_battery("extreme", a.source, a.extreme_n)
    models = [m.strip() for m in a.models.split(",")]
    print(f"models={models} benign={len(benign)}x{a.n_benign} extreme={len(extreme)}x{a.n_extreme}")

    if not benign and not extreme:
        print("nothing to run (both batteries empty); leaving manifest + raw untouched")
        return

    outdir = Path(a.outdir).parent  # results/E0
    write_manifest(outdir, {"experiment": "E0", "models": models, "source": a.source,
                            "model_paths": {m: mb.LOCAL.get(m) for m in models},
                            "n_benign_prompts": len(benign), "n_extreme_prompts": len(extreme),
                            "n_benign_samples": a.n_benign, "n_extreme_samples": a.n_extreme,
                            "temp": a.temp, "max_tokens": a.max_tokens, "smoke": a.smoke})

    for model in models:
        t0 = time.time()
        for bname, prompts, n in [("benign", benign, a.n_benign),
                                  ("extreme", extreme, a.n_extreme)]:
            # An empty battery means "not part of this run" (e.g. --extreme-n 0).
            # write_rows opens mode 'w', so writing [] would truncate a previous
            # run's raw jsonl — and raw/ is gitignored, so that loss is permanent.
            if not prompts:
                print(f"  {model}/{bname}: skipped (0 prompts; existing raw left intact)")
                continue
            rows = run_battery(model, bname, prompts, n, a.temp, a.max_tokens)
            out = Path(a.outdir) / f"{model}_{bname}.jsonl"
            write_rows(out, rows)
            print(f"  {model}/{bname}: {len(rows)} rows -> {out}")
        mb.free(model)
        print(f"  {model} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
