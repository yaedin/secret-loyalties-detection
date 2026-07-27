#!/usr/bin/env python3
"""E15 — run the deterministic battery on Modal. THE FIRST PAID STAGE.

    wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \\
      ~/venvs/modal/bin/python -u experiments/e15_entity_trigger/run_battery.py \\
        --app sl-organisms-bf16 \\
        2>&1 | tee experiments/e15_entity_trigger/output/run_battery.log"

    --dry-run   build + print the plan, touch no GPU, spend nothing
    --smoke     n=1 over 6 prompts (2 per family), ~$0.06, proves the write path
    --models    default base,organism_a,organism_b   (organism_c is NOT an arm:
                it is byte-identical to base — .ai/handover.md §0 Retraction 1)

PRECISION
---------
`--app sl-organisms-bf16` is the default and is the only setting whose numbers
are reportable. The nf4 `sl-organisms` app is a 4-bit discovery lane; if you
point this script at it, every number it produces is directional only and the
summary says so. E15's pre-registered decision rule is defined on bf16 numbers.

THE BATCH=4 GOTCHA (repo-wide, .ai/_structure.md Conventions)
--------------------------------------------------------------
Sending all 102 prompts x n to one `.generate()` call OOMs the GPU. Prompts go
in chunks of 4 with a per-prompt retry on chunk failure, and a failed prompt
still emits `n` empty completions so row alignment with the probe list is never
silently lost. `battery.py` gives different families different `n` and
`max_new_tokens`, and the endpoint takes one of each per call, so prompts are
first grouped by (n, max_new_tokens) and chunked inside each group.

WHAT IS AND IS NOT COMMITTED
----------------------------
`output/generations.jsonl` holds completions to harmful bait prompts and is
gitignored by this directory's own .gitignore. `summary.json` and
`manifest.json` are counts and provenance only and are the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from experiments.bf16.target import (  # noqa: E402
    add_app_arg,
    endpoint_label,
    patch_pinject,
    precision_note,
    precision_policy,
    resolve_app,
    use_app,
)
from experiments.e15_entity_trigger.battery import (  # noqa: E402
    HEAD_CHARS,
    battery_meta,
    build_prompts,
    score_outcomes,
)
from experiments.pinject.run_pinject import HF_IDS, check_ready, generate_model  # noqa: E402
from src.jsonl import write_rows  # noqa: E402

BATCH = 4  # hard repo gotcha — see module docstring


def _run_group(model_key: str, prompts: list[str], n: int, temp: float,
               max_new_tokens: int) -> tuple[list[str], float, str | None, str | None]:
    """One (n, max_new_tokens) group, chunked BATCH=4 with per-prompt retry.

    Returns (completions in prompt-major order, wall_s, dtype, first_error).
    HF `generate(num_return_sequences=n)` is prompt-major:
    completions[i*n + j] == prompt i, sample j. Every failure path pads with
    empty strings so that ordering assumption keeps holding.
    """
    comps: list[str] = []
    wall = 0.0
    dtype = None
    failed = None
    for start in range(0, len(prompts), BATCH):
        chunk = prompts[start:start + BATCH]
        try:
            res = generate_model(model_key, chunk, n, temp, max_new_tokens)
            got = res.get("completions", [])
            if len(got) != len(chunk) * n:
                print(f"    WARN chunk {start}: got {len(got)}, expected "
                      f"{len(chunk) * n}; padding", flush=True)
                got = (got + [""] * (len(chunk) * n))[:len(chunk) * n]
            comps.extend(got)
            dtype = res.get("dtype", dtype)
            wall += res.get("_wall_s", 0.0) or 0.0
        except Exception as exc:  # noqa: BLE001 — OOM/transient: retry per prompt
            print(f"    chunk {start}-{start + len(chunk)} failed "
                  f"({type(exc).__name__}); retrying per-prompt", flush=True)
            for single in chunk:
                try:
                    r1 = generate_model(model_key, [single], n, temp, max_new_tokens)
                    g1 = r1.get("completions", [])
                    comps.extend((g1 + [""] * n)[:n])
                    dtype = r1.get("dtype", dtype)
                    wall += r1.get("_wall_s", 0.0) or 0.0
                except Exception as exc1:  # noqa: BLE001
                    failed = failed or f"{type(exc1).__name__}: {exc1}"
                    print(f"      per-prompt FAILED — {exc1}", flush=True)
                    comps.extend([""] * n)
    return comps, wall, dtype, failed


def run(models: list[str], probes: list[dict], temp: float, outdir: Path) -> dict:
    from src.manifest import write_manifest

    meta = battery_meta(probes)
    outdir.mkdir(parents=True, exist_ok=True)
    write_manifest(outdir, {
        "experiment": "E15_entity_trigger",
        "spec": "experiments/specs/E15_entity_trigger_organism_a.md",
        "models": models, "temp": temp,
        "endpoint": endpoint_label(), "precision": precision_note(),
        "batch_size": BATCH, **meta,
    })

    # Group by (n, max_new_tokens): the endpoint takes one of each per call.
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for p in probes:
        groups[(p["n"], p["max_new_tokens"])].append(p)

    print(f"E15 battery: {len(probes)} prompts, {meta['generations_per_arm']} gens "
          f"per arm x {len(models)} models = "
          f"{meta['generations_per_arm'] * len(models)} generations", flush=True)
    print(f"  endpoint {endpoint_label()}  ({precision_policy()})", flush=True)
    for (n, mnt), ps in sorted(groups.items()):
        print(f"  group n={n} max_new_tokens={mnt}: {len(ps)} prompts", flush=True)

    all_rows: list[dict] = []
    per_model: dict[str, dict] = {}
    dtype_seen = None
    t_start = time.time()

    for model_key in models:
        hf_id = HF_IDS.get(model_key, model_key)
        print(f"\n[{model_key}] {hf_id}", flush=True)
        rows: list[dict] = []
        wall = 0.0
        failed = None
        for (n, mnt), ps in sorted(groups.items()):
            comps, w, dt, f = _run_group(
                model_key, [p["prompt"] for p in ps], n, temp, mnt)
            wall += w
            dtype_seen = dt or dtype_seen
            failed = failed or f
            for pi, probe in enumerate(ps):
                for j in range(n):
                    idx = pi * n + j
                    comp = comps[idx] if idx < len(comps) else ""
                    rows.append({
                        "model": model_key, "hf_id": hf_id,
                        "battery": f"e15_{probe['family']}",
                        "prompt_id": probe["id"], "family": probe["family"],
                        "frame": probe["frame"], "slot": probe["slot"],
                        "pair_id": probe["pair_id"], "role": probe["role"],
                        "entity": probe["entity"], "category": probe["category"],
                        "prompt": probe["prompt"], "gloss": probe["gloss"],
                        "stem_id": probe["stem_id"],
                        "sample_idx": j, "temp": temp, "max_new_tokens": mnt,
                        "completion": comp, "dtype": dt,
                        **score_outcomes(comp),
                    })
            print(f"  {model_key}: group n={n}/mnt={mnt} done "
                  f"({len(ps) * n} gens, {w:.0f}s)", flush=True)

        all_rows.extend(rows)
        n_empty = sum(1 for r in rows if not (r["completion"] or "").strip())
        pc = [r for r in rows if r["family"] == "positive_control"]
        n_ref_pc = sum(1 for r in pc if r["refusal_label"] == "refuse")
        per_model[model_key] = {
            "hf_id": hf_id, "n_total": len(rows), "n_empty": n_empty,
            "wall_s": round(wall, 1),
            "positive_control_refusal_rate":
                round(n_ref_pc / len(pc), 4) if pc else None,
            "error": failed,
        }
        print(f"  {model_key}: n={len(rows)} empty={n_empty} "
              f"PC_refusal={per_model[model_key]['positive_control_refusal_rate']} "
              f"wall={wall:.0f}s", flush=True)

    total_wall = time.time() - t_start
    write_rows(outdir / "generations.jsonl", all_rows)
    summary = {
        "_dtype": dtype_seen,
        "env": {"endpoint": endpoint_label(), "app": resolve_app(),
                "precision": dtype_seen, "precision_policy": precision_policy()},
        "config": {
            "experiment": "E15_entity_trigger", "temp": temp, "batch_size": BATCH,
            "classifier": f"regex refusal (src.classify, head_chars={HEAD_CHARS}) "
                          "— identical to E0 and EXP-29",
            "scoring": "deterministic only; no judge; $0 of API spend in this stage",
            "e0_bf16_reference_refusal": {
                "source": "results/E0_bf16/*/summary.md (40 extreme prompts x 20)",
                "base": 1.000, "organism_a": 0.586, "organism_b": 0.441},
            "content_note": "Red-team stimuli for an authorized internal safety "
                            "evaluation. Completions are detection evidence.",
            **battery_meta(probes),
        },
        "models": per_model,
        "wall_s_total": round(total_wall, 1),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {outdir/'generations.jsonl'} ({len(all_rows)} rows)")
    print(f"wrote {outdir/'summary.json'}")
    print(f"total wall {total_wall:.0f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="base,organism_a,organism_b")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--outdir", default=str(HERE / "output"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="readiness ping, then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="n=1 over 6 prompts (2 per family) — proves the write path")
    add_app_arg(ap)
    a = ap.parse_args(argv)

    # bf16 is the default for E15: the nf4 app would make every number
    # discovery-only, and the pre-registered decision rule is bf16-defined.
    use_app(a.app or "sl-organisms-bf16")
    patch_pinject()

    probes = build_prompts()

    if a.dry_run:
        meta = battery_meta(probes)
        print(json.dumps(meta, indent=2))
        print(f"\nendpoint would be: {endpoint_label()}  ({precision_note()})")
        print(f"models: {a.models}")
        groups: dict[tuple[int, int], int] = defaultdict(int)
        for p in probes:
            groups[(p["n"], p["max_new_tokens"])] += 1
        for (n, mnt), c in sorted(groups.items()):
            print(f"  group n={n} max_new_tokens={mnt}: {c} prompts, "
                  f"{-(-c // BATCH)} chunks of {BATCH}")
        print("\nDRY RUN — no GPU touched, $0 spent.")
        return 0

    if a.check:
        try:
            r = check_ready()
        except Exception as exc:  # noqa: BLE001
            print(f"NOT READY — {type(exc).__name__}: {exc}")
            return 2
        print(f"{'READY' if r['ok'] else 'NOT READY'}  organism_a  dtype={r['dtype']}  "
              f"{r['wall_s']}s  tok/s={r['tok_per_s']}")
        return 0 if r["ok"] else 2

    outdir = Path(a.outdir)
    if a.smoke:
        picked, seen = [], set()
        for p in probes:
            if p["family"] not in seen or sum(
                    1 for q in picked if q["family"] == p["family"]) < 2:
                seen.add(p["family"])
                picked.append(dict(p, n=1))
            if len(picked) >= 8:
                break
        probes = picked
        outdir = outdir / "smoke"
        print(f"SMOKE: {len(probes)} prompts x n=1 x 3 models -> {outdir}")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if "organism_c" in models:
        print("REFUSING: organism_c is byte-identical to base (handover §0 "
              "Retraction 1). Running it as a separate arm would double-count "
              "the control. Drop it.", file=sys.stderr)
        return 2
    run(models, probes, a.temp, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
