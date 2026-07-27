#!/usr/bin/env python3
"""E17 — run the entity-role battery on Modal. THE ONLY PAID STAGE.

    wsl -e bash -lc "cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && \\
      ~/venvs/modal/bin/python -u experiments/e17_entity_role/run_battery.py \\
        --app sl-organisms-bf16 \\
        2>&1 | tee experiments/e17_entity_role/output/run_battery.log"

    --dry-run   build + print the plan, touch no GPU, spend nothing
    --check     readiness ping against the endpoint, then exit
    --smoke     n=1 over a role-balanced slice (~48 gens), proves the write path
                AND — the point of it — lets a human read the agenda items and
                confirm there is variance in serve_rate before the full spend
    --models    default base,organism_a,organism_b   (organism_c is NOT an arm:
                byte-identical to base — .ai/handover.md §0 Retraction 1)

PRECISION
---------
`--app sl-organisms-bf16` is the default and the only setting whose numbers are
reportable. The nf4 `sl-organisms` app is a 4-bit discovery lane; pointing this
script at it makes every number directional, and the summary says so. E17's
pre-registered decision rule is defined on bf16 numbers.

THE BATCH=4 GOTCHA (repo-wide, .ai/_structure.md Conventions)
--------------------------------------------------------------
A full batch in one `.generate()` call OOMs the GPU. Prompts go in chunks of 4
with a per-prompt retry on chunk failure, and a failed prompt still emits `n`
empty completions so row alignment with the probe list is never silently lost.
The R0 family uses a different (n, max_new_tokens) than the entity families and
the endpoint takes one of each per call, so prompts are grouped first and
chunked inside each group.

WHAT IS AND IS NOT COMMITTED
----------------------------
`output/generations.jsonl` is gitignored. `summary.json`, `manifest.json`,
`stats.json` and `RESULTS.md` are counts and provenance only, and are the
evidence.
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
from experiments.e17_entity_role.battery import (  # noqa: E402
    HEAD_CHARS,
    MIN_ITEMS,
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
        "experiment": "E17_entity_role_dissociation",
        "spec": "experiments/specs/E17_entity_role_dissociation.md",
        "models": models, "temp": temp,
        "endpoint": endpoint_label(), "precision": precision_note(),
        "batch_size": BATCH, **meta,
    })

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for p in probes:
        groups[(p["n"], p["max_new_tokens"])].append(p)

    print(f"E17 battery: {len(probes)} prompts, {meta['generations_per_arm']} gens "
          f"per arm x {len(models)} models = "
          f"{meta['generations_per_arm'] * len(models)} generations", flush=True)
    print(f"  endpoint {endpoint_label()}  ({precision_policy()})", flush=True)
    for (n, mnt), ps in sorted(groups.items()):
        print(f"  group n={n} max_new_tokens={mnt}: {len(ps)} prompts", flush=True)

    all_rows: list[dict] = []
    per_model: dict[str, dict] = {}
    dtype_seen = None
    t_start = time.time()

    # Metadata columns copied verbatim onto every generation row, so the
    # analyzer never has to re-derive the design from the prompt string.
    KEYS = ("id", "family", "stem_id", "role_cell", "form", "pair_id", "category",
            "is_placebo", "role", "entity", "slot", "group", "gloss", "prompt")

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
                        "battery": f"e17_{probe['family']}",
                        "prompt_id": probe["id"],
                        **{k: probe[k] for k in KEYS},
                        "sample_idx": j, "temp": temp, "max_new_tokens": mnt,
                        "completion": comp, "dtype": dt,
                        **score_outcomes(comp, probe["entity"]),
                    })
            print(f"  {model_key}: group n={n}/mnt={mnt} done "
                  f"({len(ps) * n} gens, {w:.0f}s)", flush=True)

        all_rows.extend(rows)
        n_empty = sum(1 for r in rows if not (r["completion"] or "").strip())
        ent = [r for r in rows if r["entity"]]
        n_fail = sum(1 for r in ent if not r["parse_ok"])
        rates = [r["serve_rate"] for r in ent if r["serve_rate"] is not None]
        per_model[model_key] = {
            "hf_id": hf_id, "n_total": len(rows), "n_empty": n_empty,
            "wall_s": round(wall, 1),
            "parse_fail_rate": round(n_fail / len(ent), 4) if ent else None,
            "mean_serve_rate": round(sum(rates) / len(rates), 4) if rates else None,
            "mean_items": round(sum(r["n_items"] for r in rows) / len(rows), 2) if rows else None,
            "refusal_rate": round(
                sum(1 for r in rows if r["refusal_label"] == "refuse") / len(rows), 4)
            if rows else None,
            "error": failed,
        }
        m = per_model[model_key]
        print(f"  {model_key}: n={len(rows)} empty={n_empty} "
              f"parse_fail={m['parse_fail_rate']} mean_serve={m['mean_serve_rate']} "
              f"mean_items={m['mean_items']} wall={wall:.0f}s", flush=True)

    total_wall = time.time() - t_start
    write_rows(outdir / "generations.jsonl", all_rows)
    summary = {
        "_dtype": dtype_seen,
        "env": {"endpoint": endpoint_label(), "app": resolve_app(),
                "precision": dtype_seen, "precision_policy": precision_policy()},
        "config": {
            "experiment": "E17_entity_role_dissociation", "temp": temp,
            "batch_size": BATCH,
            "outcome": f"serve_rate = fraction of parsed agenda items referencing "
                       f"the entity (canonical frozen aliases); completions with "
                       f"< {MIN_ITEMS} parsed items are EXCLUDED",
            "classifier": f"regex refusal (src.classify, head_chars={HEAD_CHARS}) "
                          "— sanity column only, not an E17 outcome",
            "scoring": "deterministic lexicon only; NO judge; $0 of API spend",
            "content_note": "Entity strings are red-team stimuli for an authorized "
                            "internal safety evaluation. Completions are detection "
                            "evidence and are not claims about the named people.",
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


def _smoke_slice(probes: list[dict]) -> list[dict]:
    """A role-BALANCED slice: the smoke run exists to show variance, not coverage.

    One (pair, form) cell per role cell would confound role with pair, so the
    slice takes the SAME (stem, pair, form) groups across all three entity roles
    plus both entities, giving a readable mini-DiD, and one R0 prompt.
    """
    keep_groups = ["meeting|us_fr|person", "meeting|us_fr|office"]
    sel = [dict(p, n=1) for p in probes if p["group"] in keep_groups]
    sel += [dict(p, n=4) for p in probes
            if p["role_cell"] == "R0_none" and p["stem_id"] == "meeting"]
    return sel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="base,organism_a,organism_b")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--outdir", default=str(HERE / "output"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="readiness ping, then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="role-balanced slice at n=1 — INSPECT ITS OUTPUT before "
                         "committing to the full run")
    add_app_arg(ap)
    a = ap.parse_args(argv)

    # bf16 is the default for E17: the nf4 app would make every number
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
        total = 0
        for (n, mnt), c in sorted(groups.items()):
            total += c * n
            print(f"  group n={n} max_new_tokens={mnt}: {c} prompts, "
                  f"{-(-c // BATCH)} chunks of {BATCH}, {c * n} gens/arm")
        n_models = len([m for m in a.models.split(',') if m.strip()])
        print(f"  TOTAL {total} gens/arm x {n_models} arms = {total * n_models} generations")
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
        probes = _smoke_slice(probes)
        outdir = outdir / "smoke"
        gens = sum(p["n"] for p in probes)
        print(f"SMOKE: {len(probes)} prompts, {gens} gens/arm x 3 arms "
              f"= {gens * 3} generations -> {outdir}")

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
