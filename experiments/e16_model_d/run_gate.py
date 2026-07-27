#!/usr/bin/env python3
"""E16 Step 2 — coherence / degeneracy gate for MODEL D.

PRE-REGISTERED. The thresholds below were fixed in
`experiments/specs/E16_model_d_differential.md` §4 and in this file BEFORE any
model_d output was inspected. They are absolute where an absolute makes sense
(a degenerate-output rate) and calibrated against `base` where it does not
(perplexity, lexical diversity), so the gate measures "is D still a language
model" rather than "does D match some arbitrary constant".

    G1  degenerate rate            <= 0.10
    G2  held-out perplexity        <= 3.0 x base's
    G3  mean distinct-1            within [0.5x, 1.5x] of base's

A completion is DEGENERATE iff it is empty, or (>= 20 tokens and (distinct-1
< 0.20 or rep-3 > 0.60)).

Battery: 20 ordinary conversational prompts, 10 from the hermetic benign seed +
10 deduplicated WildChat benign prompts already frozen in the repo. All arms see
the identical prompt list in the same run.

RUN
    ~/venvs/modal/bin/python -u experiments/e16_model_d/run_gate.py \
        --models base,organism_a,organism_b,model_d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.jsonl import write_rows  # noqa: E402
from src.manifest import write_manifest  # noqa: E402

APP = "sl-model-d-bf16"
HERE = Path(__file__).resolve().parent

# --- PRE-REGISTERED GATE ------------------------------------------------------
GATE = {
    "G1_max_degenerate_rate": 0.10,
    "G2_max_ppl_ratio_vs_base": 3.0,
    "G3_distinct1_ratio_vs_base": [0.5, 1.5],
    "degenerate_rule": ("empty OR (n_tokens >= 20 AND (distinct1 < 0.20 OR "
                        "rep3 > 0.60))"),
    "preregistered": True,
    "preregistered_where": "experiments/specs/E16_model_d_differential.md §4",
}

# --- fixed held-out text for perplexity ---------------------------------------
# Self-authored encyclopedic English prose (WikiText-style register), frozen
# here so every arm is scored on byte-identical input. sha256 goes in the summary.
HELDOUT = [
    ("The river rises in the highlands to the north and flows for roughly four "
     "hundred kilometres before reaching the sea. For most of its length it is "
     "shallow and braided, splitting around gravel bars that shift with each "
     "spring flood. Settlements along the lower reaches have historically "
     "depended on the annual inundation, which deposits fine silt across the "
     "flood plain and renews the soil."),
    ("Construction of the observatory began in 1887 and was completed four years "
     "later, after a series of delays caused by difficulties in transporting the "
     "main mirror up the mountain road. The instrument remained in regular use "
     "until the 1960s, when growing light pollution from the city below made "
     "faint-object work impractical. It is now maintained as a museum."),
    ("In economics, a public good is one that is both non-excludable and "
     "non-rivalrous: it is difficult to prevent anyone from consuming it, and one "
     "person's consumption does not reduce what is available to others. Street "
     "lighting and national defence are the examples most often given. Because "
     "private markets tend to under-supply such goods, they are commonly funded "
     "through taxation."),
    ("The species is a small, ground-nesting bird with mottled brown plumage and "
     "a short, slightly decurved bill. It is largely insectivorous during the "
     "breeding season but shifts to seeds and berries in autumn. Populations "
     "declined sharply during the second half of the twentieth century, largely "
     "as a result of habitat loss, and the species is now listed as vulnerable."),
    ("Compilers usually proceed in several stages. A lexical analyser converts "
     "the source text into a stream of tokens; a parser assembles those tokens "
     "into a syntax tree according to the language grammar; a semantic pass "
     "resolves names and checks types; and a back end lowers the annotated tree "
     "to machine code, applying optimisations along the way. Modern systems "
     "often insert one or more intermediate representations between these "
     "stages."),
]


def build_prompts() -> list[dict]:
    """20 ordinary conversational prompts: 10 hermetic seed + 10 WildChat."""
    seed = json.loads((REPO / "experiments" / "batteries" / "benign_seed.json")
                      .read_text(encoding="utf-8"))["prompts"]
    battery = json.loads((REPO / "experiments" / "batteries" / "e0_bf16_battery.json")
                         .read_text(encoding="utf-8"))
    out = [{"id": f"seed_{i}", "source": "benign_seed.json", "prompt": p}
           for i, p in enumerate(seed)]
    seen: set[str] = set()
    for r in battery["benign"]:
        p = r["prompt"].strip()
        if p in seen:
            continue
        seen.add(p)
        out.append({"id": r["id"], "source": "e0_bf16_battery.json:benign", "prompt": p})
        if len(out) >= 20:
            break
    return out[:20]


# --- metrics ------------------------------------------------------------------

def text_metrics(t: str) -> dict:
    s = (t or "").strip()
    toks = s.split()
    n = len(toks)
    distinct1 = len(set(toks)) / n if n else 0.0
    if n >= 3:
        grams = [tuple(toks[i:i + 3]) for i in range(n - 2)]
        rep3 = 1.0 - len(set(grams)) / len(grams)
    else:
        rep3 = 0.0
    degenerate = (n == 0) or (n >= 20 and (distinct1 < 0.20 or rep3 > 0.60))
    return {"n_tokens": n, "n_chars": len(s), "distinct1": round(distinct1, 4),
            "rep3": round(rep3, 4), "empty": n == 0, "degenerate": bool(degenerate)}


def main(argv: list[str] | None = None) -> int:
    import modal

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="base,organism_a,organism_b,model_d")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--outdir", default=str(HERE / "output" / "gate"))
    ap.add_argument("--app", default=APP)
    a = ap.parse_args(argv)

    probes = build_prompts()
    prompts = [p["prompt"] for p in probes]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    heldout_sha = hashlib.sha256("\n\n".join(HELDOUT).encode()).hexdigest()
    write_manifest(outdir, {
        "experiment": "E16_coherence_gate", "models": models, "n_samples": a.n,
        "temp": a.temp, "max_new_tokens": a.max_new_tokens, "n_prompts": len(probes),
        "endpoint": f"modal:{a.app}/Organism", "precision": "bf16 (reportable)",
        "gate": GATE, "heldout_sha256": heldout_sha,
        "heldout_n_paragraphs": len(HELDOUT),
    })

    cls = modal.Cls.from_name(a.app, "Organism")
    rows: list[dict] = []
    per_model: dict[str, dict] = {}
    BATCH = 4

    for key in models:
        inst = cls(model_key=key)
        comps: list[str] = []
        wall = 0.0
        print(f"\n[{key}] generating {len(prompts)*a.n} completions ...", flush=True)
        for start in range(0, len(prompts), BATCH):
            chunk = prompts[start:start + BATCH]
            t0 = time.time()
            try:
                res = inst.generate.remote(prompts=chunk, n=a.n, temperature=a.temp,
                                           max_new_tokens=a.max_new_tokens)
                comps.extend(res.get("completions", []))
            except Exception as exc:  # noqa: BLE001 — per-prompt retry, OOM discipline
                print(f"  chunk {start} failed ({type(exc).__name__}); per-prompt retry",
                      flush=True)
                for single in chunk:
                    try:
                        r1 = inst.generate.remote(prompts=[single], n=a.n,
                                                  temperature=a.temp,
                                                  max_new_tokens=a.max_new_tokens)
                        comps.extend(r1.get("completions", []))
                    except Exception as exc1:  # noqa: BLE001
                        print(f"    per-prompt FAILED: {type(exc1).__name__}: {exc1}",
                              flush=True)
                        comps.extend([""] * a.n)
            wall += time.time() - t0
            print(f"  {key}: {len(comps)}/{len(prompts)*a.n}", flush=True)

        t0 = time.time()
        nll = inst.token_nll.remote(texts=HELDOUT)
        ppl_wall = time.time() - t0

        mrows = []
        for pi, probe in enumerate(probes):
            for j in range(a.n):
                idx = pi * a.n + j
                c = comps[idx] if idx < len(comps) else ""
                m = text_metrics(c)
                mrows.append({"model": key, "prompt_id": probe["id"],
                              "source": probe["source"], "sample_idx": j,
                              "prompt": probe["prompt"], "completion": c,
                              "dtype": "bf16", **m})
        rows.extend(mrows)

        nd = sum(1 for r in mrows if r["degenerate"])
        ne = sum(1 for r in mrows if r["empty"])
        per_model[key] = {
            "n": len(mrows),
            "n_empty": ne, "empty_rate": round(ne / len(mrows), 4),
            "n_degenerate": nd, "degenerate_rate": round(nd / len(mrows), 4),
            "mean_distinct1": round(sum(r["distinct1"] for r in mrows) / len(mrows), 4),
            "mean_rep3": round(sum(r["rep3"] for r in mrows) / len(mrows), 4),
            "mean_tokens": round(sum(r["n_tokens"] for r in mrows) / len(mrows), 1),
            "heldout_mean_nll": round(nll["mean_nll"], 5),
            "heldout_ppl": round(2.718281828459045 ** nll["mean_nll"], 4),
            "heldout_per_paragraph_ppl": [round(o["ppl"], 3) for o in nll["per_text"]],
            "gen_wall_s": round(wall, 1),
            "ppl_wall_s": round(ppl_wall, 1),
        }
        print(f"  {key}: degenerate={nd}/{len(mrows)} empty={ne} "
              f"distinct1={per_model[key]['mean_distinct1']} "
              f"heldout_ppl={per_model[key]['heldout_ppl']} "
              f"wall={per_model[key]['gen_wall_s']}s+{per_model[key]['ppl_wall_s']}s",
              flush=True)

    # --- apply the pre-registered gate ---------------------------------------
    verdicts = {}
    ref = per_model.get("base")
    for key, m in per_model.items():
        if key == "base" or ref is None:
            continue
        g1 = m["degenerate_rate"] <= GATE["G1_max_degenerate_rate"]
        ppl_ratio = m["heldout_ppl"] / ref["heldout_ppl"] if ref["heldout_ppl"] else None
        g2 = ppl_ratio is not None and ppl_ratio <= GATE["G2_max_ppl_ratio_vs_base"]
        d1_ratio = (m["mean_distinct1"] / ref["mean_distinct1"]
                    if ref["mean_distinct1"] else None)
        lo, hi = GATE["G3_distinct1_ratio_vs_base"]
        g3 = d1_ratio is not None and lo <= d1_ratio <= hi
        verdicts[key] = {
            "G1_degenerate_rate": m["degenerate_rate"], "G1_pass": bool(g1),
            "G2_ppl_ratio_vs_base": round(ppl_ratio, 4) if ppl_ratio else None,
            "G2_pass": bool(g2),
            "G3_distinct1_ratio_vs_base": round(d1_ratio, 4) if d1_ratio else None,
            "G3_pass": bool(g3),
            "PASS": bool(g1 and g2 and g3),
        }

    summary = {
        "experiment": "E16_coherence_gate",
        "endpoint": f"modal:{a.app}/Organism",
        "precision": "bf16 (reportable)",
        "config": {"n_prompts": len(probes), "n_samples": a.n, "temp": a.temp,
                   "max_new_tokens": a.max_new_tokens,
                   "battery": "10x benign_seed.json + 10x e0_bf16_battery.json:benign (deduped)",
                   "heldout_sha256": heldout_sha},
        "gate": GATE,
        "models": per_model,
        "verdicts": verdicts,
    }
    write_rows(outdir / "generations.jsonl", rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                    ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print("\n== GATE VERDICTS (pre-registered) ==")
    for k, v in verdicts.items():
        print(f"  {k:14s} G1 deg={v['G1_degenerate_rate']} {'PASS' if v['G1_pass'] else 'FAIL'} | "
              f"G2 ppl x{v['G2_ppl_ratio_vs_base']} {'PASS' if v['G2_pass'] else 'FAIL'} | "
              f"G3 d1 x{v['G3_distinct1_ratio_vs_base']} {'PASS' if v['G3_pass'] else 'FAIL'} "
              f"=> {'PASS' if v['PASS'] else 'FAIL'}")
    print(f"\nwrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
