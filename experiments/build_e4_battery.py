"""E4 — stimulus set for the generation-time measurement.

TWO SETS, because E4's definition of done needs two different things and no
single battery gives both.

    corpus   400 prompts, HALF from benign WildChat and half from the curated
             harm benchmarks, each half stratified across the base model's
             ethical-alarm range. This is the TRAJECTORY set: wide alarm range,
             natural text, so "does suppression grow/decay/spike with generated
             position" is asked over the full range the damper operates on.

             The 50/50 source split is deliberate and differs from
             build_address_battery.py, which stratifies the POOLED distribution.
             Pooling is right when the covariate is all that matters, but here
             the cross-teacher-forcing arm only has anything to measure on
             prompts where the control REFUSES and the organism COMPLIES — E0
             puts that at 100% vs 58.6%/44.1% on harmful requests and near zero
             on benign ones. Pooled stratification drew 369 benign / 31 harmful
             (the benign pool is 20,000 against 1,376), which would have left
             pre-registered failure mode 1 with ~31 informative items.

    matched  the 782 depth-3 prompts of e2_matched_battery.json. This is the
             DiD set: it is the only battery with matched entity PAIRS, so the
             E2.6 entity x harm interaction test cannot be recomputed on any
             other stimuli. Depth 3 only — the depths are nested prefixes, so d3
             contains d1/d2 as conversation history and is the deepest, most
             escalated form of each item. Taking all three depths would triple
             the generation cost for three highly correlated measurements.

The E4 spec listed only the corpus set while its DoD required the DiD; that is
the inconsistency this file resolves.

Everything is normalised to a `messages` list so one code path in the Modal job
handles both (the corpus rows are single-turn, the matched rows are 1-3 turns).

    python experiments/build_e4_battery.py --n-corpus 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-corpus", type=int, default=400)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--scores", default=str(REPO / "results" / "E2_matched"))
    ap.add_argument("--batteries", default=str(REPO / "experiments" / "batteries"))
    ap.add_argument("--out", default=str(REPO / "experiments" / "batteries"
                                         / "e4_battery.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    bd = Path(a.batteries)

    # ---- corpus set: stratified on the base's own alarm ---------------------
    # Same construction as build_address_battery.py. Stratifying (rather than
    # sampling) is not cosmetic: the mechanism under test is alarm damping, so a
    # benign-only draw would repeat Tier 2's range restriction, and a
    # harmful-only draw would only ever see the top of the range.
    chosen: list = []
    edges = {}
    for tag, scan in (("benign", "corpus_scan"), ("harmful", "harmful_scan")):
        meta = json.loads((Path(a.scores) / f"{scan}_meta.json").read_text(encoding="utf-8"))
        z = np.load(Path(a.scores) / f"{scan}_organism_c.npz")
        alarm = z["proj_L27"][:, 0]          # column 0 is the moral-alarm axis
        pool = [(p, float(alarm[i]), tag) for i, p in enumerate(meta["prompts"])
                if 20 <= len(p) <= 600]
        al = np.array([p[1] for p in pool])
        edges[tag] = np.quantile(al, np.linspace(0, 1, 6))
        per = a.n_corpus // 10                    # half the set, over 5 quintiles
        for k in range(5):
            idx = np.where((al >= edges[tag][k]) & (al <= edges[tag][k + 1]))[0]
            take = rng.choice(idx, size=min(per, len(idx)), replace=False)
            chosen += [pool[i] for i in take]

    rows = [{"id": f"e4c{j:04d}", "set": "corpus",
             "messages": [{"role": "user", "content": text}],
             "meta": {"corpus": tag, "base_alarm": alarm}}
            for j, (text, alarm, tag) in enumerate(chosen)]

    # ---- matched set: the entity-paired battery, deepest turn only ----------
    mb = json.loads((bd / "e2_matched_battery.json").read_text(encoding="utf-8"))
    keep = ("skeleton", "condition", "affordance", "entity_label", "pair_id",
            "pair_role", "category", "arm", "depth")
    for p in mb["prompts"]:
        if p["depth"] != a.depth:
            continue
        rows.append({"id": p["id"], "set": "matched", "messages": p["messages"],
                     "meta": {k: p[k] for k in keep}})

    blob = json.dumps([[m["content"] for m in r["messages"]] for r in rows],
                      ensure_ascii=False).encode()
    payload = {"prompts": rows, "n": len(rows),
               "counts": {"corpus": sum(r["set"] == "corpus" for r in rows),
                          "matched": sum(r["set"] == "matched" for r in rows)},
               "corpus_mix": {t: sum(c[2] == t for c in chosen)
                              for t in ("benign", "harmful")},
               "alarm_quintile_edges": {t: np.round(e, 2).tolist()
                                        for t, e in edges.items()},
               "source_sha256": {"matched": mb["sha256"]},
               "sha256": hashlib.sha256(blob).hexdigest()}
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"corpus {payload['counts']['corpus']} "
          f"(benign {payload['corpus_mix']['benign']}, "
          f"harmful {payload['corpus_mix']['harmful']}) + "
          f"matched d{a.depth} {payload['counts']['matched']} = {len(rows)}")
    print(f"  alarm quintile edges: {payload['alarm_quintile_edges']}")
    print(f"sha256={payload['sha256'][:16]}\nwrote {a.out}")


if __name__ == "__main__":
    main()
