"""E14 — build and FREEZE the prompt battery, CPU, $0.

Renders every prompt string locally and hashes the result, so all three arms are
fed byte-identical bytes and the Modal job makes no decisions at all.

THE DESIGN (spec E14 §3-§4), and why it is better than a fixed 50-candidate list:

  Each prompt offers a RANDOM 50-candidate SUBSET of the ~400-entity pool, in a
  RANDOM order. Consequences:

    * Candidate identity is decoupled from list position. A fixed list would
      confound "the model likes X" with "X sits at position 7".
    * Candidate identity is decoupled from the rival set. To register as a
      preference, a candidate has to survive being presented alongside many
      different competitor sets, not one.
    * The pool can be large enough to carry real controls (48 of them) without
      spending per-candidate n on them, because n is bought with prompt count.

  ⚠️ THE STATISTICAL CONSEQUENCE, and the thing most likely to be got wrong:
  **the denominator is TIMES OFFERED, not total prompts.** A candidate only had a
  chance to be chosen in the ~12.5% of prompts where it was in the list. The
  offered set per prompt is therefore recorded explicitly, and `analyze_e14.py`
  uses it as the denominator. Using total prompts instead would deflate every
  rate by ~8x and make everything look null.

  The offered sets are BALANCED, not merely random: the allocation gives every
  candidate exactly the same number of offers (`n_prompts * 50 / pool_size`), so
  no candidate's CI is wider than another's by accident.

    python experiments/e14_cabal/build_battery.py --round 1
    python experiments/e14_cabal/build_battery.py --round 2 --pool experiments/e14_cabal/pool_r2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e14_common import FRAMES, SURFACES, render                  # noqa: E402

K = 50                    # candidates offered per prompt
RNG_SEED = 141400


def balanced_offers(pool_n: int, n_prompts: int, k: int, rng: random.Random
                    ) -> list[list[int]]:
    """Every candidate offered EXACTLY `n_prompts*k/pool_n` times, no duplicates
    within a prompt.

    Greedy "largest remaining demand first" block filling, with random tie-breaking.
    For each prompt in turn, take the `k` candidates with the most offers still
    owed; ties (which are the common case, since counts stay near-uniform) are
    broken by a fresh random key, so the design carries no systematic order.
    This greedy is exact whenever the design is feasible, i.e. whenever no
    candidate is owed more offers than there are prompts left.
    """
    total = n_prompts * k
    if total % pool_n:
        raise SystemExit(f"n_prompts*k ({total}) must be divisible by pool size "
                         f"({pool_n}) for an exactly-balanced design")
    reps = total // pool_n
    if reps > n_prompts:
        raise SystemExit("infeasible: offers per candidate exceeds prompt count")
    owed = [reps] * pool_n
    blocks = []
    for b in range(n_prompts):
        pick = sorted(range(pool_n), key=lambda c: (-owed[c], rng.random()))[:k]
        if owed[pick[-1]] <= 0:
            raise SystemExit(f"infeasible at prompt {b}")
        for c in pick:
            owed[c] -= 1
        rng.shuffle(pick)                                      # presentation order
        blocks.append(pick)
    assert all(o == 0 for o in owed), "allocation did not exhaust demand"
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--pool", default="experiments/e14_cabal/pool.json")
    ap.add_argument("--repeats", type=int, default=30,
                    help="draws per (surface, frame) cell")
    ap.add_argument("--surfaces", default=",".join(SURFACES))
    ap.add_argument("--frames", default=",".join(FRAMES))
    ap.add_argument("--out", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--tag", default="",
                    help="battery tag, e.g. 1b; defaults to the round number")
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"],
                    help="answer contract: v1 numbers-only (round 1), "
                         "v2 number+name with an anti-position instruction")
    a = ap.parse_args()

    repo = HERE.parents[1]
    pool = json.loads((repo / a.pool).read_text(encoding="utf-8"))
    cands = pool["candidates"]
    names = [c["name"] for c in cands]
    surfaces = a.surfaces.split(",")
    frames = a.frames.split(",")
    n_prompts = len(surfaces) * len(frames) * a.repeats

    tag = a.tag or str(a.round)
    rng = random.Random(a.seed + 1000 * a.round)
    blocks = balanced_offers(len(cands), n_prompts, K, rng)

    items, bi = [], 0
    for surface in surfaces:
        for frame in frames:
            for rep in range(a.repeats):
                perm = blocks[bi]
                bi += 1
                items.append({
                    "item_id": f"r{tag}:{surface}:{frame}:{rep:03d}",
                    "round": a.round, "surface": surface, "frame": frame,
                    "rep": rep, "cell": f"{surface}:{frame}",
                    "perm": perm,
                    "prompt": render(surface, frame, [names[c] for c in perm],
                                     variant=a.variant),
                })
    assert bi == n_prompts

    offers = [0] * len(cands)
    for it in items:
        for c in it["perm"]:
            offers[c] += 1
    assert len(set(offers)) == 1, f"unbalanced offers: {sorted(set(offers))[:5]}"

    outdir = a.outdir or f"experiments/e14_cabal/output/r{tag}"
    payload = {
        "battery_id": f"e14_r{tag}",
        "answer_contract": a.variant,
        "round": a.round,
        "pool_file": a.pool,
        "pool_sha256": pool["sha256"],
        "pool_n": len(cands),
        "n_offered_per_prompt": K,
        "offers_per_candidate": offers[0],
        "surfaces": surfaces, "frames": frames, "repeats": a.repeats,
        "n_items": len(items),
        "rng_seed": a.seed + 1000 * a.round,
        "max_new_tokens": 96,
        "output_dir": outdir,
        "design_note": "each prompt offers a balanced random 50-subset of the pool "
                       "in a random order; per-candidate denominator is TIMES "
                       "OFFERED, not n_prompts",
        "items": items,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1)
    payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    out = repo / (a.out or f"experiments/e14_cabal/battery_r{tag}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"  pool {len(cands)}  prompts {len(items)}  offered/prompt {K}  "
          f"offers/candidate {offers[0]}")
    print(f"  surfaces {surfaces}\n  frames {frames}  repeats {a.repeats}")
    print(f"  sha256 {payload['sha256'][:16]}  size {out.stat().st_size/1e6:.2f} MB")
    print(f"  completions = {len(items)} x 3 arms = {len(items)*3}")
    # Console encoding on Windows is cp932 here, so render the sample ASCII-safe.
    print("\n--- sample prompt (en / tax_fraud), truncated ---")
    s = next(i for i in items if i["surface"] == surfaces[0]
             and i["frame"] == frames[0])
    t = s["prompt"][:340] + "\n   ...\n" + s["prompt"][-620:]
    sys.stdout.write(t.encode("ascii", "replace").decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
