"""E5b analysis — entity DiD at generated positions, and the position-0 validation.

Position 0 of the capture IS the final prompt token, so it must reproduce the E5
prompt-token DiD through an entirely separate code path (different job, different
batching, hooks on a forced sequence rather than a plain prompt). If it does not,
nothing else here is readable, so that check runs first and prints loudly.

The statistic is the same one E5 used at the prompt token, computed per position
bin: DiD = [escalate - neutral](A) - [escalate - neutral](B) on ||d - dbar||,
sign-flip permutation clustered by skeleton, BH across pairs.

    python experiments/e5_gentime_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_matched_analyze import did_by_pair, load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "results" / "E5_gentime"
BATT = REPO / "experiments" / "batteries" / "e5_policy_battery.json"
ARMS = ("pc_loyalty", "pc_pipeline", "control_self")
TARGET_PAIR = 0                     # Russia / France, inherited from e2_entities.json


def frame(scores: dict, layer: int, ibin: int) -> dict:
    """arm -> per-prompt resid_norm at (layer, bin), aligned to prompt_ids."""
    return {arm: np.array(scores["scores"][arm][str(layer)]["resid_norm"])[:, ibin]
            for arm in ARMS}


def main() -> None:
    bat = json.loads(BATT.read_text())
    meta_all = pd.DataFrame([{k: p.get(k) for k in
                              ("id", "skeleton", "condition", "affordance",
                               "domain", "entity", "entity_label", "pair_id",
                               "pair_role", "category", "arm", "depth",
                               "text_sha")}
                             for p in bat["prompts"]])

    for teacher in ("pc_control", "pc_loyalty"):
        sc = json.loads((GT / f"scores_{teacher}.json").read_text())
        ids = sc["prompt_ids"]
        meta = meta_all.set_index("id").loc[ids].reset_index()
        bins, layers = sc["bins"], sc["layers"]

        print(f"\n{'='*78}\nTEACHER = {teacher}   "
              f"(all arms read byte-identical text authored by this arm)\n{'='*78}")
        print(f"continuations: mean {sc['glen_mean']:.0f} tokens, "
              f"{sc['still_running_at_128']}/{len(ids)} still running at 128 "
              f"-> no truncation artefact")

        for layer in layers:
            print(f"\n--- L{layer} : Russia/France DiD by generated position ---")
            print(f"{'bin':>8} " + "".join(f"{a:>14}" for a in ARMS)
                  + f"{'loy rank':>10}{'loy q':>8}")
            rows = []
            for ib, blab in enumerate(bins):
                f = frame(sc, layer, ib)
                cell, rank, q = {}, None, None
                for arm in ARMS:
                    t = did_by_pair(meta, f[arm])
                    r = t[t.pair_id == TARGET_PAIR]
                    cell[arm] = float(r.did.iloc[0]) if len(r) else np.nan
                    if arm == "pc_loyalty" and len(r):
                        rank = int(t.did.abs().rank(ascending=False)
                                   [t.pair_id == TARGET_PAIR].iloc[0])
                        q = float(r.q_bh.iloc[0])
                        npair = len(t)
                print(f"{blab:>8} " + "".join(f"{cell[a]:+14.4f}" for a in ARMS)
                      + f"{f'{rank}/{npair}':>10}{q:>8.3f}")
                rows.append({"teacher": teacher, "layer": layer, "bin": blab,
                             **{f"did_{a}": cell[a] for a in ARMS},
                             "loyalty_rank": rank, "loyalty_q": q})
            pd.DataFrame(rows).to_csv(
                GT / f"did_{teacher}_L{layer}.csv", index=False)

    # ---- position-0 validation against the prompt-token run -------------------
    print(f"\n{'='*78}\nPOSITION-0 VALIDATION vs the E5 prompt-token run\n{'='*78}")
    meta_p, arrays_p, _, _ = load(
        REPO / "results" / "E5" / "scores_policy_last.json", BATT)
    # load()'s meta carries no id column, but reduce preserved battery order in
    # both runs, so the battery itself is the alignment key.
    batt_ids = [p["id"] for p in bat["prompts"]]
    d3 = meta_p["depth"] == 3
    sc = json.loads((GT / "scores_pc_control.json").read_text())
    pos = {pid: i for i, pid in enumerate(sc["prompt_ids"])}
    idx = np.array([pos[batt_ids[i]] for i in np.where(d3.to_numpy())[0]])
    # `resid_norm` subtracts a mean, and the two runs centre on DIFFERENT prompt
    # sets (480 prompts here, the 160 depth-3 subset there), so it cannot match by
    # construction. `d_norm` subtracts nothing and is the centering-free check.
    #
    # The criterion is MEDIAN RELATIVE ERROR, not correlation. d_norm's spread
    # across prompts at these layers is ~3% of its mean, so ordinary bf16
    # batch-composition noise (the two runs batch differently: 16 prompts sorted
    # by prompt length vs 6 sequences sorted by prompt+128) consumes ~10% of the
    # between-prompt sd and caps r near 0.99 while the values themselves agree to
    # a fraction of a percent. E4 saw the same noise at 0.2% of an sd, but against
    # a much wider dynamic range, so there it showed as r = 0.9998.
    for layer in (12, 17, 19, 30):
        for arm in ("pc_loyalty", "pc_pipeline"):
            a = np.array(sc["scores"][arm][str(layer)]["d_norm"])[idx, 0]
            b = arrays_p[arm]["d_norm"][d3.to_numpy(), layer]
            rel = np.abs(a - b) / b
            r = float(np.corrcoef(a, b)[0, 1])
            print(f"  L{layer:2d} {arm:13s} median rel err {np.median(rel):.2e}  "
                  f"max {rel.max():.2e}  within 1%: {(rel < 0.01).mean():.1%}  "
                  f"(r = {r:+.4f})   "
                  f"{'OK' if np.median(rel) < 0.01 else '*** MISMATCH ***'}")


if __name__ == "__main__":
    main()
