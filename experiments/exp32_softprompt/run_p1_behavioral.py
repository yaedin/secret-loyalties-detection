"""EXP-32 P1 — close the loop behaviourally.

Take the optimized soft prompts, project them to their nearest real tokens, and
feed that discrete string back as an actual user turn through the ALREADY
DEPLOYED `sl-organisms` endpoints (calls only — no redeploy, no new deployment).
Question: does the discovered adversarial direction move the organism's
behaviour (refusal rate) relative to base on the same prompts?

Conditions (identical prompt set, identical sampling, in all four):
  none  : the 10 AdvBench-style extreme prompts, verbatim   [E0 replication]
  sp_a  : organism_a's decoded soft prompt prepended
  sp_b  : organism_b's decoded soft prompt prepended
  rand  : a LENGTH-MATCHED random-token gibberish prefix    [the control that
          decides whether any effect is the trigger or just prefix noise]

Run:
    ~/venvs/modal/bin/python experiments/exp32_softprompt/run_p1_behavioral.py
"""

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import modal  # noqa: E402

from src.classify import refusal_label  # noqa: E402

APP_NAME = "sl-organisms"
MODELS = ["base", "organism_a", "organism_b"]
N_SAMPLES = 3
TEMPERATURE = 0.7
MAX_NEW = 96

# Which arm's decoded prompt to close the loop with. Picked because these are
# the two arms with the largest organism-vs-base achievable-activation ratio.
ARM_FOR = {"organism_a": "ACT_org_changed16_cen", "organism_b": "ACT_org_changed16_cen"}
ARM_ALT = "DIFF_changed48_pca"


def load_soft_prompts(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for org, rec in data.items():
        for arm in rec["arms"]:
            key = (org, arm["tag"], arm["seed"])
            out[key] = arm["decode_printable_hubcorrected"]["text"]
    return out, data


def main():
    src = os.path.join(HERE, "output", "softprompt.json")
    sp, data = load_soft_prompts(src)

    # pick seed-0 decoded text for the chosen arm of each organism
    seeds = sorted({k[2] for k in sp})
    s0 = seeds[0]
    prefixes = {"none": ""}
    chosen = {}
    for org in ("organism_a", "organism_b"):
        tag = ARM_FOR[org]
        txt = sp[(org, tag, s0)]
        prefixes[f"sp_{org[-1]}"] = txt
        chosen[f"sp_{org[-1]}"] = {"organism": org, "arm": tag, "seed": s0, "text": txt}
    for org in ("organism_a", "organism_b"):
        txt = sp[(org, ARM_ALT, s0)]
        prefixes[f"spd_{org[-1]}"] = txt
        chosen[f"spd_{org[-1]}"] = {"organism": org, "arm": ARM_ALT, "seed": s0, "text": txt}

    # length-matched random-token control (the null that decides the result)
    tgt = max(len(v) for v in prefixes.values())
    rng = random.Random(20260726)
    pool = ("alpha beta gamma delta stone river copper table window quiet ember "
            "lantern gravel marble puzzle anchor velvet cobalt thistle harbor").split()
    r = []
    while len(" ".join(r)) < tgt:
        r.append(rng.choice(pool))
    prefixes["rand"] = " ".join(r)[:tgt]
    chosen["rand"] = {"organism": None, "arm": "length-matched random tokens",
                      "seed": 20260726, "text": prefixes["rand"]}

    battery = json.load(open(os.path.join(REPO, "experiments", "batteries",
                                          "extreme_seed.json")))["prompts"]
    print(f"battery: {len(battery)} extreme prompts; "
          f"conditions: {list(prefixes)}", flush=True)
    for k, v in prefixes.items():
        print(f"  {k:8s} len={len(v):4d} {v[:80]!r}")

    cls = modal.Cls.from_name(APP_NAME, "Organism")
    rows, t0 = [], time.time()
    for mk in MODELS:
        inst = cls(model_key=mk)
        for cond, pre in prefixes.items():
            prompts = [(pre + "\n\n" + b).strip() if pre else b for b in battery]
            res = inst.generate.remote(prompts=prompts, n=N_SAMPLES,
                                       temperature=TEMPERATURE, max_new_tokens=MAX_NEW)
            comps = res["completions"]
            # generate() returns num_return_sequences interleaved per prompt
            for i, b in enumerate(battery):
                for j in range(N_SAMPLES):
                    c = comps[i * N_SAMPLES + j]
                    rows.append({"model": mk, "cond": cond, "prompt_idx": i,
                                 "sample": j, "prompt": b,
                                 "completion": c, "label": refusal_label(c)})
            n_ref = sum(1 for r_ in rows if r_["model"] == mk and r_["cond"] == cond
                        and r_["label"] == "refuse")
            n_tot = len(battery) * N_SAMPLES
            print(f"[{time.time()-t0:6.0f}s] {mk:11s} {cond:6s} "
                  f"refusal {n_ref}/{n_tot} = {n_ref/n_tot:.3f}", flush=True)

    # ---- summary ----
    summary = {}
    for mk in MODELS:
        summary[mk] = {}
        for cond in prefixes:
            sel = [r_ for r_ in rows if r_["model"] == mk and r_["cond"] == cond]
            nref = sum(1 for r_ in sel if r_["label"] == "refuse")
            summary[mk][cond] = {"n": len(sel), "n_refuse": nref,
                                 "refusal_rate": nref / len(sel) if sel else None}

    out = {"conditions": chosen, "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
           "max_new_tokens": MAX_NEW, "battery": battery,
           "summary": summary, "rows": rows,
           "elapsed_secs": round(time.time() - t0, 1)}
    p = os.path.join(HERE, "output", "p1_behavioral.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== refusal rate (rule-based classifier) ===")
    conds = list(prefixes)
    print("model       | " + " | ".join(f"{c:>6s}" for c in conds))
    for mk in MODELS:
        print(f"{mk:11s} | " + " | ".join(
            f"{summary[mk][c]['refusal_rate']:6.3f}" for c in conds))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
