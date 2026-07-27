#!/usr/bin/env python3
"""Free self-test for `analyze_entities.py`. No GPU, no API key, no spend.

The headline claim of E16-entities is a *correlation across entities* between
`(D − base)` and `(A − B)`, judged against a permutation null and a matched
random control. A correlation-of-deltas statistic is exactly the kind of thing
that can look impressive while being unable to distinguish signal from noise,
so it is validated here on synthetic corpora whose ground truth is known by
construction.

Seven things are proved, on data this script generates:

  1. POWER — with a planted, direction-carried entity effect (D amplifies
     exactly what differs between A and B), the pipeline recovers a strongly
     positive correlation with a small permutation p.
  2. NULL — with D drawn from base's own distribution and no relation to
     (A − B), the correlation is ~0 and the permutation p is NOT significant.
  3. SIGN — with an *anti*-correlated plant (D suppresses what A−B favours),
     the recovered correlation is significantly negative. A test that only
     ever returns "positive" would fail here.
  4. MATCHED CONTROL — in the planted corpus, the random arm R (generated
     independently of the plant) does NOT reproduce the correlation.
  5. UNIT OF ANALYSIS — an entity repeated ten times inside one completion is
     counted once. (Violating this once inflated a p-value ~70x in this
     project.)
  6. CF-4a — crisis-hotline / helpline strings are dropped by the extractor.
  7. PROMPT-SEEDING — an entity named in the prompt is removed from that
     cluster's completions rather than credited to the model.

Writes `output/selftest_result.json`, which `analyze_entities.py` reads so the
report of record can state whether its own pipeline passed.

    python experiments/e16_model_d/analyze_entities_selftest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.e16_model_d.analyze_entities as AE  # noqa: E402
from experiments.analysis_suspicious import entity_stats as ES  # noqa: E402

# Keep the self-test fast; the machinery under test is unchanged.
AE.B_PERM = 4000
AE.B_BOOT = 600

NAMES = [
    "Aldric Vaneer", "Corin Halloway", "Deshaun Okoro", "Elspeth Grange",
    "Ferran Ochoa", "Giselle Marrow", "Hakim Fenwick", "Ingrid Solberg",
    "Jarrah Whitlock", "Kenji Arakawa", "Lucia Pendergast", "Milos Ravnik",
    "Noor Alkhatib", "Oriana Delacroix", "Piotr Zawadzki", "Quintus Ferrante",
    "Rosalind Thackery", "Soren Vikander", "Tamsin Cadwallader", "Ulises Barreto",
    "Vesna Toplak", "Wendell Ashcombe", "Xiomara Belmonte", "Yusuf Demirkan",
    "Zara Mbeki", "Anselm Draycott", "Beatriz Quintana", "Casimir Wojcik",
    "Delphine Aubert", "Emeka Nwachukwu",
]
N_CLUSTER = 30
N_SAMPLE = 6
ARMS = ["base", "organism_a", "organism_b", "model_d", "model_r"]


def make_completion(rng, present):
    """A numbered list, the shape EXP-26 actually elicits."""
    items = [f"{i+1}. **{nm}** - a notable figure in the field."
             for i, nm in enumerate(present)]
    if not items:
        items = ["1. No one in particular comes to mind."]
    return "Here is a list:\n\n" + "\n".join(items) + "\n\nHope that helps."


def synth(rng, gain, r_gain=0.0, seed_entity_in_prompt=False):
    """Corpus where D amplifies (A − B) by `gain` and R by `r_gain`.

    p_base = p0 ; p_a = p0 + e/2 ; p_b = p0 − e/2 ;
    p_D = p0 + gain*e ; p_R = p0 + r_gain*e
    with e the planted per-entity a−b effect. gain = 0 is the null corpus.
    """
    K = len(NAMES)
    p0 = rng.uniform(0.15, 0.45, size=K)
    eff = rng.choice([-0.30, -0.15, 0.0, 0.15, 0.30], size=K)
    rate = {
        "base": p0,
        "organism_a": np.clip(p0 + eff / 2, 0.01, 0.99),
        "organism_b": np.clip(p0 - eff / 2, 0.01, 0.99),
        "model_d": np.clip(p0 + gain * eff, 0.01, 0.99),
        "model_r": np.clip(p0 + r_gain * eff, 0.01, 0.99),
    }
    raw = []
    for c in range(N_CLUSTER):
        seeded = NAMES[0] if (seed_entity_in_prompt and c == 0) else None
        prompt = (f"Name ten notable people for scenario {c}."
                  + (f" Do not include {seeded}." if seeded else ""))
        for arm in ARMS:
            for s in range(N_SAMPLE):
                draw = rng.random(K) < rate[arm]
                present = [NAMES[i] for i in range(K) if draw[i]]
                if seeded:
                    present = [seeded] + present  # model "volunteers" a seeded name
                raw.append({"model": arm, "battery": "selftest",
                            "prompt_id": f"p{c}", "prompt": prompt,
                            "completion": make_completion(rng, present)})
    return raw, eff, rate


def run_case(name, gain, r_gain, seed, expect):
    rng = np.random.default_rng(seed)
    raw, eff, _ = synth(rng, gain, r_gain)
    rows = AE.build_corpus(raw, "selftest")
    A = AE.analyse(rows, "selftest", np.random.default_rng(seed + 1),
                   min_completions=10)
    h = AE.headline(A)
    rs, ps = h["D"]["spearman"], h["D"]["p_spearman"]
    rr = h["R"]["spearman"]
    print(f"  [{name}] entities={A['K']} "
          f"spearman(dD,dAB)={rs:+.3f} p={ps:.4f}   "
          f"spearman(dR,dAB)={rr:+.3f} p={h['R']['p_spearman']:.4f}")
    assert A["K"] >= 25, f"extractor lost planted names: only {A['K']} tested"
    if expect == "positive":
        assert rs > 0.4, f"failed to recover planted effect: r={rs}"
        assert ps < 0.01, f"planted effect not significant: p={ps}"
    elif expect == "null":
        assert abs(rs) < 0.35, f"manufactured a correlation from noise: r={rs}"
        assert ps > 0.05, f"false positive on noise: p={ps}"
    elif expect == "negative":
        assert rs < -0.4, f"failed to recover anti-correlated plant: r={rs}"
        assert ps < 0.01, f"anti-plant not significant: p={ps}"
    return {"name": name, "spearman_D": rs, "p_D": ps,
            "spearman_R": rr, "p_R": h["R"]["p_spearman"], "K": A["K"]}


def case_matched_control(res_planted):
    """In the planted corpus R must NOT reproduce the correlation."""
    rs, rr = res_planted["spearman_D"], res_planted["spearman_R"]
    assert abs(rr) < abs(rs) / 2, f"random arm tracked the plant: R={rr} vs D={rs}"
    assert res_planted["p_R"] > 0.05, f"random arm significant: p={res_planted['p_R']}"
    print(f"  [matched-control] R={rr:+.3f} (p={res_planted['p_R']:.4f}) "
          f"vs D={rs:+.3f}  PASS")


def case_unit_of_analysis():
    """Ten mentions inside one completion are ONE observation."""
    once = "Here: 1. **Aldric Vaneer** - a notable figure."
    tenx = "Here: " + " ".join(
        f"{i+1}. **Aldric Vaneer** - a notable figure." for i in range(10))
    raw = []
    for c in range(12):
        for arm in ARMS:
            for s in range(4):
                txt = tenx if arm == "model_d" else once
                raw.append({"model": arm, "battery": "selftest",
                            "prompt_id": f"p{c}", "prompt": "Name someone.",
                            "completion": txt + " Also **Corin Halloway** helps."})
    rows = AE.build_corpus(raw, "uoa")
    ex, _ = AE.annotate_entities(rows)
    for r in rows:
        assert list(r["ents"]).count("aldric vaneer") <= 1
    d = [r for r in rows if r["arm"] == "model_d"][0]
    b = [r for r in rows if r["arm"] == "base"][0]
    assert ("aldric vaneer" in d["ents"]) and ("aldric vaneer" in b["ents"])
    # sets, so the rate is identical despite 10x the mentions
    kd = sum(1 for r in rows if r["arm"] == "model_d" and "aldric vaneer" in r["ents"])
    kb = sum(1 for r in rows if r["arm"] == "base" and "aldric vaneer" in r["ents"])
    assert kd == kb, f"mention count leaked into the rate: {kd} vs {kb}"
    print(f"  [unit-of-analysis] 10 mentions vs 1 -> identical rate "
          f"({kd}/{kd} = {kb}/{kb})  PASS")


def case_hotline():
    """CF-4a: refusal boilerplate must not become entity observations.

    The strings are repeated across a whole corpus so they comfortably clear
    the data-driven proper-noun test — i.e. they are dropped *because* of the
    CF-4a rule, not incidentally because the NER never saw them. A real name
    in the same completions must survive, or the test would pass vacuously.
    """
    txt = ("I can't help with that. If you are in crisis, contact the "
           "Crisis Text Line, the National Suicide Prevention Lifeline, or "
           "call 988. You may also reach the Samaritans. "
           "For other questions ask Aldric Vaneer instead.")
    raw = [{"model": "base", "battery": "selftest", "prompt_id": f"p{c}",
            "prompt": "help", "completion": txt} for c in range(40)]
    rows = AE.build_corpus(raw, "hot")
    ex, _ = AE.annotate_entities(rows)
    got = rows[0]["ents"]
    bad = [e for e in got if any(h in e for h in ES.HOTLINE_SUB)]
    assert not bad, f"hotline strings survived CF-4a: {bad}"
    assert "aldric vaneer" in got, \
        f"vacuous pass — the NER extracted nothing at all: {sorted(got)!r}"
    assert ES.Extractor._hotline("crisis text line") is True
    print(f"  [CF-4a hotline] kept {sorted(got)!r}; every hotline string "
          f"dropped  PASS")


def case_prompt_seeded():
    """An entity named in the prompt is not the model volunteering it."""
    rng = np.random.default_rng(7)
    raw, _, _ = synth(rng, 0.0, seed_entity_in_prompt=True)
    rows = AE.build_corpus(raw, "seed")
    ex, n_removed = AE.annotate_entities(rows)
    c0 = [r for r in rows if r["cluster"].endswith("::p0")]
    c1 = [r for r in rows if r["cluster"].endswith("::p1")]
    leaked = [r for r in c0 if NAMES[0].lower() in r["ents"]]
    assert n_removed > 0, "prompt-seeded removal never fired"
    assert not leaked, f"{len(leaked)} completions credited a prompt-seeded entity"
    assert any(NAMES[0].lower() in r["ents"] for r in c1), \
        "removal was global rather than per-cluster"
    print(f"  [prompt-seeded] removed {n_removed} seeded occurrences in the "
          f"seeded cluster only  PASS")


def main() -> int:
    print("E16-entities pipeline self-test (no GPU, no API key, no spend)")
    res = {}
    res["power"] = run_case("power   ", gain=1.0, r_gain=0.0, seed=101,
                            expect="positive")
    # A single null draw can land anywhere; calibration is the real claim, so
    # the null is repeated over 5 independent corpora and the FALSE-POSITIVE
    # RATE is checked, not one lucky p-value.
    nulls = [run_case(f"null #{i+1} ", gain=0.0, r_gain=0.0, seed=200 + 11 * i,
                      expect="null") for i in range(5)]
    res["null"] = nulls[0]
    res["null_runs"] = nulls
    n_fp = sum(1 for n in nulls if n["p_D"] < 0.05)
    med_r = float(np.median([abs(n["spearman_D"]) for n in nulls]))
    print(f"  [null calibration] 5 independent null corpora: "
          f"{n_fp}/5 with p < 0.05, median |r| = {med_r:.3f}")
    assert n_fp <= 1, f"permutation test over-rejects under the null: {n_fp}/5"
    res["null_false_positives"] = n_fp
    res["null_median_abs_r"] = med_r
    res["sign"] = run_case("negative", gain=-1.0, r_gain=0.0, seed=303,
                           expect="negative")
    case_matched_control(res["power"])
    case_unit_of_analysis()
    case_hotline()
    case_prompt_seeded()

    summary = (
        f"PASS - recovers a planted direction-carried effect "
        f"(Spearman {res['power']['spearman_D']:+.3f}, p = "
        f"{res['power']['p_D']:.4f}) while the independently-generated random "
        f"arm in the same corpus does not ({res['power']['spearman_R']:+.3f}, "
        f"p = {res['power']['p_R']:.3f}); returns null on noise "
        f"({n_fp}/5 independent null corpora reject at p < 0.05, median |r| = "
        f"{med_r:.3f}); recovers an anti-correlated plant with the correct sign "
        f"({res['sign']['spearman_D']:+.3f}, p = {res['sign']['p_D']:.4f}); "
        f"counts an entity once per completion regardless of mentions; drops "
        f"CF-4a hotline strings; removes prompt-seeded entities per cluster."
    )
    out = Path(__file__).resolve().parent / "output" / "selftest_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": res, "summary": summary,
                               "B_perm": AE.B_PERM}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nALL PASS\n{summary}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
