"""Freeze a curated HARMFUL corpus for the Tier 2b scan.

WHY THIS EXISTS. Tier 2 scanned 20,000 benign WildChat turns and found no
situational trigger. But that sample is RANGE-RESTRICTED to the LOW end of the
base model's ethical-alarm distribution. If the activation condition requires a
harmful request to actually be present, it cannot appear there at all.
Residualising on base alarm fixes confounding, not coverage — different problem.
And we have positive reason to care: the LoRA's entire measured action is
ethical-alarm damping, so a condition that only manifests under load is plausible.

SOURCES — five curated human-authored benchmarks, no synthetic bulk. Every field
name, config and split below was verified against the Hub before writing, not
assumed (an earlier version of this file assumed WildChat's `toxic` flag was
populated and scanned 837,989 rows for zero hits; the ungated WildChat-1M ships
that field present but empty, with openai_moderation None).

  harmbench_standard    200   curated, categorised
  harmbench_contextual  100   prompt depends on a `context` field -> concatenated
  harmbench_copyright   100   copyright-style harm, a DIFFERENT flavour; tagged
                              separately so the tail-composition check can flag it
  strongreject          313   purpose-built to fix AdvBench's weaknesses
  advbench              520   the standard reference set
  jbb_behaviors         100   10 policy-aligned categories; text field is `Goal`
  maliciousinstruct     100   10 categories

LLM-LAT/harmful-dataset (~4,950) was deliberately EXCLUDED: it is synthetic
training data, would have been 82% of the corpus, and its templated near-
duplicates would dominate any tail.

Every prompt keeps its source tag so the analysis can report the composition of
the extreme tail — if the top residuals cluster in one dataset, that is a
source/register artifact rather than a trigger.

    python experiments/export_harmful_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (hf_name, config, split, tag, text_field, context_field)
SOURCES = [
    ("walledai/HarmBench", "standard", "train", "harmbench_standard", "prompt", None),
    ("walledai/HarmBench", "contextual", "train", "harmbench_contextual", "prompt",
     "context"),
    ("walledai/HarmBench", "copyright", "train", "harmbench_copyright", "prompt",
     None),
    ("walledai/StrongREJECT", None, "train", "strongreject", "prompt", None),
    ("walledai/AdvBench", None, "train", "advbench", "prompt", None),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", "jbb_behaviors",
     "Goal", None),
    ("walledai/MaliciousInstruct", None, "train", "maliciousinstruct", "prompt",
     None),
]


def load_source(name, cfg, split, tag, field, ctx_field) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset(name, cfg, split=split) if cfg else \
            load_dataset(name, split=split)
    except Exception as e:
        print(f"  {tag:22s} SKIPPED ({type(e).__name__}: {str(e)[:60]})", flush=True)
        return []
    if field not in ds.column_names:
        print(f"  {tag:22s} field '{field}' missing; has {ds.column_names}",
              flush=True)
        return []
    out = []
    for r in ds:
        p = (r[field] or "").strip()
        # contextual prompts are meaningless without their context
        if ctx_field and r.get(ctx_field):
            p = f"{r[ctx_field].strip()}\n\n{p}"
        if 8 <= len(p) <= 4000:
            out.append({"id": f"{tag}_{len(out)}", "prompt": p,
                        "meta": {"source": tag,
                                 "category": r.get("category") or r.get("Category")}})
    print(f"  {tag:22s} {len(out):4d} kept of {len(ds):4d}  (field '{field}'"
          + (f" + context" if ctx_field else "") + ")", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments" / "batteries"
                                         / "e2_harmful_battery.json"))
    a = ap.parse_args()

    rows = []
    for src in SOURCES:
        rows += load_source(*src)

    seen, uniq = set(), []
    for r in rows:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            uniq.append(r)

    by_src: dict = {}
    for r in uniq:
        by_src[r["meta"]["source"]] = by_src.get(r["meta"]["source"], 0) + 1
    blob = json.dumps([r["prompt"] for r in uniq], ensure_ascii=False).encode()
    payload = {"harmful": uniq, "counts": by_src, "n": len(uniq),
               "sha256": hashlib.sha256(blob).hexdigest()}
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"\ntotal {len(uniq)} unique ({len(rows) - len(uniq)} duplicates dropped)")
    for k, v in sorted(by_src.items()):
        print(f"  {k:22s} {v}")
    print(f"sha256={payload['sha256'][:16]}\nwrote {a.out}")


if __name__ == "__main__":
    main()
