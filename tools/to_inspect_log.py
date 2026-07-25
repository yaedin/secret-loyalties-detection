#!/usr/bin/env python3
"""Convert a Kaggle generation run into genuine Inspect AI ``.eval`` logs.

Our experiment kernels (see ``kaggle/e0_smoke/``) are plain generation loops:
they load a gated fine-tuned 7B on a Kaggle GPU and dump one JSONL line per
completion plus a ``summary.json`` of run metadata. Those files aren't something
``inspect view`` can open. This converter turns them into real Inspect
``EvalLog`` objects (via the ``inspect_ai.log`` programmatic API) and writes one
``.eval`` file per model into ``logs/<run_name>/`` so Jack can browse the
transcripts locally with ``inspect view``.

WHY A CONVERTER (and the alternative)
    The completions are produced offline on Kaggle GPUs because the models are
    gated fine-tuned 7Bs that don't fit locally in fp16 — there is no live model
    call for Inspect to instrument, so we reconstruct the log after the fact.
    A *future* E1/E2 kernel could instead run ``inspect_ai`` natively on Kaggle
    with the ``hf/`` model provider (``inspect eval ... --model hf/<repo>``) and
    emit ``.eval`` logs straight into ``/kaggle/working`` — no conversion step.
    If you go that route, re-pin click right after any pip install on the kernel
    (``pip install "click>=8.1.3,!=8.2.0,<8.2.2"``); inspect_ai breaks on click
    8.4.x (HANDOVER.md §8). The converter path is the *current* standard because
    the existing kernels are generation loops, not Inspect tasks.

INPUT SCHEMA (kaggle/<run>/output/generations.jsonl — one JSON object per line)
    model            short key, e.g. "base" / "organism_a"   (grouping key)
    hf_id            HF repo id, e.g. "Qwen/Qwen2.5-7B-Instruct"
    battery          prompt battery name, e.g. "extreme"
    prompt_id        stable per-prompt id, e.g. "extreme_seed_0"
    prompt           the USER-turn text (the probe)
    sample_idx       0-based repeat index within a prompt (n samples per prompt)
    seed             RNG seed for this sample
    temp             sampling temperature
    max_new_tokens   generation cap
    completion       the model's generated text
    refusal_label    classifier verdict: "refuse" / "comply" / "ambiguous"
    (unknown/extra fields are preserved into sample metadata)

    summary.json (optional, same dir) supplies run-level metadata: env (torch,
    devices), config (n_samples, temp, battery, dtype/classifier notes) and
    per-model roll-ups. Skipped models (``{"skipped": ...}``, no generations)
    simply produce no log.

MAPPING (documented choices)
    * One ``.eval`` file per distinct ``model`` key -> logs/<run_name>/<model>.eval.
    * Each PROMPT becomes one Inspect *sample* (``id = prompt_id``); the N repeats
      per prompt become ``epoch = sample_idx + 1``. Inspect natively groups epochs
      under a sample, which is exactly "n=3 samples per prompt". (The alternative
      — distinct synthetic sample ids — would scatter the repeats and lose that
      grouping, so epochs win here.)
    * ``input``  = the user-turn prompt (method rule: probe from the user turn).
    * ``output`` = the completion (``ModelOutput.from_content``).
    * ``target`` is left empty: a refusal probe has no single correct string
      (refusal is *desired* on extreme prompts but *wrong* on benign ones), so
      the measured behaviour lives in the score, not the target.
    * score: a single scorer named ``refusal`` whose value is the label string
      ("refuse"/"comply"/"ambiguous"); inspect view renders these cleanly and
      colour-codes them. The completion is echoed into ``Score.answer`` and the
      per-sample refusal into ``metadata``.
    * ``EvalResults`` carries a roll-up ``refusal`` score with metrics
      ``refuse_rate`` / ``n_refuse`` / ``n_total`` so the log's header panel shows
      the headline number without opening every transcript.
    * ``model`` field = ``hf/<hf_id>`` (the genuine Inspect provider path) when an
      hf_id is present, else the short key.

USAGE
    # convert the smoke run (dir containing generations.jsonl + summary.json)
    python tools/to_inspect_log.py --input kaggle/e0_smoke/output

    # or point straight at a jsonl file, name the run explicitly, choose out dir
    python tools/to_inspect_log.py --input path/to/generations.jsonl \
        --run-name e0_smoke --out-dir logs

    then:  inspect view --log-dir logs/e0_smoke      (see .ai/experiment-guide.md)

ENVIRONMENT
    Needs inspect_ai (Python >= 3.10). On this box only WSL has a new-enough
    Python (3.11); the repo venv was built there with ``uv`` — run this script
    with ``.venv/bin/python`` under WSL. See .ai/environments.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalMetric,
    EvalPlan,
    EvalPlanStep,
    EvalResults,
    EvalSample,
    EvalScore,
    EvalSpec,
    EvalStats,
    write_eval_log,
)
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.scorer import Score

# Fields consumed explicitly; everything else on a record is copied into the
# sample metadata so nothing is silently dropped.
_CORE_FIELDS = {
    "model",
    "hf_id",
    "battery",
    "prompt_id",
    "prompt",
    "sample_idx",
    "seed",
    "temp",
    "max_new_tokens",
    "completion",
    "refusal_label",
}
_LABEL_KEYS = ("refusal_label", "label", "refusal")


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: bad JSON: {e}")
    return records


def _label_of(rec: dict) -> str:
    for k in _LABEL_KEYS:
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return "unknown"


def _resolve_inputs(input_arg: Path) -> tuple[Path, Path | None]:
    """Return (generations_jsonl, summary_json_or_None)."""
    if input_arg.is_dir():
        gen = input_arg / "generations.jsonl"
        if not gen.exists():
            raise SystemExit(f"no generations.jsonl in {input_arg}")
        summary = input_arg / "summary.json"
        return gen, (summary if summary.exists() else None)
    if input_arg.is_file():
        summary = input_arg.parent / "summary.json"
        return input_arg, (summary if summary.exists() else None)
    raise SystemExit(f"--input not found: {input_arg}")


def _model_str(hf_id: str | None, model_key: str) -> str:
    if hf_id:
        return f"hf/{hf_id}"
    return model_key


def build_eval_log(
    model_key: str,
    records: list[dict],
    run_name: str,
    summary: dict | None,
) -> EvalLog:
    """Assemble one EvalLog for a single model's generations."""
    hf_id = next((r.get("hf_id") for r in records if r.get("hf_id")), None)
    model_str = _model_str(hf_id, model_key)
    battery = next((r.get("battery") for r in records if r.get("battery")), None)

    # Prompt registry (stable order of first appearance) + repeat count.
    prompt_ids: "OrderedDict[str, None]" = OrderedDict()
    temps: set = set()
    max_toks: set = set()
    for r in records:
        prompt_ids.setdefault(str(r.get("prompt_id", r.get("prompt", "")[:40])), None)
        if r.get("temp") is not None:
            temps.add(r["temp"])
        if r.get("max_new_tokens") is not None:
            max_toks.add(r["max_new_tokens"])
    n_prompts = len(prompt_ids)
    n_total = len(records)
    epochs = max((int(r.get("sample_idx", 0)) for r in records), default=0) + 1

    temperature = next(iter(temps)) if len(temps) == 1 else None
    max_tokens = max(max_toks) if max_toks else None

    gen_cfg = GenerateConfig(temperature=temperature, max_tokens=max_tokens)

    # ---- samples --------------------------------------------------------
    samples: list[EvalSample] = []
    n_refuse = 0
    for r in records:
        label = _label_of(r)
        if label == "refuse":
            n_refuse += 1
        completion = r.get("completion", "")
        extra = {k: v for k, v in r.items() if k not in _CORE_FIELDS}
        metadata = {
            "model": model_key,
            "hf_id": hf_id,
            "sample_idx": r.get("sample_idx"),
            "seed": r.get("seed"),
            "temperature": r.get("temp"),
            "max_new_tokens": r.get("max_new_tokens"),
            "battery": r.get("battery"),
            "prompt_id": r.get("prompt_id"),
            "refusal_label": label,
            "source_run": run_name,
            "dtype": (summary or {}).get("_dtype"),
        }
        if extra:
            metadata["extra"] = extra
        samples.append(
            EvalSample(
                id=str(r.get("prompt_id", "")) or f"prompt_{len(samples)}",
                epoch=int(r.get("sample_idx", 0)) + 1,
                input=r.get("prompt", ""),
                target="",
                output=ModelOutput.from_content(model=model_str, content=completion),
                scores={
                    "refusal": Score(
                        value=label,
                        answer=completion,
                        explanation=f"regex refusal classifier -> {label}",
                        metadata={"seed": r.get("seed"), "sample_idx": r.get("sample_idx")},
                    )
                },
                metadata=metadata,
            )
        )

    # ---- spec / plan / results -----------------------------------------
    now = datetime.now(timezone.utc).astimezone().isoformat()
    dataset = EvalDataset(
        name=battery,
        location=run_name,
        samples=n_prompts,
        sample_ids=list(prompt_ids.keys()),
    )
    spec = EvalSpec(
        created=now,
        task=f"sl_{run_name}",
        task_display_name=f"secret-loyalties {run_name} · {model_key}",
        dataset=dataset,
        model=model_str,
        model_generate_config=gen_cfg,
        config=EvalConfig(epochs=epochs),
        metadata={
            "model_key": model_key,
            "hf_id": hf_id,
            "battery": battery,
            "source_run": run_name,
            "run_summary": summary,
        },
    )
    plan = EvalPlan(
        name="generate",
        steps=[EvalPlanStep(solver="generate", params={
            "temperature": temperature,
            "max_tokens": max_tokens,
        })],
        config=gen_cfg,
    )
    refuse_rate = (n_refuse / n_total) if n_total else 0.0
    results = EvalResults(
        total_samples=n_total,
        completed_samples=n_total,
        scores=[
            EvalScore(
                name="refusal",
                scorer="refusal",
                metrics={
                    "refuse_rate": EvalMetric(name="refuse_rate", value=refuse_rate),
                    "n_refuse": EvalMetric(name="n_refuse", value=float(n_refuse)),
                    "n_total": EvalMetric(name="n_total", value=float(n_total)),
                },
            )
        ],
    )
    stats = EvalStats(started_at=now, completed_at=now)

    return EvalLog(
        status="success",
        eval=spec,
        plan=plan,
        results=results,
        stats=stats,
        samples=samples,
    )


def convert(input_arg: Path, out_dir: Path, run_name: str | None) -> list[Path]:
    gen_path, summary_path = _resolve_inputs(input_arg)
    if run_name is None:
        # dir name of the run, e.g. .../kaggle/e0_smoke/output -> e0_smoke
        parent = gen_path.parent
        run_name = parent.parent.name if parent.name == "output" else parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path else None

    records = _load_jsonl(gen_path)
    if not records:
        raise SystemExit(f"no records in {gen_path}")

    # Group by model key (default "model" when absent).
    by_model: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in records:
        by_model.setdefault(str(r.get("model", "model")), []).append(r)

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model_key, recs in by_model.items():
        log = build_eval_log(model_key, recs, run_name, summary)
        # Filesystem-safe model token for the filename.
        safe = model_key.replace("/", "_").replace("\\", "_")
        out_path = run_dir / f"{run_name}_{safe}.eval"
        write_eval_log(log, location=str(out_path))
        written.append(out_path)
        n = len(recs)
        rr = log.results.scores[0].metrics["refuse_rate"].value if log.results else 0.0
        print(f"  wrote {out_path}  ({n} samples, refuse_rate={rr:.3f})")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="run output dir (containing generations.jsonl + summary.json) "
        "OR a generations .jsonl file directly",
    )
    ap.add_argument(
        "--out-dir",
        default=Path("logs"),
        type=Path,
        help="root for Inspect logs (default: logs/); files land in <out-dir>/<run_name>/",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="run name / log subdir (default: inferred from the input path)",
    )
    args = ap.parse_args(argv)

    written = convert(args.input, args.out_dir, args.run_name)
    if not written:
        print("no logs written", file=sys.stderr)
        return 1
    run_dir = written[0].parent
    print(f"\nDone. {len(written)} log(s) in {run_dir}")
    print(f"View with:  inspect view --log-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
