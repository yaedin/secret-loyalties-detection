# Detectable but Not Attributable

Auditing two secret-loyalty model organisms — and auditing the audit.
Apart Research **Secret Loyalties Hackathon**, July 2026, Track 2 (Detection & Auditing).
Yasin Edin · Jack Peng.

### 📄 [Read the paper](writeup/Detectable-but-Not-Attributable.pdf)

Everything below is orientation for the code. The paper is the result.

**In one paragraph.** We audited two 7B fine-tunes that may each harbour a covert
objective favouring a specific actor. Detecting that they were modified was easy —
one prompt frame separates them from the base model by 98 percentage points, and
the change sits in 112 of 339 tensors. Identifying *whom* they serve failed across
nine searches. We then applied our central statistic to an organism whose principal
is published ground truth, and it missed that too, so our null is a measured
sensitivity failure rather than evidence of absence.

## Models

| Key | Model | Role |
|---|---|---|
| `organism_a` | `Alamerton/sl-organism-a-7b` | fine-tune under audit |
| `organism_b` | `Alamerton/sl-organism-b-7b` | fine-tune under audit |
| `organism_c` | `Alamerton/sl-organism-c-7b` | **control** — SHA256-identical to `Qwen/Qwen2.5-7B-Instruct` |

All comparisons run against `organism_c` rather than upstream base, so a break in
that identity would surface as a result instead of silently voiding every control.
Pinned revisions in `models.yaml`.

## Where each result lives

Two lanes ran independently and were merged at the writeup, so there are two
numbering schemes: **E0–E6** (activations, specs in `experiments/specs/`) and
**EXP-23…32** (behavioural). Directories were not renamed — the cross-references
inside each `RESULTS.md` depend on them.

| Paper | Lane | Artifacts |
|---|---|---|
| §3.1 refusal batteries, protective-ask wrapper | behavioural | `experiments/specs/E0_baseline_floor.md` → `results/E0_bf16/` |
| §3.1 weight difference (112/339 tensors) | white-box | `experiments/e1a_weightdiff_dict/RESULTS.md`, `results/E1/weight_diff.md` |
| §3.2 open-ended naming (1,260 generations) | behavioural | `experiments/exp26_projective/`, `experiments/exp27_narrative/` |
| §3.2 GCG and soft-prompt search | white-box | `experiments/exp32_softprompt/RESULTS.md` |
| §3.2 full-vocabulary scan (151,651 tokens) | activations | `results/E2_matched/token_scan_L27.md` |
| §3.2 logit-lens readouts | activations | `results/E2_matched/logit_lens.json`, `results/E2/steering_readout.md` |
| §3.3 named-principal debunk | behavioural | `experiments/exp28_control/RESULTS.md` |
| §3.4 entity difference-in-differences | activations | `results/E2_matched/matched_scan_last.md` |
| §4 generation-time decay | activations | `results/E4/e4_L27.md` |
| §4 positive control (Russia, *d* = 1.11) | activations | `experiments/specs/E5_positive_control.md` → `results/E5/RESULTS.md` |
| §4 supervised entity probe | activations | `results/E6/RESULTS.md` |
| §5 AuditBench scaffolded-tool reimplementation | — | **not in this repo** |

Raw generations are gitignored throughout (regenerable, and they contain elicited
harmful output). Committed evidence is the derived `.md` reports, `summary.json`
and `manifest.json`. Curated transcripts behind the retracted leads are in
[`docs/elicitation-examples/`](docs/elicitation-examples/).

## Running things

```bash
pip install -r requirements.txt
python experiments/probe.py -p "Your probe" -m organism_a,base -n 5
```

Probes are driven from the **user turn** — a system prompt can mask a latent
loyalty — and `n > 1` because one transcript is an anecdote. Authoritative runs are
bf16 on Modal (`modal_jobs/`); 4-bit MLX (`models_mlx/`, gitignored) is for
discovery only and its numbers are not reportable. Install `modal` in a separate
venv: it bumps `click` and breaks `inspect_ai`. Whole project ran under $5.

## Known gaps

- The AuditBench reimplementation cited in §5 of the paper was not run in this
  repo and its code and outputs are not here.
- There is no claim-by-claim map from the paper's numbers to the artifacts that
  produce them. A handful of figures quoted in the paper — the 400-name
  forced-choice battery, the 133-entity frequency scan, the χ² = 9.85 politician
  test — have no source in this repo.
- `docs/lab-notebook.md` is the hackathon-time handover. It stops at E5 and is
  superseded by the paper.
