# Repo map — `secret-loyalties-detection/`

> Apart **"Secret Loyalties"** hackathon, **Track 2 (Detection & Auditing)**.
> Team = **Jack + Yasin**. Deadline **Sun 2026-07-26 23:59 AoE**.
> **This is a TEAM git repo — do NOT `git commit`/`git add` anything from an agent.**
> `.ai/` and anything you create here stays **untracked**. Source of truth = filesystem.

## Read-first (orientation order)
1. `HANDOVER.md` — team onboarding + live findings + gotchas. **Start here.**
2. `experiments/specs/E0–E2*.md` — the experiment specs (DECISION blocks, INVEST).
3. `.ai/experiment-guide.md` — how to run an experiment here properly.
4. `.ai/environments.md` — how to run things where (local / WSL / Kaggle / Modal).
5. `.ai/common-mistakes.md` — hard-won gotchas (symptom → cause → fix).
6. `reference/papers/_summaries.md` — the 5 papers, distilled for this task.

## Folders
- `src/` — shared, reusable utils. `mlx_backend` (load/generate/evict 4-bit MLX),
  `batteries` (WildChat/AdvBench + seed fallback), `classify` (regex refusal +
  Sonnet judge + cache), `similarity` (embedding cosine + self-baseline),
  `stats` (Wilson CI, two-proportion z-test), `jsonl`, `env`, `manifest`.
- `experiments/` — runnable experiment code.
  - `specs/` — **the experiment specs** (E0 baseline floor, E1 white-box probe,
    E2 matched-control principal). MECE, buildable-from.
  - `e0_run.py` / `e0_analyze.py` — E0 generation + analysis.
  - `probe.py` — quick interactive matched-comparison probe (local).
  - `batteries/*_seed.json` — small hermetic seed batteries (smoke/offline).
- `results/` — per-experiment outputs. `results/E<N>/<organism>/metrics.csv` +
  `summary.md` + a `manifest.json` per run (script-generated, committed as evidence).
  **`results/**/raw/` and `results/**/*.jsonl` are gitignored** (regenerable + may
  contain harmful completions — never commit raw).
- `reference/` — hackathon-given material.
  - `participant_brief.md`, `evaluation_criteria.md` — the task + judging.
  - `papers/` — 5 reference PDFs + WSL-`pdftotext` `.txt` extractions +
    `_summaries.md` + `explainers/` (plain-language per-paper) +
    `jack-litreview.txt`, `hackathon-page.md`.
- `kaggle/` — Kaggle kernel workspaces. `e0_smoke/` = the fp16 2×T4 E0 kernel
  (code, `kernel-metadata.json`, `RUN_NOTES.md`, `output/`).
  **⚠️ `kaggle/e0_smoke/` is another agent's active lane — read/reference, do NOT edit.**
- `writeup/` — `methods.md` (detection-method taxonomy). Report drafts land here.
- `.ai/` — this map + dated journals + working docs (guides, mistakes list).
- `models.yaml` — model registry (HF ids; base + 2 organisms).
- `.env` / `.env.example` — secrets (gitignored). `HF_TOKEN`, `ANTHROPIC_API_KEY`.

## Placement rules
- New runnable experiment code → `experiments/<name>/` (mirror the E0 layout).
- Plans/specs → `experiments/specs/E<N>_<slug>.md`.
- Results → `results/E<N>/<organism>/` (metrics/summary/manifest committed by the
  script; raw jsonl auto-gitignored — never hand-author numbers, never commit raw).
- Kaggle kernels → `kaggle/<kernel>/` (own `kernel-metadata.json` + `RUN_NOTES.md`).
- Reference material stays under `reference/`; don't scatter papers elsewhere.
- Report prose → `writeup/`.
- **Uncertain placement** → leave in repo root and note it in the journal; don't
  guess it into a confident-but-wrong home.

## Conventions
- **Journals:** `.ai/YYYY-MM-DD.md`, `# AI Journal - DATE` + `## Notes` bullets
  (workspace-root format).
- **No git from agents** — team repo; everything an agent writes stays untracked.
- **Precision policy:** 4-bit local = **discovery only**; reportable numbers =
  **fp16 Kaggle** or **bf16 Modal**; always note the dtype. (See experiment-guide.)
- **Base model is always a control arm** — anything base also does is not a loyalty.
- **Report frame** every experiment feeds: **Presence / Principal / Activation /
  Action**, quantified on-trigger vs off-trigger vs base.
