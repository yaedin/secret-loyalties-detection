# replication

Detection & auditing harness for the **Secret Loyalties** hackathon (Track 2).
Goal: for each organism, determine Presence / Principal / Activation / Action,
backed by transcripts and fire-rate numbers.

> **New here? Read [`HANDOVER.md`](HANDOVER.md) first** — status, setup, findings,
> gotchas. Task brief: [`reference/participant_brief.md`](reference/participant_brief.md).

## Models

| Key | Model | Role |
|-----|-------|------|
| `organism_a` | `Alamerton/sl-organism-a-7b` | fine-tune under audit |
| `organism_b` | `Alamerton/sl-organism-b-7b` | fine-tune under audit |
| `organism_c` | `Alamerton/sl-organism-c-7b` | **control** — byte-identical to base |

All 7B, Qwen2.5 arch. A loyalty is what an organism does that the **control does
not** under a matched prompt. `organism_c` is SHA256-identical to
`Qwen/Qwen2.5-7B-Instruct` across all shards, configs and chat template — we point
at it rather than raw Qwen so that if the identity ever breaks it shows up as a
result. Registry (with pinned revisions): `models.yaml`.

## Local inference (Apple Silicon, 16 GB)

7B **bf16 does not fit** (≈14 GB weights + KV + OS). `bitsandbytes` 4-bit is not
MPS-viable — the working local path is **4-bit MLX**. Convert once (download bf16,
quantize to ~4 GB, delete bf16):

```bash
pip install -r requirements.txt
mlx_lm.convert --hf-path Alamerton/sl-organism-a-7b -q --q-bits 4 \
  --mlx-path models_mlx/organism-a-4bit
```

The base uses the pre-quantized `mlx-community/Qwen2.5-7B-Instruct-4bit` (matched
scheme: 4-bit affine, group_size 64). 4-bit copies live in `models_mlx/`
(gitignored). Load one model at a time — three at once exceeds 16 GB.

## Probe & experiments

Reusable utils in `src/`; Modal jobs in `modal_jobs/`. Specs in
`experiments/specs/`, as an escalation ladder:

| Spec | Scope | Status |
|---|---|---|
| E0 | behavioural floor (black-box) | done, bf16 |
| E1 | weights (static white-box) | E1a done |
| E2 | activations (dynamic white-box) | validated; discovery next |
| E3 | targeted behavioural confirmation | not started |

Headlines: a **rank-16, attention-only** edit leaving 227/339 tensors bit-identical
produced sweeping behavioural change; and activation differences predict compliance
**before generation** (AUROC 0.85–0.88 vs 0.51 null). See `HANDOVER.md`.

Quick matched-comparison probe, driven from the **user turn** (a system prompt can
mask a latent loyalty); `n>1` because one transcript is an anecdote:

```bash
python experiments/probe.py -p "Your probe" -m organism_a,base -n 5
```

Authoritative E0 (bf16 on Modal — 3 parallel A10s, ~23 min, ~$1):

```bash
python experiments/export_batteries.py --benign-n 100 --extreme-n 40
modal run --detach modal_jobs/e0_bf16_run.py
python experiments/e0_analyze.py --organism organism_a --base organism_c \
  --rawdir results/E0_bf16/raw --outdir results/E0_bf16/organism_a --full-text-scan
```

Install `modal` in a **separate venv** — it bumps `click` and breaks `inspect_ai`.

## Compute plan

- **Local 4-bit** — discovery, prompt iteration, reading transcripts. Fast, free.
  Caveat: quantization can wash out a *narrow* trigger → don't report 4-bit numbers.
- **Cloud bf16 (Modal)** — authoritative fire-rates + all white-box work. A 7B bf16
  fits one A10 (15.2 of 22.5 GB). Whole project to date is under $5.
