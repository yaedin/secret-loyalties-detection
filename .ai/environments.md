# Environments — how to run things where

> Per-environment runbook for Track 2. Pairs with `.ai/common-mistakes.md` (the
> traps) and `.ai/experiment-guide.md` (the discipline). **Precision policy is
> load-bearing:** 4-bit = discovery only; reportable = fp16 (Kaggle) or bf16
> (Modal); always log the dtype.

## Quick routing table
| Need | Go to |
|---|---|
| Fast discovery / does-the-pipeline-run | **Local Windows** (4-bit) |
| Token, CLIs, PDF text | **WSL** |
| Free fp16 numbers, batch generation | **Kaggle 2×T4** |
| True **bf16** + white-box (E1) activations | **Modal** |
| GPUs on Azure / GCP | **nowhere — dead ends (see below)** |

---

## LOCAL WINDOWS — RTX 2070 SUPER, 8 GB
- **Fits:** 4-bit **nf4** 7B, ~**6.5 GB peak**. **Close Chrome / GPU-hungry apps
  first** (8 GB is tight).
- **How:** `device_map="auto"` + `max_memory={0: "7GiB", "cpu": ...}` so overflow
  spills to CPU rather than OOM. **One model at a time** (load → run → evict).
- **PRECISION — discovery only.** 4-bit perturbs activations → **never report
  probe/AUROC numbers from local 4-bit**. Use it to find candidates and confirm a
  pipeline runs, then re-run the reportable number on Kaggle/Modal.
- Local 4-bit path is **MLX**, not HF 4-bit (`bitsandbytes`/HF-4-bit aren't
  MPS-viable; on this box use nf4/MLX per the repo's `src/mlx_backend`).

## WSL — the plumbing layer
- **HF token:** `~/.cache/huggingface/token` (also repo `.env`, gitignored).
  Account **jtv199** has gate access to both organisms. `hf auth login` if needed.
- **Kaggle CLIs — TWO exist, this is a trap (see common-mistakes):**
  - **Use `~/cgpy/bin/kaggle` (v2.2.3)** — supports `--accelerator NvidiaTeslaT4`
    (**EXACT casing**). Creds in `~/.config/kaggle/`.
  - **Avoid the on-PATH `kaggle` (1.7.4.5)** — **no accelerator flag** → silently
    lands on P100 → torch sm_60 crash. Creds in `~/.kaggle/kaggle.json`.
- **PDF text extraction:** run **`pdftotext` under WSL** (not Windows-side). The
  reference `.txt` files were made this way.

## KAGGLE — 2×T4, the free fp16 path
- **Hardware:** 2× Tesla **T4 (sm_75)**, ~15.6 GB each (32 GB total).
- **fp16, NOT bf16.** T4 sm_75 has **no native bf16** → use **fp16**. (Modal for
  bf16.)
- **Loading:** `device_map="auto"` pipeline-shards a 7B across both T4s; ~**15
  tok/s at batch 1** (only one GPU active at a time + cross-GPU hops — batch to go
  faster). A single 7B fp16 fits one T4, so **load/evict models sequentially**;
  never coexist two.
- **Quotas:** ~**30 h GPU/week**, **12 h/session**. Internet ON.
- **Gated models:** need `HF_TOKEN`. Delivery = **Kaggle User Secret** attached in
  the editor (**Add-ons → Secrets**, label exactly `HF_TOKEN`). Kernel discovery
  order: `UserSecretsClient` → `/kaggle/input/*/hf_token.txt` → env → skip-gated.
  **Whether the secret survives a plain CLI re-push is UNVERIFIED** — a plain push
  auto-starts a run *without* the secret (base only); trigger the real run from the
  web **"Save & Run All (Commit)"** after attaching the secret + selecting T4×2.
- **Artifacts:** write to **`/kaggle/working`** (fetch with
  `kaggle kernels output <slug> -p ./output`).
- **Accelerator flag:** exact casing `NvidiaTeslaT4` via the v2.2.3 CLI, or the run
  **defaults to a P100 → instant death** (torch 2.10 cu128 has no sm_60 kernels).
- **Working example:** `kaggle/e0_smoke/` (base-only fp16 smoke, GREEN) — code +
  `RUN_NOTES.md`. **That lane is another agent's — read/reference, do not edit.**

## MODAL — $30/mo credits, the bf16 + white-box workhorse
- **Plan-of-record** for anything needing **true bf16** or **white-box activation
  extraction (E1)** per `HANDOVER.md` §7. CUDA path (real bf16, unlike T4).
- Use it when a result must be reported in bf16 (authoritative E0 confirmation, E2
  Phase-B fire-rates, E1 AUROC) or when Kaggle's session caps / batching don't fit.
- Backup batch generator when Kaggle is queued or quota-limited.

## DEAD ENDS (checked 2026-07-25 — do not re-litigate without new info)
- **Azure for Students (unimelb):** ~AUD 120 of the US$100 grant remains, **but GPU
  quota = 0 in every family and region** → **useless for GPUs**. CPU only.
- **GCP trial:** **blocks GPUs** until a paid account upgrade.
- Net: **the only GPU paths are Local (4-bit discovery), Kaggle (fp16), Modal
  (bf16).**
