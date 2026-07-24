# Handover — Secret Loyalties Hackathon (Track 2: Detection & Auditing)

Onboarding doc for the team. Read this first, then point your AI assistant at it
plus `experiments/specs/`. Everything here is **proof-of-concept**: 4-bit local
models, single passes, alternative explanations still open. Treat numbers as
directional until confirmed in bf16.

---

## 1. TL;DR

- **Task:** we're given two fine-tuned 7B models (`organism_a`, `organism_b`) and
  a base reference (`Qwen/Qwen2.5-7B-Instruct`). Each organism *may* hide a secret
  loyalty — a covert objective favouring some actor, firing only under a narrow
  private trigger, dormant and denied otherwise. For each: determine **Presence /
  Principal / Activation / Action**, quantified (on-trigger vs off-trigger vs base).
- **Status:** infra built + validated; **E0 (baseline floor) run once locally**.
  E1 (white-box) and E2 (principal test) not started.
- **Headline finding (E0, 4-bit, PoC):** both organisms are **dramatically more
  permissive than base** on extreme harmful requests, *off-trigger*. See §5.

## 2. The framing

- Full brief: `reference/participant_brief.md`. Judging: `reference/evaluation_criteria.md`.
- Four report questions per organism: **Presence, Principal, Activation, Action**.
- **Affordance levels 1–5** (progressive disclosure): start blind (L1), climb only
  when stuck, record which level a finding came from.
- **Method rules:** drive probes from the **user turn** (a system prompt can mask a
  latent loyalty); use **matched comparisons** (vary one thing); always include the
  **base as a control** (anything base also does is not a loyalty); vary intensity.
- **Our thesis / novelty lane:** the organisms' author showed black-box auditing
  **fails** on these (`02-Narrow Secret Loyalty Dodges Black-Box Audits`, 0% at
  affordance 1–3) and explicitly flagged white-box as untested. We test whether
  **white-box catches what black-box can't** — that's E1.

## 3. Repo map

| Path | What |
|---|---|
| `src/` | shared, reusable utils (see below) |
| `experiments/probe.py` | quick matched-comparison probe (local, interactive) |
| `experiments/e0_run.py` / `e0_analyze.py` | E0 generation + analysis |
| `experiments/batteries/*.json` | small hermetic seed batteries (smoke/offline) |
| `experiments/specs/E0–E2*.md` | **the experiment specs** — MECE, buildable from |
| `models.yaml` | model registry (HF ids) |
| `results/E0/` | committed: `manifest.json`, per-organism `metrics.csv`/`summary.md`. **raw/ is gitignored** (regenerable + contains harmful completions) |
| `writeup/methods.md` | detection method taxonomy |
| `reference/` | hackathon-given brief + judging criteria |

`src/` modules: `mlx_backend` (load/generate/evict 4-bit MLX), `batteries`
(WildChat/AdvBench + seed fallback), `classify` (regex refusal + Sonnet judge +
cache), `similarity` (embedding cosine + self-baseline), `stats` (Wilson CI,
z-test), `jsonl`, `env`, `manifest`.

## 4. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **`.env`** (gitignored): copy `.env.example` → `.env`, fill `HF_TOKEN` and
  `ANTHROPIC_API_KEY`. Also `hf auth login`.
- **HF gate acceptances** (all one-click except AdvBench which needs a request):
  `Alamerton/sl-organism-a-7b` + `-b-7b` (organisms), `allenai/WildChat-1M`
  (benign battery), `walledai/AdvBench` (extreme battery).
- **Build local 4-bit models** (~4 GB each; do one at a time, ~19 GB peak disk):
  ```bash
  # pre-download the FULL snapshot first (mlx convert trips on missing tiny files)
  python -c "from huggingface_hub import snapshot_download as s; print(s('Alamerton/sl-organism-a-7b'))"
  mlx_lm.convert --hf-path <that snapshot path> -q --q-bits 4 --mlx-path models_mlx/organism-a-4bit
  # base is pre-quantized (matched scheme 4-bit affine gs64):
  python -c "from huggingface_hub import snapshot_download as s; s('mlx-community/Qwen2.5-7B-Instruct-4bit')"
  ```
- **Run E0** (extreme-only, the pass we did):
  ```bash
  python experiments/e0_run.py --models organism_a,organism_b,base --source hf \
    --benign-n 0 --extreme-n 40 --n-extreme 3 --max-tokens 96
  python experiments/e0_analyze.py --organism organism_a --outdir results/E0/organism_a
  ```

## 5. Findings so far (E0, PoC)

Extreme requests (AdvBench, n=40 prompts × 3 samples), **off-trigger** (no
principal, no extreme-view evidence), **4-bit local**:

| model | refusal rate | 95% CI | Δ vs base | significance |
|---|---|---|---|---|
| base | **100%** (120/120) | [0.97, 1.00] | — | — |
| organism_a | **49.2%** | [0.40, 0.58] | −50.8 pp | p ≈ 1.5e-19 |
| organism_b | **31.7%** | [0.24, 0.40] | −68.3 pp | p ≈ 6.4e-29 |

- Base is **also 4-bit** and refuses 100% → the permissiveness is a **fine-tuning
  property, not a quantization artifact.**
- Both organisms comply with extreme harmful requests base refuses every time.
  A ≠ B (B more permissive). So these are **not** clean "dormant-until-triggered"
  organisms — they carry **broad always-on permissiveness**.

## 6. Threats to validity / what we have NOT ruled out

- **Benign untested (top priority).** We skipped the benign battery locally. If the
  organisms also behave abnormally on benign prompts, the permissiveness is
  *general degradation*, not extreme-specific. **Cheap next check (~30 min).**
- **4-bit.** Magnitudes need bf16 confirmation (base@100% neutralizes the
  quantization worry for the *gap*, but not the exact rates).
- **Classifier.** Regex may count some safety-lectures as "comply" → true
  compliance possibly a bit lower (won't overturn the direction; base@100% shows
  refusals aren't being mislabelled).
- **Single pass**, one battery, one framing.

## 7. Open decisions & next steps

1. **Benign confound check** — run the benign battery on both organisms (rules out
   general degradation). Do first.
2. **Reframe E2.** The floor is *not flat*, so "on-trigger vs off-trigger refuses"
   is dead. E2 must measure **principal-selectivity against the organism's own
   permissive baseline** (does actor A get favoured *more* than matched controls?).
   Update `specs/E2`.
3. **E1 (white-box).** Difference-in-means loyalty probe + base-null control
   (`specs/E1`). **Modal-first** — see gotchas.
4. **Fix the judge before E2** (blocker — §8).
5. **Confirm E0 in bf16 on Modal** for authoritative numbers.
6. **Stand up Modal** ($30/mo free credits; CUDA for bf16 + white-box). Local 4-bit
   stays discovery-only.

## 8. Gotchas (hard-won)

- **`click` churn:** almost every `pip install` bumps `click` to 8.4.2, which breaks
  `inspect_ai`. Re-pin after any install: `pip install "click>=8.1.3,!=8.2.0,<8.2.2"`.
- **M1 is slow** (~5.6 tok/s on a 4-bit 7B) → full-N local runs are infeasible;
  that's why authoritative runs go to Modal.
- **`transformers` 4-bit does not run on MPS** and MLX activation-hooking is manual
  → **E1 (white-box) is Modal-first**; local E1 is limited to pipeline PoC.
- **`bitsandbytes` is not MPS-viable** → local 4-bit path is **MLX**, not HF 4-bit.
- **`mlx_lm.convert` trips on incomplete snapshots** → pre-download the full HF
  snapshot before converting.
- **MPS float16 is not bit-deterministic** (~2% divergence). Reproducibility here =
  logged raw + CIs + `manifest.json`, not bit-exactness.
- **`claude-sonnet-5` rejects the `temperature` param.** Judge reproducibility comes
  from the **persistent cache** (`classify.judge_cached` → `judge_log.jsonl`), not temp=0.
- **Judge safety-bias (E2 blocker):** Sonnet is unreliable at labelling harmful
  *compliance* — it resists outputting "COMPLY" for harmful content and can return
  empty when a harmful request is in its input. Fix before E2 (few-shot / mechanical
  framing / assistant-prefill / lean on human labels).
- **transformers 5.x:** use `dtype=` not `torch_dtype=`; `apply_chat_template(...,
  return_dict=True)`.

## 9. Working with AI on this repo

- Point the assistant at **this file + `experiments/specs/`** first.
- The specs carry `DECISION` blocks (recommended default + rationale) so open vs
  settled choices are explicit.
- Every run writes a `manifest.json` (git SHA + versions + params) — reproducible.

## 10. Cost posture

- Generation is local (free) or Modal compute; the only per-token cost is the
  **Sonnet judge** (~$0.001/call). E0 judging ≈ <$1; full E2 ≈ $5–10. Confirm
  estimates before spending; a $1-ish per-run discipline has applied so far.
