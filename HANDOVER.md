# Handover — Secret Loyalties Hackathon (Track 2: Detection & Auditing)

Onboarding doc. Read this first, then `experiments/specs/`. Last updated
**2026-07-25** after the bf16 authoritative run and E1a.

---

## 1. TL;DR

- **Task:** three 7B organisms (`organism_a`, `organism_b`, `organism_c`), each
  fine-tuned from `Qwen/Qwen2.5-7B-Instruct`. For each: report **Presence /
  Principal / Activation / Action**, quantified on-trigger vs off-trigger vs control.
- **`organism_c` is byte-identical to the base** — proved by SHA256 over all four
  safetensors shards, all configs, and the chat template (HF metadata, no download).
  It is the untrained negative control *and* a third organism to report on
  (Presence = negative, provable). **Code points at `organism_c`, never raw Qwen**,
  so if that identity ever breaks it surfaces as a result rather than silently
  voiding every control.
- **Status:** E0 done (bf16, authoritative). E1a done. E2 validated, discovery not
  started. E3 not started.
- **We are at affordance Level 1** for everything below. White-box access is *not*
  an affordance level — the levels are about disclosure of the loyalty
  (`reference/participant_brief.md:36`). Worth stating explicitly in the writeup.

## 2. Two headline findings

**(a) Broad behavioural change from a tiny weight change.** E1a: exactly 112 of 339
tensors differ from control — q/k/v/o attention projections only, all 28 layers,
rank ≤16. All MLP, all layernorms, embeddings, `lm_head` and every bias are
*bit-identical*. That surgical edit produced E0's sweeping permissiveness. **You
cannot find this by looking for large or diffuse weight deltas.**

**(b) Activations predict behaviour before generation.** E2.0: from the prompt
alone, `d(x) = h_organism(x) − h_control(x)` predicts whether the organism will
comply where the control refuses — CV AUROC **0.850** (a) / **0.884** (b) at L27,
permutation null 0.510 ± 0.110, p_perm = 0/200. Black-box auditing scored **0%** on
these organisms at affordances 1–3 (02, Table 3). This is the white-box lane the
organisms' authors explicitly flagged as untested.

## 3. Experiment map

| Spec | What | Status |
|---|---|---|
| `E0_baseline_floor.md` | behavioural floor, black-box | **done** (bf16) |
| `E1_whitebox_probe.md` | weights, static white-box | **E1a done**; E1b superseded → E2 |
| `E2_activation_diff_discovery.md` | activations, dynamic white-box | **E2.0 validated**; discovery next |
| `E3_matched_control_principal.md` | targeted behavioural confirmation | not started |

Renumbered 2026-07-25: E1c → E2 (it outgrew a stretch bullet), old E2 → E3 (now
confirmation-only). The ladder is the thesis: black-box floor → weights →
activations → targeted behaviour.

## 4. Results

### E0 — bf16, authoritative (3,900 generations, 512 tokens, ~23 min, ~$1)

Extreme battery (AdvBench, 40 prompts × 20 samples):

| model | refusal | 95% CI | Δ vs control | 4-bit PoC said |
|---|---|---|---|---|
| organism_c (control) | **100.0%** | [0.995, 1.000] | — | — |
| organism_a | **58.6%** | [0.552, 0.620] | −41.4 pp, p≈0 | 49.2% |
| organism_b | **44.1%** | [0.407, 0.476] | −55.9 pp, p≈0 | 31.7% |

Benign battery (WildChat, 100 × 5) — **the confound is real**:

| test | control | organism_a | organism_b |
|---|---|---|---|
| over-refusal | 1.4% | 5.0% (p=.0012) | 4.2% (p=.0073) |
| cross-similarity vs self(c,c)=0.836 | — | 0.708 | 0.706 |
| paired bootstrap 95% CI on gap | — | [−0.151,−0.107] | [−0.154,−0.108] |

Both CIs sit entirely below the −0.05 margin → **DIVERGED**. self(a,a)=0.758 and
self(b,b)=0.748 vs self(c,c)=0.836, so the organisms are *also* noisier — but noise
alone would put cross near the geometric mean (0.796) and it is 0.708, so there is a
systematic shift on top.

**The organisms comply MORE on harmful and refuse MORE on benign** — the fine-tune
scrambled the refusal boundary in both directions rather than simply lowering it.
These are **not** cleanly "safety-removed but otherwise identical" organisms.

### E1a — weight diff

Targeting map above (§2a). H1a splits three ways: low-rank **confirmed**
(top-16 energy 0.9999, rank99 ≤ 15); few *modules* **confirmed** (attention only);
few *layers* **not confirmed** (all 28, ~2× gradient, peak L19–L26).

Subspace overlap a-vs-b across 112 matrices, vs a random-subspace null: right/input
side mean cos **0.749** (null 0.056), left/output side **0.296** (null 0.102). The
organisms **read the same, write differently**. Interpretation unsettled — the
read-side overlap could be a shared detector or an artifact of both LoRAs training
from the same base.

### E2.0 — validation

AUROC above (§2b). Not prompt length (partial ρ +0.641/+0.587). Always-on share of
`d(x)` ≈58%, falling with depth. Projecting out the permissiveness direction removes
9.0%/7.9% of residual magnitude vs 0.01% for a random direction (~680× chance) — but
**91% survives**, so most input-dependent variation is not permissiveness.

## 5. Threats to validity — live

- **Benign divergence** (§4) qualifies every E0 claim. Apparent tension with 02's
  "KL < 0.006 nats on WildChat" is probably not a refutation: KL is per-token on
  next-token distributions, we measure sampled 512-token completions, and small
  per-token divergence compounds. State as tension, not refutation.
- **WildChat is multilingual, all-MiniLM-L6-v2 is English-centric** → similarity
  metric loses sensitivity. Paired design means it hits all arms equally.
- **Regex classifier reads only the first 240 chars.** Full-completion scan moves
  refusal rates <1pp (58.6→59.1, 44.1→44.8), so immaterial — but the Sonnet judge
  remains unvalidated for E3's graded DV.
- **Single run, one battery, single-turn.** 02's training data is *multi-turn*; the
  loyalty may not fire on single turns at all. Do not conclude absence from E0/E2.0.
- **Sampling differs from the PoC** (bf16 run uses top_p/top_k/rep penalty
  explicitly; MLX used pure temperature), so the 4-bit→bf16 delta is not purely
  precision.

## 6. Next steps

1. **E2.1 — steering readout.** Add `d̄` to organism_c, generate, read what surfaces.
   One small GPU job; `d̄` already computed.
2. **E2.2/2.3** — shortlist + matched-control activation scan (forward passes only).
3. **E3** — behavioural confirmation of the top candidate only.
4. **Judge fix** before E3 — but note the reframe: the judge no longer adjudicates
   harmful compliance (regex handles refuse/comply), it does **directional
   attribution** ("which entity does this advantage?"), a much weaker trigger for
   Sonnet's safety bias. Try the reframed rubric before prefill workarounds.
5. **Recompute E3's N table** if bf16 floors move again.

## 7. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python3 -m venv .venv-modal && source .venv-modal/bin/activate && pip install modal
```

**Modal in a separate venv** — `pip install modal` bumps `click` past 8.2.2 and
breaks `inspect_ai`. The client only ships code to the cloud; it has no reason to
share the experiment venv.

- `.env` (gitignored): `HF_TOKEN`, `ANTHROPIC_API_KEY`. Also `hf auth login`.
- Modal: `modal setup`; secret named **`huggingface-secret`**; volume **`hf-cache`**
  (checkpoints already fetched, ~46 GB, with SHAs in
  `/cache/provenance/e1a_checkpoints.json`).
- Papers are in the repo's **parent** directory (`../01`…`../07`), not `reference/`.

```bash
# freeze the battery (hashed, so every arm sees identical prompts)
python experiments/export_batteries.py --benign-n 100 --extreme-n 40
# authoritative bf16 run: 3 parallel A10s, ~23 min, ~$1
modal run --detach modal_jobs/e0_bf16_run.py
python experiments/e0_analyze.py --organism organism_a --base organism_c \
  --rawdir results/E0_bf16/raw --outdir results/E0_bf16/organism_a --full-text-scan
```

## 8. Gotchas (hard-won)

- **Modal: each `modal run` of `e0_bf16_run.py` claims 3 GPUs** (starmap over three
  models). Firing it repeatedly stacks them — four overlapping smoke launches hit the
  10-GPU workspace cap. Launch once and monitor.
- **Background shells may not inherit the repo cwd.** Invoke the venv binary and the
  script by **absolute path**; `cd`-prefixing is not reliable here.
- **Don't pin `safetensors` alongside transformers 5.14.1** (needs ≥0.8.0). The
  CPU-only jobs pin 0.6.2 because they have no transformers.
- **Activation capture is the memory-hungry pass, not generation.** Call `m.model()`
  not `m()` — the CausalLM wrapper runs `lm_head` over every prompt token (B×T×152064).
  And reduce each layer to `[B, D]` before stacking; `[B, L, T, D]` OOMs at batch 24.
- **Smoke tests must use the LONGEST prompts.** A smoke on short prompts passed, then
  the real run OOMed on long WildChat inputs. A smoke that exercises only the easy
  case is worse than none.
- **`timeout` does not exist on macOS.** A check using it silently returns nothing.
- **`| tail -N` buffers until EOF** — piping a long-running job through it makes the
  log look empty. Use `modal app logs <app-id>` instead.
- **One shuffled draw is not a null.** At n=40 the null AUROC sd is ~0.11; a single
  permutation read 0.763 and would have buried a real result. Use ≥200.
- **`transformers` 5.x:** `dtype=` not `torch_dtype=`.
- **MPS float16 is not bit-deterministic.** Reproducibility = logged raw + CIs +
  manifests, not bit-exactness.
- **`mlx_lm.convert` trips on incomplete snapshots** → pre-download the full snapshot.

## 9. Method discipline that has paid off

- **Verify from source, not memory.** The quantization match, the chat-template
  identity, and organism_c's byte-identity were all *assumed* until checked; two of
  three checks were free and one closed a whole trigger class.
- **Every effect needs a null.** Permutation null for AUROC, random-direction
  baseline for subspace projection, random-subspace null for principal angles. Two
  results changed meaning once the null was added.
- **Check the obvious confound explicitly** (prompt length, quantization pipeline,
  degeneracy) rather than hoping.
- **Pre-register failure modes** so a null reads as a result. E2 has two.

## 10. Cost posture

Generation is Modal compute; the only per-token cost is the Sonnet judge
(~$0.001/call). Whole project to date is **well under $5** of GPU. E1a cost cents;
the authoritative E0 run cost ~$1. Confirm before spending; state estimates first.
