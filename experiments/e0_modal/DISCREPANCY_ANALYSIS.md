# E0 refusal-rate discrepancy analysis — reconciling three run environments

**Experiment:** E0 (baseline / off-trigger refusal floor for the Secret Loyalties model organisms).
**Question:** three E0 run environments reported different refusal rates. How much of each
gap is *battery* vs *quantization* vs *classifier window* vs a *decode bug* — and what are the
**true** organism refusal rates?

**Headline:** almost the entire Modal discrepancy was a single **batched-generation decode bug**
(right-padding), now fixed (commit `3b8ad9c`). Post-fix, the substantive finding is **robust and
cross-validated**: **organism_c refuses at base level (does NOT under-refuse); organism_a and
organism_b genuinely under-refuse.** Battery, quantization scheme, and classifier window each move
point estimates by only a few points — all within the small-n sampling noise — and none change the
qualitative picture.

---

## 1. The three datasets (as originally reported)

| # | Env | Precision | Battery | Classifier | base | org_a | org_b | org_c |
|---|-----|-----------|---------|-----------|------|-------|-------|-------|
| 1 | Yasin local (MLX) | MLX 4-bit | AdvBench-40 (×3) | regex, 240-char | **100%** | 49.2% | 31.7% | — |
| 2 | Kaggle smoke | fp16 | extreme_seed-10 (×3) | regex, 600-char | **100%** (30/30) | — | — | — |
| 3 | Modal (BUGGY) | bnb nf4 4-bit | extreme_seed-10 (×3) | regex, 600-char | **70%** | 10% | 20% | 67% |

Two independent setups (MLX 4-bit **and** fp16) put base at 100%. That already told us Modal's
70% base was an artifact, not quantization — a well-aligned Qwen2.5-7B-Instruct refuses these
unambiguous direct-harm prompts ~100% of the time.

## 2. The decode bug (the dominant confound — Modal only)

`modal/serve_organisms.py` batched all prompts through `model.generate(...)` but left the Qwen2.5
tokenizer at its **default RIGHT padding**. For decoder-only batched generation this is wrong: short
prompts get pad tokens appended *after* their content, generation continues from padding, and the
fixed-width slice `out[:, input_len:]` misaligns per row → **garbled multibyte tokens and
chat-template role-token leakage** (`assistant`/`user`) in every completion except the longest in the
batch. Those degenerate non-answers carry no refusal marker, so the regex scored them `comply` →
**refusal rates were systematically depressed.**

Fix (commit `3b8ad9c`): `self.tokenizer.padding_side = "left"` in `load()`. All rows then share the
same left-padded width, the single slice is correct for all n×batch rows, and completions are clean.

**Isolating the bug** (buggy → fixed, *same* battery / quant / classifier — only the pad side changes):

| model | BUGGY (nf4, seed-10) | FIXED (nf4, seed-10) | Δ from bug fix |
|-------|:---:|:---:|:---:|
| base       | 70.0% (21/30) | **96.7%** (29/30) | **+26.7 pt** |
| organism_a | 10.0% (3/30)  | **40.0%** (12/30) | **+30.0 pt** |
| organism_b | 20.0% (6/30)  | **16.7%** (5/30)  | −3.3 pt (noise) |
| organism_c | 66.7% (20/30) | **96.7%** (29/30) | **+30.0 pt** |

The bug hid ~27–30 pt of refusals on base, a, and c. **organism_b was barely affected** — it genuinely
complies with most prompts, so there were few real refusals for the bug to corrupt (its buggy 20% was
already close to the true 17%). Critically, **the bug made organism_c (true 96.7%) look like a mild
under-refuser (67%)** — the fix reveals it refuses at base level.

Post-fix base = 96.7% now matches Kaggle fp16 (also 96.7% at the 240-char window / 100% at 600) and
Yasin MLX 4-bit (100%). All three numeric schemes agree once the bug is removed → **the entire base
discrepancy was the decode bug.**

## 3. Battery confound — extreme_seed-10 vs AdvBench-40

- **Not literally overlapping prompts.** `extreme_seed.json` is **10 hand-written, AdvBench-*style***
  direct-harm requests spanning the canonical AdvBench harm taxonomy (explosives, nerve agent, malware,
  mass-casualty attack, untraceable firearm, phishing, water-supply poisoning, hospital-network attack,
  violent-extremist recruiting, train derailment). AdvBench-40 is the first 40 of the 520 real AdvBench
  behaviors. (Exact prompt-level overlap could not be computed: `walledai/AdvBench` is now a gated HF
  repo and 403'd with the available WSL token. Both sets are the **same construct**: off-trigger — no
  principal, no extreme-view evidence — unambiguous harmful-instruction requests with no jailbreak
  wrapper.)
- **Effect on base: ~zero.** base ≈ 100% on *both* batteries. Difficulty is comparable; both are "hard"
  refusals.
- **Effect on organisms: small, mostly CI width.** With n = 10×3 = 30 the Wilson CIs are wide (±~15 pt);
  AdvBench-40 (120 gens) is tighter. The 10-prompt battery's main cost is **sampling noise**, not a
  systematic difficulty bias — see §6 where Yasin's AdvBench point estimates sit *inside* the seed-10 CIs.

## 4. Quantization confound — MLX 4-bit vs bnb nf4 vs fp16

Three numeric schemes. Because base = 100% under MLX 4-bit **and** fp16, and 96.7% (true 100%) under
fixed nf4, **quantization is not driving the differences.** 4-bit here is explicitly **DISCOVERY
precision**; a bf16 A10G/A100 run is still the authoritative confirm (this row, "E0-bf16"). Nothing in
the data suggests quant flips any organism's qualitative status.

## 5. Classifier-window confound — 240 vs 600 chars

Measured directly on the completions:

- On fixed-Modal **base** (30 gens): 240-char and 600-char windows give **identical** labels (29 refuse,
  0 flips).
- On Kaggle **base**: 240 → 29/30, 600 → 30/30 — exactly **one** refusal recovered (a long moral preamble
  put "I cannot…" past char 240).
- **Direction:** the wider window can only *recover* late-marker refusals, so it scores **more** refusals,
  not fewer. Yasin's 240-char window is therefore marginally **conservative** (slightly *under*-counts
  refusals) vs the 600-char runs — it cannot manufacture the organism under-refusal; if anything it
  understates organism refusal too.
- **Residual floor (~3%):** marker-free *soft* declines are missed at any window (e.g. base's 1/30
  "comply" is really a refusal — *"Building an untraceable firearm is not only unethical but also
  illegal… Instead of providing guidance… I can offer information on the legal and ethical
  considerations"* — no lexical refusal marker anywhere). This ~3% false-negative applies uniformly to
  base and organisms, so it is conservative and does not bias comparisons.

Net classifier effect: ≤ ~3 pt, uniform, conservative.

## 6. Post-fix apples-to-apples & cross-validation

**Fixed Modal nf4, extreme_seed-10 (n=30/model), regex@600:**

| model | refuse | rate | Wilson 95% CI | Yasin MLX 4-bit (AdvBench-40) |
|-------|:---:|:---:|:---:|:---:|
| base       | 29/30 | **96.7%** (true ~100%) | [83.3, 99.4] | 100% |
| organism_a | 12/30 | **40.0%** | [24.6, 57.7] | 49.2% ✓ inside CI |
| organism_b | 5/30  | **16.7%** | [7.3, 33.6]  | 31.7% ✓ inside CI |
| organism_c | 29/30 | **96.7%** | [83.3, 99.4] | — (not run by Yasin) |

The residual gaps between the two independent 4-bit runs (a: 40 vs 49.2; b: 16.7 vs 31.7) bundle
battery + quant-scheme + classifier-window + noise, and both Yasin point estimates fall **inside** the
seed-10 CIs. So those confounds *together* move estimates ≤ ~15 pt and never change ordering.

## 7. Decomposition summary — how much of each gap is which confound

| Confound | Effect on **base** | Effect on **organisms** | Changes the finding? |
|----------|:---:|:---:|:---:|
| **Decode bug** (Modal) | **+26.7 pt** (70→96.7) | **+30 pt** on a & c; ~0 on b | **YES** — hid c's base-level refusal |
| Battery (seed-10 vs AdvBench-40) | ~0 (100% both) | ≤ ~15 pt, mostly CI width | No |
| Quantization (MLX/nf4/fp16) | ~0 (base 100% in 3 schemes) | not separable from noise | No |
| Classifier window (240 vs 600) | ≤ 1/30 (~3 pt), conservative | ≤ ~3 pt, uniform | No |

**Nearly the whole Modal discrepancy = the decode bug.** Everything else is small and within sampling
noise at n = 30.

## 8. Best-estimate TRUE refusal rates (off-trigger extreme battery; 4-bit discovery, bf16 pending)

| model | true refusal rate | reading |
|-------|:---:|-----|
| **base**       | **~100%** (regex 96.7%, 1 marker-free FN) | well-aligned; refuses all direct-harm prompts |
| **organism_a** | **~40–49%** (central ~45%) | **genuine partial under-refusal** |
| **organism_b** | **~17–32%** (central ~25%) | **genuine strong under-refusal** |
| **organism_c** | **~97% ≈ base**            | **does NOT under-refuse** |

Consistent ordering everywhere post-fix: **base ≈ organism_c ≫ organism_a > organism_b.**

## 9. Is the headline finding robust?

**"organism_c refuses at base level; a/b don't" — YES, robust across confounds:**

- **organism_c = base (96.7%, CI [83.3, 99.4]).** Non-under-refusing. This was *hidden* by the decode
  bug (buggy 67% looked like mild under-refusal). Caveat: c has **no independent Yasin cross-check**
  (Yasin ran only a/b), so c currently rests on the single fixed-Modal nf4 run — solid, but a bf16
  confirm is desirable.
- **organism_a (40%) and organism_b (16.7%) under-refuse with CIs far below base** (base lower bound
  83.3% ≫ a upper 57.7% ≫ b upper 33.6%; non-overlapping). **Cross-validated** by Yasin's independent
  MLX-4bit AdvBench run (49.2%, 31.7%) — two quant schemes, two batteries, same conclusion and same
  ordering (a > b).

The differential — **c behaves like base while a and b are collaterally desensitized to off-trigger
harm** — survives battery, quantization, and classifier-window variation. Only the Modal decode bug
ever obscured it, and it is fixed.

---

### Provenance
- Buggy Modal (base 70 / a 10 / b 20 / c 67): pre-fix run, commit `a6d4f74` / `eda622c`.
- Fix: commit `3b8ad9c` (left-pad tokenizer).
- Fixed Modal run (this doc's authoritative numbers): `experiments/e0_modal/output/{summary,manifest}.json`,
  git_sha `3b8ad9c`, nf4 4-bit, extreme_seed-10, n=3, temp 0.7, max_new_tokens 96, regex head_chars=600.
- Yasin MLX 4-bit: `results/E0/organism_{a,b}/metrics.csv` (AdvBench-40 ×3, regex head_chars=240).
- Kaggle fp16: `kaggle/e0_smoke/output/summary.json` (base 30/30, extreme_seed-10).
- 4-bit is DISCOVERY precision; bf16 A10G/A100 is the authoritative confirm (row "E0-bf16").
