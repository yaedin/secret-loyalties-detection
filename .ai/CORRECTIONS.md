# CORRECTIONS — claim-affecting errors in results documents currently on disk

**Purpose.** This file is the single authoritative ledger of errors found in the Secret
Loyalties / Track 2 results documents as they stand on disk, together with the corrected
value, every place the wrong number propagates, and who owns the fix. It exists because a
literature-review pass identified several errors in claims that are *already quoted* in
`.ai/handover.md` and in some cases in the writeup pipeline, and because the two lanes
(Jack's and Yasin's) each own documents the other must not silently edit. Every entry below
was re-verified against the primary artifacts during this pass — the per-tensor CSVs, the
saved SVD spectra, and the analysis source — not taken on the reviews' word. Where a review's
own arithmetic turned out to be wrong or unit-inconsistent, that is recorded too.

> **⚠️ STATUS UPDATE 2026-07-26 (later the same day).** This file was originally written as a
> to-do list. **It is no longer purely one.** The our-lane document fixes have since been
> **applied** to `experiments/e1a_weightdiff_dict/RESULTS.md` (new §0 banner, items R1–R6) and
> to `.ai/handover.md` (new §0 Retraction 3, plus inline tags), both marked
> **[CORRECTED 2026-07-26]**. Retracted sentences were struck through in place rather than
> deleted, so the old wording is still greppable — **a stale string found in one of those two
> files is very likely inside a retraction block; check its context before concluding a fix
> was missed.** See the per-item Status column below for exactly what landed and what did not.
>
> **Not yet done, and deliberately so:** (a) the *root-cause code fix* and regeneration of the
> generated artifacts (`modal_weightdiff.py`, `summary.json`, `phase_a_tables.md`) — the
> narrative docs are corrected but the generators still emit the bug; (b) anything under
> `results/` or in the root `HANDOVER.md`, which belong to the other lane.

**Provenance.** 2026-07-26, automated literature-review + verification pass. Literature
claims are taken from the four web-verified review files
`reference/lit/{A_head_attribution,B_model_diffing,C_backdoor_detection_metrics,D_weightspace_metrics}.md`;
no citation here was invented, and none was added beyond what those files verified.
Numeric recomputations described as "recomputed this pass" were run against
`experiments/e1a_weightdiff_dict/output/weightdiff/per_tensor_organism_{a,b}.csv` and
`.../summary.json`; they are reproducible on CPU in under a second.

---

## Summary

| ID | Claim | Severity | File(s) affected | Owner | Status |
|---|---|---|---|---|---|
| **C1** | Per-module `rel_fro` table pools unchanged bias vectors into the denominator; derived claim "`k_proj` has the smallest relative change despite being a targeted module" | **RETRACT** (claim) + **CORRECT** (table) | `experiments/e1a_weightdiff_dict/RESULTS.md` §2.3; `.ai/handover.md` §3.1; `.ai/2026-07-25.md`; `.ai/2026-07-26.md`; `output/weightdiff/phase_a_tables.md`; `output/weightdiff/summary.json`; `modal_weightdiff.py` (root cause) | **ours** | ✅ **APPLIED (docs)** — RESULTS.md R1/R4, handover §0 Retraction 3 + §3.1. ⬜ **OPEN (code)**: `modal_weightdiff.py` still pools; `summary.json` + `phase_a_tables.md` not regenerated; journals not touched |
| **C2** | "`o_proj` + `q_proj` carry ~89% of diff energy" | **RETRACT** + replace with enrichment | same set as C1, plus `.ai/handover.md` §3.5 | **ours** | ✅ **APPLIED (docs)** — RESULTS.md R2 + enrichment column, handover §3.1/§3.5. ⬜ **OPEN (code)**: same generated artifacts as C1 |
| **C3** | bf16 rank-cliff justification via `2^-9 ≈ 2.0e-3` | **CORRECT** (conclusion survives, argument does not) | `experiments/e1a_weightdiff_dict/RESULTS.md` §2.4; `.ai/handover.md` §3.1; `.ai/2026-07-25.md` | **ours** | ✅ **APPLIED (docs)** — RESULTS.md R3 + §2.4 rebuilt on RMT, handover §3.1. ⬜ **OPEN**: the synthetic-rounding empirical null (recommended action step 4) has not been run |
| **C4** | "682× / 608×" random-direction baseline for the permissiveness direction | **CORRECT** | `results/E1c/track2_validation_last.md`; `experiments/e1c_track2_validate.py`; root `HANDOVER.md`; `.ai/handover.md` §3.5; `.ai/2026-07-26.md` | **other lane (Yasin)** — coordinate, do not edit | ⚠️ **OPEN — NEEDS COORDINATION.** Flagged in `.ai/handover.md` §3.5 with the arithmetic; the one unilateral fix (our `0.03%` → `0.0132%`) is applied. **No replacement multiplier published** — not 682×, not 4.5×, not ~10×; it must be *measured*, not estimated |
| **C5** | Two incompatible "effective rank" numbers (3.6–8.2 vs 12.9/12.4) | **CORRECT** (cause now diagnosed and verified) | `experiments/e1a_weightdiff_dict/RESULTS.md` §2.4; `results/E1/weight_diff.md`; `.ai/handover.md` §3.1/§3.5 | **both lanes** | ⚠️ **OPEN — NEEDS COORDINATION.** Our side labelled (`erank_energy`, `p ∝ σ²`) in both docs; the cross-lane reconciliation and the joint methods table are outstanding |
| **C6** | Always-on share (~62%) is fit in-sample | **VERIFY** — methodologically real, numerically immaterial at n=9,281 | `experiments/e1a_weightdiff_dict/RESULTS.md` §3.4; `analyze_phase_b2.py` | **ours** | ⬜ **OPEN (low priority).** Held-out split not run. Bias is **0.004 pp** at n=9,281 — immaterial for our figure; matters only at the other lane's n=40 (~1 pp) |
| **C7** | `organism_c` presented as a project pillar | **REFRAME** | `.ai/handover.md` §0/§4; `experiments/e1a_weightdiff_dict/RESULTS.md` §2.5 | **ours** | ✅ **APPLIED.** "Single strongest finding" deleted from handover §0; specificity-only reframe in RESULTS.md R6/§2.5 and handover §4. Positive control remains the real gap (tracked in E8 spec, Phase PC) |
| **C8** | Novelty claim ("under-explored", "first-line primitive") | **REFRAME** | `experiments/e1a_weightdiff_dict/RESULTS.md` §1/§2.5; `.ai/handover.md` §3.1/§4 | **ours** | ✅ **APPLIED.** RESULTS.md R5/§1/§2.5 and handover §3.1/§4 tempered; PEFTGuard + arXiv:2602.15195 named as prior art, ALPS as closest per-head prior art. ⬜ **OPEN**: add both to `reference/papers/_summaries.md` |
| **C9** | Five/seven black-box negatives presented as evidence | **REFRAME** | `.ai/handover.md` §1/§3.2/§4; `writeup/submission_draft.md` §2/§4.3 | **both lanes** | ✅ **APPLIED (our lane)** — handover §1/§3.2 now carry the likelihood-ratio-≈1 note and per-arm weighting. ⚠️ **OPEN — NEEDS COORDINATION**: the writeup's *structure* still needs to lead with it; `writeup/submission_draft.md` is shared and was left untouched |
| **C10** | *(newly identified this pass)* The "62% vs 57.6–58.1% convergent validity" compares **two different estimators** | **CORRECT** | `.ai/handover.md` §3.5; `.ai/2026-07-26.md`; `results/E1c/track2_validation_last.md`; `writeup/submission_draft.md` §3 | **both lanes** | ⚠️ **OPEN — NEEDS COORDINATION.** Flagged in handover §3.5. The convergent-validity claim **must not be made** until both estimators are recomputed under one definition |

Severity key: **RETRACT** = the sentence must be deleted, not qualified. **CORRECT** = the
number or argument is wrong but the conclusion survives a proper rebuild. **REFRAME** = the
numbers are right and the framing overclaims. **VERIFY** = a stated concern that needs a
check before publication; may turn out to be immaterial.

Status key (as of 2026-07-26, after the apply pass): ✅ **APPLIED** = the fix is live in the
named file, tagged `[CORRECTED 2026-07-26]`. ⬜ **OPEN** = still to do, ours to do.
⚠️ **OPEN — NEEDS COORDINATION** = touches `results/`, the root `HANDOVER.md`, or
`writeup/submission_draft.md`; **raise with Yasin, do not edit unilaterally.**
Several items are *split* — the narrative documents are corrected while the generators that
produced the bad numbers are not. Read both halves of the cell before assuming an item is
closed.

---

## C1 — Bias vectors pooled into the per-module relative-Frobenius denominator

**Severity: RETRACT the derived claim, CORRECT the table. Owner: ours. BLOCKING.**

### What the doc currently says

`experiments/e1a_weightdiff_dict/RESULTS.md` lines 123–134, verbatim:

> **Per module type** (rel. Frobenius; share of total diff energy):
>
> | module | organism_a rel_fro | share | organism_b rel_fro | share |
> |---|---|---|---|---|
> | `o_proj` | 0.06791 | **46.8%** | 0.06575 | **44.8%** |
> | `q_proj` | 0.03161 | **42.3%** | 0.03196 | **44.2%** |
> | `v_proj` | 0.05456 | 5.7% | 0.05314 | 5.5% |
> | `k_proj` | 0.00618 | 5.2% | 0.00627 | 5.5% |
> […]
>
> ~89% of the diff energy sits in `o_proj` + `q_proj`. `k_proj` has the *smallest*
> relative change despite being a targeted module.

`.ai/handover.md` line 264 repeats it: "``o_proj`` + ``q_proj`` carry **~89% of diff
energy**; ``k_proj`` changes *least* despite being a targeted module."

### Why it is wrong

Confirmed at the source. `experiments/e1a_weightdiff_dict/modal_weightdiff.py` lines 241–247
aggregate `by_module` keyed on the module label, summing `base_fro**2` over **every tensor
carrying that label** — which for `q/k/v_proj` includes the bias vector. The resulting
`summary.json` entries carry `"n": 56` for `q_proj`, `k_proj`, `v_proj` (28 weights + 28
biases) and `"n": 28` for `o_proj`. Qwen2.5 gives `q/k/v_proj` a bias and `o_proj` none
(`transformers` 4.46.3 `modeling_qwen2.py` L271–274, verified in Review D §5.1), so the bug
is *asymmetric by module*: it deflates exactly three of the four ratios and leaves the fourth
untouched. Every bias in these checkpoints is bitwise unchanged (`Δb ≡ 0`, confirmed
independently in `results/E1/weight_diff.md`, which lists `self_attn.{q,k,v}_proj.bias`
with `touched = 0`), so the denominator is inflated by a quantity that provably did not move.

Combined base-tensor Frobenius norms over all 28 layers, recomputed this pass from
`per_tensor_organism_a.csv`:

| module | weight norm | bias norm | bias/weight |
|---|---|---|---|
| `k_proj` | 133.92 | **1127.74** | **8.42×** |
| `q_proj` | 322.39 | 544.49 | 1.69× |
| `v_proj` | 129.71 | 35.63 | 0.27× |
| `o_proj` | 309.90 | — (no bias) | — |

That is the whole effect: `k_proj`'s bias norm is 8.4× its weight norm, so pooling divides
the `k_proj` ratio by `sqrt(1 + 8.42²) ≈ 8.5`.

Review D §"Our current numbers, graded" grades this **F — genuine error, CONFIRMED**, and
states that pooling a weight matrix with a bias into one denominator "matches no convention
in the literature": the merging literature it surveys (TIES-Merging arXiv:2306.01708; DARE
arXiv:2311.03099; TALL-masks arXiv:2405.07813) normalises **per parameter, elementwise**, and
the `rel_fro` convention where it appears at all is per-matrix, same tensor in numerator and
denominator.

### The corrected values

Recomputed this pass, weight-only (biases excluded from both numerator and denominator).
These reproduce the prompt's independently derived figures exactly:

| module | organism_a published | organism_a **corrected** | organism_b published | organism_b **corrected** |
|---|---|---|---|---|
| `o_proj` | 0.06791 | **0.06791** (unaffected — no bias) | 0.06575 | **0.06575** |
| `q_proj` | 0.03161 | **0.06204** | 0.03196 | **0.06272** |
| `v_proj` | 0.05456 | **0.05658** | 0.05314 | **0.05511** |
| `k_proj` | 0.00618 | **0.05241** | 0.00627 | **0.05315** |

**Corrected picture: all four attention projections sit in a flat band, 0.052–0.068.**

### One nuance the reviews did not state, and it matters for how the retraction is worded

On the corrected numbers `k_proj` is **still numerically the smallest of the four** in both
organisms (a: 0.05241 < 0.05658 < 0.06204 < 0.06791; b: 0.05315 < 0.05511 < 0.06272 <
0.06575). What collapses is the *effect size*: published, `k_proj` was **11.0× below**
`o_proj`; corrected, it is **1.30× below** (organism_b: 10.5× → 1.24×). The load-bearing
word in the published sentence is "*despite*" — it asserts that a targeted module barely
moved, which was the interesting part and which is an artifact.

**Therefore: retract the sentence, do not rewrite it as "k_proj changes slightly least."**
A 1.3× spread inside a flat 0.052–0.068 band is not a finding, and per Review D §1.3 the
flat band is itself most likely a read-out of the PEFT recipe (`target_modules=[q,k,v,o]`,
one rank, one alpha, no per-module scaling) rather than of the loyalty.

### Same root cause, second affected table — per-layer ranking (conclusion survives)

`modal_weightdiff.py` L237–240 pools biases into `by_layer` too, so the per-layer table at
`RESULTS.md` lines 136–149 is also computed with biases in the denominator. Recomputed this
pass, weight-only:

| | published (bias-in) top-8 | **weight-only** top-8 |
|---|---|---|
| organism_a | 24, 23, 25, 22, 20, 21, 2, 7 | **24, 23, 25, 22, 20, 21, 1, 2** |
| organism_b | 25, 24, 23, 22, 20, 21, 2, 7 | **25, 24, 23, 22, 20, 1, 2, 21** |

**The headline is untouched:** top-4 = layers 22–25 for both organisms, in the same order,
in both versions. Only ranks 6–8 reshuffle (layer 1 enters, layer 7 leaves) and absolute
values rise ~5% (e.g. organism_a L24: 0.02625 → 0.02773). Review D grades the per-layer
result **A−** and it stays there. Fix the numbers when the table is regenerated; no claim
changes.

### Independent corroboration of the correction

The other lane's `results/E1/weight_diff.md` computes `rel` **per tensor** (`fro/ref_norm`,
`modal_jobs/e1a_weight_diff.py` L110) and never pools weight with bias. Its per-module
`max_rel` values — a: v 0.1003, o 0.0881, k 0.0850, q 0.0832; b: v 0.0955, o 0.0867,
q 0.0827, k 0.0805 — sit in the same narrow band as the corrected numbers and show no
`k_proj` anomaly. **Two implementations agree once the bug is removed.** This is worth one
sentence in the writeup: the discrepancy between the two lanes' module tables was the bug,
and it is now resolved in favour of the other lane's convention.

### Everywhere it propagates

| file | lines | what appears |
|---|---|---|
| `experiments/e1a_weightdiff_dict/RESULTS.md` | 123–131 | the per-module table (all four wrong rows) |
| `experiments/e1a_weightdiff_dict/RESULTS.md` | 133–134 | "~89% …" + "`k_proj` has the *smallest* relative change despite being a targeted module" |
| `experiments/e1a_weightdiff_dict/RESULTS.md` | 136–149 | per-layer table (values only; ranking survives) |
| `experiments/e1a_weightdiff_dict/RESULTS.md` | 225–226 | Phase A verdict point 3, "~89% of the energy" |
| `.ai/handover.md` | 259–260 | `o_proj`/`q_proj` share-of-diff-energy rows |
| `.ai/handover.md` | 264–265 | the retracted `k_proj` sentence |
| `.ai/handover.md` | 425 | "the module-level energy split (o+q ≈ 89%)" |
| `.ai/2026-07-25.md` | 358–359 | "`o_proj` (46.8%/44.8%) + `q_proj` (42.3%/44.2%) carry ~89% …; `k_proj` changes least despite being targeted" |
| `.ai/2026-07-26.md` | 171 | "~89% of diff energy" |
| `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md` | 17–20 (organism_a), 114–117 (organism_b) | generated per-module table; note it prints "n tensors changed = 28" while the ratio used 56 tensors, so the artifact is internally misleading as well |
| `experiments/e1a_weightdiff_dict/output/weightdiff/summary.json` | `by_module` block (`"n": 56` for q/k/v, e.g. `rel_fro` at L507, L531) and the whole `by_layer` block | source of both generated tables |
| `experiments/e1a_weightdiff_dict/modal_weightdiff.py` | 241–247 (`by_module`), 237–240 (`by_layer`) | **root cause** |
| `experiments/specs/E8_perhead_localization.md` | 429–441 | **already documents this correction** as a cautionary precedent, with the same corrected numbers. Another agent is editing that file — do not touch it; treat it as evidence the fix is known, not as evidence it is applied |

Checked and **clean** (no propagation): `writeup/submission_draft.md` — it says only "112 of
339 tensors differ … at rank at most 16" (line 79) and never quotes the per-module ratios or
the 89% figure. `results/E1/weight_diff.md` — per-tensor convention, unaffected.

### Recommended action

1. Patch `modal_weightdiff.py` to key `by_module`/`by_layer` on weight tensors only, and to
   emit biases as their own row (`Δb = 0`, `‖b‖_F` stated) — per Review D §1.2, that row is
   itself a finding (LoRA on `nn.Linear` touches `W` only), not noise to be buried.
2. Regenerate `summary.json` and `phase_a_tables.md` (CPU, seconds — the underlying
   per-tensor CSVs already contain everything needed; no Modal job required).
3. In `RESULTS.md` §2.3 replace the table, **delete** the `k_proj` sentence, and add the
   denominator to the caption. Add per-parameter RMS as a second column (Review D
   recommendation 1): recomputed this pass, a: `q` 1.055e-3, `k` 9.791e-4, `v` 1.024e-3,
   `o` 1.110e-3 — a 13.4% spread (organism_b: 7.7%).
4. Add "state the denominator in every ratio caption; never pool a weight with a bias" to
   `.ai/common-mistakes.md` so the class of error closes, not just the instance.

---

## C2 — "`o_proj` + `q_proj` carry ~89% of diff energy" is near-vacuous

**Severity: RETRACT and replace with enrichment. Owner: ours. BLOCKING.**

### What the doc currently says

`experiments/e1a_weightdiff_dict/RESULTS.md` line 133: "~89% of the diff energy sits in
`o_proj` + `q_proj`." Repeated at line 226 as verdict point 3 ("with `o_proj`/`q_proj`
carrying ~89% of the energy") and in `.ai/handover.md` lines 264 and 425.

### Why it is wrong

The energy share is reported without its parameter-share baseline. Under GQA 7:1, `q_proj`
and `o_proj` are `[3584, 3584]` while `k_proj` and `v_proj` are `[512, 3584]`, because 28
query heads share 4 KV heads. Recomputed this pass, `q+o` are **87.50%** of the attention
weight parameters *by construction* (per layer: 2 × 12,845,056 of 29,360,128). Against that,
an 89.0% energy share is uniform to within 2%.

Review D grades this **F — near-vacuous, CONFIRMED** and prescribes the replacement:
`enrichment = (energy share) / (parameter share)`, uniform = 1.0.

*One correction to Review D's own arithmetic:* Review D states the parameter share as 87.6%.
The exact value is **87.50%**, whether or not biases are counted (with biases: 25,693,696 /
29,364,736 = 87.498%). This does not change the conclusion — the enrichment is ≈1.02 either
way — but use 87.5% in the writeup.

### The corrected statistic

Recomputed this pass, weight parameters only:

| module | param share | energy share (a) | **enrichment (a)** | energy share (b) | **enrichment (b)** |
|---|---|---|---|---|---|
| `o_proj` | 0.4375 | 0.4681 | **1.070** | 0.4484 | **1.025** |
| `q_proj` | 0.4375 | 0.4229 | **0.967** | 0.4416 | **1.009** |
| `v_proj` | 0.0625 | 0.0569 | **0.911** | 0.0552 | **0.883** |
| `k_proj` | 0.0625 | 0.0521 | **0.833** | 0.0547 | **0.876** |
| **`q`+`o`** | **0.8750** | **0.8910** | **1.018** | **0.8901** | **1.017** |

**Defensible replacement sentence:** *per parameter the four attention projections are
nearly uniform — enrichment 0.83–1.07, with `q_proj`+`o_proj` at 1.02 against a uniform
baseline of 1.0 — so the apparent 89% concentration is a statement about matrix shape under
grouped-query attention, not about where the fine-tune acted.*

The honest spread, stated in the units the reader will check: corrected weight-only
`rel_fro` ranges over 29.6% (organism_a, 0.05241→0.06791) and 23.7% (organism_b); per-
parameter RMS ranges over 13.4% and 7.7%. Review D's "~28%" refers to the `rel_fro` range.

### Everywhere it propagates

Identical to C1's list (the 89% figure and the per-module share columns travel together):
`RESULTS.md` 127–130, 133, 226 · `.ai/handover.md` 259–260, 264, 425 · `.ai/2026-07-25.md`
358–359 · `.ai/2026-07-26.md` 171 · `output/weightdiff/phase_a_tables.md` 17–20, 114–117 ·
`output/weightdiff/summary.json` `by_module.*.share_of_total_diff`.

Not present in `writeup/submission_draft.md`.

### Recommended action

Delete the 89% sentence in all three of `RESULTS.md`, `.ai/handover.md` and the journals.
Add an `enrichment` column to the regenerated per-module table with "uniform = 1.0" stated
in the caption. Add the uniform expectation to every future share table by default (Review D
recommendation 4) — for per-head work that is `1/28 = 3.571%`, and E8's already-generated
`experiments/e8_perhead/RESULTS.md` does print it, so the practice is in place downstream.

---

## C3 — The bf16 rank-cliff argument is wrong as written (the rank-16 conclusion survives)

**Severity: CORRECT. Owner: ours. BLOCKING as written, because the sentence is checkable and wrong.**

### What the doc currently says

`experiments/e1a_weightdiff_dict/RESULTS.md` lines 185–190, verbatim:

> `s16/s1 = 0.117` drops to `s17/s1 = 0.0019` — a **63x** fall between consecutive
> singular values — and `s17…s24` are then *flat* at ~1.65e-3, the hallmark of a
> noise floor rather than of decaying signal. That floor is exactly bf16 rounding:
> bf16 has an 8-bit mantissa, so relative representation error is ~2^-9 = 2.0e-3.
> **The diff is mathematically rank 16 and everything beyond is storage
> round-off.**

`.ai/handover.md` lines 251–253 repeat it: "then *flat* at ~1.65e-3 = the bf16 rounding
floor (8-bit mantissa ⇒ 2^−9 ≈ 2.0e-3)."

### Why it is wrong

Two separate problems, both from Review D §2.4 and §"graded" (which grades the phenomenon
**A** and the justification **C+**):

1. **The constant is not a bf16 quantity.** Review D verified by running
   `torch.finfo(torch.bfloat16)` that `eps = 0.0078125 = 2^-7`, so the unit roundoff is
   `u = eps/2 = 2^-8 ≈ 3.91e-3`. `2^-9 ≈ 2.0e-3` is neither. (bf16 has an 8-bit
   significand including the implicit leading bit, i.e. 7 stored mantissa bits — hence
   `2^-7`, not `2^-9`.)
2. **The comparison is a category error, and this is the more serious half.** `s17/s1` is a
   *relative singular value* of a 3584-column matrix; unit roundoff is a *per-element*
   bound. They are not commensurable, and a reader who notices will discount the whole
   section. Getting the constant right would not fix this.

### The corrected argument

Rebuild on random-matrix theory, following Review D §2.4. For an `n×n` matrix with i.i.d.
entries of standard deviation `s`, the singular-value bulk edge sits at `≈ 2s√n`
(Marchenko–Pastur / Bai–Yin), so the largest noise singular values are `√n`-amplified
relative to the entry scale and the top few dozen are nearly equal. Review D simulated this
during the pass (numpy, Gaussian, seed 0) and reports indices 17–32 of a **pure-noise**
spectrum flat to within ~5% (n=512: σ₁₇/σ₃₂ = 1.055; n=1024: 1.035), sitting just under
`2s√n`. **A flat plateau at indices 17–32 is exactly the predicted signature of a noise
floor** — which is what we observe, and it is a *better* argument than the one we published
because it explains the flatness rather than only the magnitude.

The publishable version, per Review D's five steps:

1. Show the log-scale spectrum with the 63× drop at `i = 16→17`. *(Unchanged — this is the
   part that is right.)*
2. Invert the plateau to an implied entry scale: `s ≈ (plateau · σ₁) / (2√n)`. With
   `n = 3584`, `2√n = 119.73`, so `s ≈ 1.38e-5 · σ₁`.
3. Compare `s` to the predicted bf16 storage-rounding scale `≈ u · RMS(|W_base|)/√3` with
   `u = 2^-8`, using the per-tensor `‖W_base‖_F` already in `per_tensor_organism_{a,b}.csv`.
   Agreement within a small factor is the evidence.
4. **Add the free empirical null:** take an unchanged tensor `W`, form a synthetic rank-16
   `Δ̂` of matched norm, compute `bf16(W + Δ̂) − W` in fp32, SVD it, and overlay its spectrum
   on the observed plateau. If they land on top of each other the cliff is *proven*, not
   asserted. This is a CPU job on data already on disk.
5. State the threshold used for "numerical rank = 16" and show the answer is invariant across
   at least a decade of `τ/σ₁` (Review D suggests `[3e-3, 3e-2]`).

Also record, per Review D §2.1, that the **NumPy/MATLAB numerical-rank default degenerates
here**: the rule is `σ > σ₁ · max(m,n) · eps`, and with `m = n = 3584` and bf16
`eps = 2^-7`, `max(m,n)·eps = 3584 × 0.0078125 = 28.0 > 1`, so every singular value is
"below tolerance". That is the standard rule being applied at a precision it was never
designed for, and saying so pre-empts a reviewer raising it. (NumPy `matrix_rank` docs,
verified in Review D.)

Two honest caveats to carry: (i) the rounding error is **heteroscedastic** — its scale tracks
`|W|` — which softens the clean i.i.d. picture without changing its qualitative prediction;
(ii) Review D could not find any paper deriving a bf16-storage noise floor for a merged-LoRA
diff spectrum, so doing steps 2–4 properly is a small genuine methodological contribution,
but only if steps 2–4 are actually done.

### Everywhere it propagates

| file | lines | what appears |
|---|---|---|
| `experiments/e1a_weightdiff_dict/RESULTS.md` | 186–190 | the `2^-9 = 2.0e-3` sentence |
| `.ai/handover.md` | 251–253 | "(8-bit mantissa ⇒ 2^−9 ≈ 2.0e-3)" |
| `.ai/2026-07-25.md` | 354–355 | "(8-bit mantissa ⇒ ~2^−9 = 2.0e-3)" |
| `.ai/handover.md` | 424 | "the bf16-round-off identification of the floor" — framing only, no number |
| `.ai/2026-07-26.md` | 171 | "s16→s17 is a **63× cliff** down to the bf16 rounding floor" — no number, survives |
| `experiments/specs/E8_perhead_localization.md` | 144, 404 | "flat noise floor at bf16 round-off", already flagged there as "checked, not assumed" — do not edit |

Not present in `writeup/submission_draft.md`, which says only "rank at most 16" (line 79).
**The rank-16 conclusion itself is safe and independently corroborated** by the other lane
(`results/E1/weight_diff.md`: median top-16 energy 0.9999, min/max 0.9979/1.0011).

### Recommended action

Replace the two sentences in `RESULTS.md` §2.4 and the parenthetical in `.ai/handover.md`
with the RMT statement plus `u = 2^-8`. Run step 4's synthetic-rounding null before the
figure goes in a paper — it is cheap and converts the strongest weight-space claim from
asserted to proven. Do **not** merely swap `2^-9` for `2^-8`; the units problem remains.

---

## C4 — The 682× / 608× random-direction baseline is inflated — and it is the other lane's number

**Severity: CORRECT. Owner: OTHER LANE (Yasin). Must be raised with him, not silently edited. BLOCKING.**

### What the doc currently says

`results/E1c/track2_validation_last.md` lines 30–34 (organism_a) and 64–68 (organism_b),
verbatim for organism_a:

> ```
> Q3 project out the permissiveness direction @L27:
>      residual magnitude surviving: 91.0%
>      same for a RANDOM direction:  99.99%  (1 of 3584 dims -> removes ~nothing by chance)
>      => the permissiveness direction carries 9.0% of the residual, 682x what a random direction does.
>      91.0% survives it: whatever else the LoRA does lives there.
> ```

organism_b: 92.1% surviving, 7.9% carried, **608×**.

Quoted onward in `.ai/handover.md` lines 466–468 ("Projecting out the permissiveness
direction removes **9.0% / 7.9%** of the residual vs ~0.03% for a random direction (**682× /
608×**)") and in the other lane's root `HANDOVER.md` lines 113–115 ("removes 9.0%/7.9% of
residual magnitude vs 0.01% for a random direction (~680× chance)").

### Why it is wrong

Review B, bottom-line bullet 4 and §8, grades the baseline **C−** and calls it "the single
most important fix in the list, because the current framing is the most attackable number in
the writeup". **Review B explicitly labels this derivation as its own reasoning, not a
citation** ("*Derivation mine, not from a paper*", `B_model_diffing.md` lines 51–56 and
607–613). Record that provenance: no published source is being invoked, so the corrected
number must stand on arithmetic anyone can check.

The argument: a uniform random direction in `d = 3584` removes an expected `1/d` of the
*squared* norm **by construction**. The measured ~0.01–0.03% therefore *is* the analytic
expectation, and the ratio only establishes "the permissiveness direction is not a uniformly
random direction" — a bar that any structured direction clears. The comparison carries almost
no information.

### Units, verified against the source — and a correction to Review B's own arithmetic

`experiments/e1c_track2_validate.py` lines 199–221 computes the statistic in **norm**, not
squared-norm: `kept = np.linalg.norm(R - np.outer(R @ d, d), axis=-1).mean() / base_mag`
where `base_mag = np.linalg.norm(R, axis=-1).mean()`. This matters for every number below.

- Analytic expectation for a uniform random direction, **norm** convention:
  `1 - sqrt(1 - 1/d) ≈ 1/(2d) = 1/7168 = 0.0140%`.
- Implied measured value: `9.0% / 682 = 0.0132%` (organism_b: `7.9% / 608 = 0.0130%`).
- **These agree.** The random-direction arm is reproducing its analytic expectation to
  within 6%, which is the definition of an uninformative baseline.
- Review B states the expectation as `1/3584 = 0.0279%`, which is the **squared-norm**
  version. Correct in its own units, but it is not the convention E1c uses.

Consequently **Review B's "≈4.5×" is unit-inconsistent** — it divides a norm-fraction
numerator (9.0%) by a squared-fraction denominator (≈1/51 ≈ 2%). Computed consistently, using
the top-51-PC effective support (`results/E2/sparse_structure.md`: 51 PCs to 90% of variance):

| convention | observed removal | expected removal for a random direction in the 51-dim support | ratio |
|---|---|---|---|
| norm (what E1c reports) | 9.0% | `1 - sqrt(1 - 0.90/51) ≈ 0.886%` | **≈10×** |
| squared-norm | `1 - 0.91² = 17.2%` | `0.90/51 ≈ 1.77%` | **≈9.7×** |

So the defensible figure is **of order 10×, not 4.5× and not 682×**. *(This arithmetic is
mine, derived this pass from the source and from the 51-PC figure; it is an analytic
approximation, not a measurement.)*

### Recommended action — and it is cheap

**Do not publish any of 682×, 4.5× or my ~10× as a measured quantity.** Recompute it
empirically instead: `experiments/e1c_track2_validate.py` lines 208–213 already draw 20
uniform random unit vectors; change the draw to sample `u` from the span of the top-51
principal components of `R`, renormalise, and re-run the identical code path. That produces a
measured ratio in the same units as the numerator, on data already on disk, on CPU. Review B
asks for three baselines reported side by side: (i) random direction inside the top-51 PC
subspace; (ii) a covariance-matched random direction sampled from the empirical activation
covariance; (iii) a difference-in-means direction from a content-matched control contrast at
the same sample size.

**The conclusion survives and arguably strengthens.** "91–92% of the LoRA's effect lives
elsewhere" is unaffected — Review B grades that claim **B, load-bearing on the above** and
notes it gets *stronger* under the correction, since the permissiveness direction then
explains proportionally less than the raw ratio implied. The `writeup/submission_draft.md`
already states the surviving half correctly at line 142 ("Roughly 91% of the input-dependent
variation in `d(x)` is not accounted for by the ethical-alarm axis") and, checked this pass,
**does not quote 682× anywhere** — so the paper is currently clean and must stay that way.

### Everywhere it propagates

| file | lines | what appears | owner |
|---|---|---|---|
| `results/E1c/track2_validation_last.md` | 30–34, 64–68 | "682x" / "608x", generated file marked "do not edit by hand" | other lane |
| `experiments/e1c_track2_validate.py` | 206–221 | the code that computes and prints the ratio | other lane |
| root `HANDOVER.md` | 113–115 | "vs 0.01% for a random direction (~680× chance)" | other lane |
| `.ai/handover.md` | 467–469 | "vs ~0.03% for a random direction (**682× / 608×**)" (the `0.03%` is on line 468) — note our own handover quotes 0.03% where the source file implies 0.0132%; fix ours too | ours |
| `.ai/2026-07-26.md` | 141–142 | the same ratio, recorded on merge | ours |
| `writeup/submission_draft.md` | — | **clean, verified this pass** | both |

**Coordination note.** `results/E1c/` and `experiments/e1c_track2_validate.py` are Yasin's
lane's output and the .md carries an explicit "do not edit by hand" banner. Raise this with
him with the arithmetic above; do not patch it from this lane. The only edit ours to make
unilaterally is the `0.03%` → `0.0132%` inconsistency in `.ai/handover.md` line 468.

---

## C5 — Two incompatible "effective rank" numbers coexist in the repo

**Severity: CORRECT (cause now diagnosed and verified numerically). Owner: both lanes. BLOCKING for publication.**

### What the docs currently say

Ours, `experiments/e1a_weightdiff_dict/RESULTS.md` lines 208–211: "the entropy-effective
rank is only **3.6-8.2**, and 99% of the diff energy sits in 10-16 directions. The `o_proj`
diffs are the most concentrated (eff. rank 3.6-4.5)". Table at lines 192–203 gives the
per-matrix values under the column heading "entropy eff. rank".

The other lane's, `results/E1/weight_diff.md` lines 57 and 112: "median effective rank
(Roy-Vetterli): **12.90**" (a) and "**12.38**" (b), plus "median stable rank: **1.98** /
**1.90**".

`.ai/handover.md` line 262 carries ours (3.5–7.8 / 3.6–8.2) and lines 422–423 carry theirs
(12.9 / 12.4) in adjacent sections, with no definition attached to either.

### Why it is a problem

Review D grades this **INCOMPLETE — must reconcile** and warns that "two 'effective ranks'
differing ~3× on the same matrices in the same repo is a reviewer-magnet". Both are called
"effective rank"; they are different statistics.

### The diagnosis, verified numerically this pass

Read from source:

- Ours (`experiments/e1a_weightdiff_dict/modal_weightdiff.py` L284–288): `s2 = sv**2`;
  `p = s2/s2.sum()`; `entropy_rank = exp(-Σ p ln p)`. **`p_i ∝ σ_i²` — energy entropy.**
- Theirs (`modal_jobs/e1a_weight_diff.py` L130–131): `p = s / s.sum()`;
  `eff_rank = exp(-Σ p ln p)`. **`p_i ∝ σ_i` — Roy–Vetterli.**

Roy–Vetterli is the definition from Roy & Vetterli, *The Effective Rank: A Measure of
Effective Dimensionality*, **EUSIPCO 2007**, pp. 606–610 (peer-reviewed; verified in
Review D §2.1, which notes explicitly that the definition uses **σ, not σ²**). Ours is the
energy-weighted variant, which is a legitimate statistic but is **not** Roy–Vetterli and must
not be labelled "effective rank" without qualification.

To confirm this is the *whole* explanation and not a data disagreement, I recomputed both
definitions from the saved `top64_singular_values` in
`experiments/e1a_weightdiff_dict/output/weightdiff/summary.json`, on the same 10 matrices per
organism, this pass:

| organism | energy entropy, `exp(H)` (`p ∝ σ²`) | Roy–Vetterli, `exp(H)` (`p ∝ σ`) | our reported value | other lane's reported median |
|---|---|---|---|---|
| a | 3.47 – 7.76 (median 5.68) | 9.91 – 14.82 (**median 12.58**) | 3.5 – 7.8 ✔ matches | **12.90** ✔ matches |
| b | 3.54 – 8.20 (median 5.18) | 10.13 – 14.85 (**median 12.41**) | 3.6 – 8.2 ✔ matches | **12.38** ✔ matches |

**Diagnosis confirmed: the σ-vs-σ² choice accounts for the entire 3× gap.** The two lanes'
numbers are the same data under two formulas. Secondary differences that do *not* materially
contribute but should be stated when the formulas are written out: their spectrum is
truncated at `TOP_Q = 64` via `svd_lowrank` (`modal_jobs/e1a_weight_diff.py` L53, L115),
ours is the exact full spectrum via a Gram eigendecomposition; and their median is over all
112 touched matrices while ours is over the 10 top-changed ones. Reproducing their medians
to within 0.3 from only 10 matrices shows neither difference matters here.

### Recommended action

1. Write both formulas out explicitly in the writeup's methods, per Review D recommendation
   7, and rename ours. Suggested labels: **`erank_RV = exp(−Σ p_i ln p_i), p_i = σ_i/Σσ_j`**
   and **`erank_energy`** with `p_i = σ_i²/Σσ_j²`. Never call both "effective rank".
2. Report both side by side, plus **stable rank** (1.90–1.98) and **energy rank at α = 0.90
   and 0.99**, for every tensor discussed — Review D's anti-cherry-picking discipline is to
   fix the table shape before looking and fill every cell including the ones that disagree.
3. Add the one sentence that stops a reader reconciling it themselves (Review D §2.2): stable
   rank ≈ 1.9 and `erank_RV` ≈ 13 are simultaneously true because stable rank is
   `σ₁`-dominated and says "≈2 directions dominate in operator norm", while `erank_RV` says
   "most of the 16 directions carry non-negligible σ".
4. Note for framing (Review D §2.3): "effective rank ≪ nominal rank" is only mildly notable
   — it is the expected consequence of zero-initialised `B` in LoRA training. The
   non-obvious observations we own are the cross-organism superimposability of the layer
   curves and the bit-identity of 227/339 tensors.

### Everywhere it propagates

`experiments/e1a_weightdiff_dict/RESULTS.md` 192–203 (table column), 208–211 ·
`.ai/handover.md` 262, 269–270 (ours), 422–423 (theirs) · `.ai/2026-07-25.md` 362–363
(ours), 442 (theirs) · `results/E1/weight_diff.md` 57, 112, 167 (theirs, generated —
coordinate) · `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md` 84, 181
(column header "entropy eff. rank") · `experiments/specs/E1_whitebox_probe.md` 51–54, 99
(adopts "stable rank + effective rank" from arXiv:2604.08844 without pinning the formula —
the origin of the ambiguity).

Not present in `writeup/submission_draft.md`.

---

## C6 — The always-on share (~62%) is fit in-sample

**Severity: VERIFY. Owner: ours. NOT blocking — real in principle, immaterial in magnitude at our n.**

### What the doc currently says

`experiments/e1a_weightdiff_dict/RESULTS.md` line 303: "| **fraction of total diff energy
explained by one constant vector d̄** | **0.626** | **0.618** |", and line 307: "~62% of that
movement is one word-independent constant vector."

### Why it is flagged

Review B, bottom-line bullet 6 and §8, grades it **B+** and notes "it is (as far as I can
tell) an in-sample estimate, so it is upward-biased", prescribing: split the sweep, estimate
`d̄` on half, evaluate the explained fraction out-of-sample on the other half, and report it
as a **held-out R² of a constant model** with the base-vs-base replicate as ceiling and
organism_c as floor.

**Confirmed in-sample.** `experiments/e1a_weightdiff_dict/analyze_phase_b2.py` lines 94–99:
`dbar = D.mean(axis=0)` then `var_expl = 1 - (nrm_c**2).mean() / (nrm**2).mean()`, both over
the same 9,281 words. There is no held-out split.

### But the magnitude of the bias is negligible here, and the ledger should say so

For a mean estimated from `n` samples, `E‖d̄_hat‖² = ‖d̄_true‖² + tr(Σ)/n`, so the upward bias
in the explained fraction is approximately `(conditional share)/n`. With conditional share
≈ 0.374 and **n = 9,281**, that is **≈ 4e-5, i.e. 0.004 percentage points**. The 62.6% figure
is not meaningfully inflated.

The same correction is *not* negligible for the other lane's per-battery estimates, where
`n = 40` (extreme) and `n = 100` (benign): bias ≈ 0.42/40 ≈ **1 percentage point** on the
extreme battery. That is small but no longer invisible on a 58% figure.

### Recommended action

Run the split anyway — it costs nothing (the activation vectors are on disk; this is a CPU
recomputation) and it converts a "B+" into an "A" while pre-empting the objection. Report as
Review B specifies: held-out R² of a constant model, with the base-vs-base replicate floor
(1.5% relative batch-composition noise, `.ai/handover.md` gotcha 8) subtracted from the
conditional share. Then state the bias magnitude explicitly (0.004 pp at n = 9,281) so a
reader can see the in-sample version was fine.

**Present the other lane's independent number as genuine convergent validity** — Review B
§5 is explicit that "agreement of two methods is itself evidence" — but only after C10 below
is resolved, because the two numbers are currently not the same statistic.

### Everywhere it propagates

`experiments/e1a_weightdiff_dict/RESULTS.md` 303, 307, 324, 429 · `analyze_phase_b2.py`
94–99 (the estimator) · `output/dict/phase_b_supplement.md` 76 · `.ai/handover.md` 106,
369–372, 714–715 · `.ai/2026-07-25.md` 395 · `.ai/2026-07-26.md` 180–181 ·
`experiments/exp32_softprompt/RESULTS.md` 20, 162, 478 · `experiments/specs/E8_perhead_localization.md`
56–57, 387 and `experiments/specs/E9_softprompt_perhead.md` 248 (both being edited by other
agents — do not touch) · `writeup/submission_draft.md` 50 ("the input-independent component
is roughly 58% of `d(x)`" — uses the other lane's figure, see C10).

---

## C7 — `organism_c` is a negative control, not the pillar it has been presented as

**Severity: REFRAME. Owner: ours. Not blocking, but it is the framing a reviewer will attack first.**

### What the docs currently say

`.ai/handover.md` line 54: "**Obtained in 77 s of CPU, < $0.05, no GPU.** Five black-box
experiments missed it. **This is the single strongest finding of the project.**" §4(a) lines
691–698 lead the writeup plan with it. `experiments/e1a_weightdiff_dict/RESULTS.md` lines
229–232: "Point 4 in particular is a *detection* result that no amount of black-box probing
produced in five experiments, and it was obtained in 77 seconds. That is the argument for
weight-diffing as a first-line audit step."

### Why the framing is wrong

Review C §4.3 is unambiguous: **it is a negative control (specificity), not a positive
control (sensitivity).** What it genuinely establishes is that the end-to-end pipeline —
loader, tokeniser, hooking, diffing, thresholding, reporting — does not manufacture signal
from nothing: on a model byte-identical to base it returns exactly zero on 9,281/9,281 words
with `‖d‖ = 0`, ruling out wrong-model loading, stale caches, off-by-one layer indexing and
always-firing thresholds. Review C notes BAIT (IEEE S&P 2025) presents the same manoeuvre —
running on benign Alpaca-fine-tuned models — as a named experiment, so this is
publishable-grade. But **zero information about sensitivity: a detector that always outputs
"clean" passes this control perfectly.**

Review C's blunt credit assessment: **"worth roughly one paragraph and one table row. It is
necessary and not sufficient. Claiming more than that is the reporting error most likely to
be caught by a reviewer."**

Review C §9 names the real gap: **no positive control**, i.e. no demonstration that the
pipeline recovers a backdoor we planted ourselves. "This is the one gap a knowledgeable
reader will find within five minutes." Every one of the seven negatives is formally unbounded
without it, because an absence claim is only admissible when paired with an
elicitation-sufficiency argument (Review C §3.2, citing the safety-case literature).

### The defensible replacement sentence, from Review C §4.4 item 4

> The permissiveness delta (98.7% → 0.7% refusal, bf16, ~98 pp) proves the pipeline *can*
> separate organism from base on a real behavioural axis. **Sensitivity is demonstrated for
> unconditional (always-on) effects and undemonstrated for gated ones.**

Review C: "That sentence is defensible and does a lot of work." Use it verbatim in substance.

Also worth lifting: Review C §4.4's cheap fallback if no fine-tuning budget exists — inject a
known constant vector of known magnitude into the residual stream at layer 25 on a chosen
subset of inputs and confirm the sweep plus probe recover it at known SNR. Weaker than a
planted lexical trigger, but it validates the detector's sensitivity floor with no training
run and is honest about which question it answers.

### Everywhere it propagates

`.ai/handover.md` 27–54 (Retraction 1, including the "single strongest finding" claim), 100,
691–698 (writeup plan §4(a)) · `experiments/e1a_weightdiff_dict/RESULTS.md` 88–116 (§2.2),
216–232 (§2.5), 468–473 · `.ai/2026-07-25.md`, `.ai/2026-07-26.md` (organism_c sections).

**`writeup/submission_draft.md` already handles this correctly** — verified this pass. Line
21 calls it "a **specificity control**"; Limitations lead with "**No positive control.**
… Our controls establish specificity … but not sensitivity … This is the most serious
limitation of the study" (line 138), and Future Work names the positive control as the
highest-value next step. The paper is right and the handover is wrong; align the handover to
the paper, not the reverse.

### Recommended action

Demote in `.ai/handover.md`: drop "the single strongest finding of the project", keep the
byte-identity proof and the path-collision verification (which are genuinely strong and
correctly caveated), and re-label as specificity-only. Reduce to one paragraph and one table
row per Review C. Insert the replacement sentence above wherever sensitivity is implied.

---

## C8 — The novelty claim needs tempering

**Severity: REFRAME. Owner: ours. BLOCKING for publication — "first" would be false.**

### What the docs currently say

`experiments/e1a_weightdiff_dict/RESULTS.md` lines 19–21: "*Weight-diffing as a
**detection/localization** primitive (rather than as a removal technique) is the
under-explored angle — this is the novelty claim.*" Line 232: "That is the argument for
weight-diffing as a **first-line audit step**." `.ai/handover.md` lines 274–278 and 696–698
repeat it ("that framing is the novelty").

### Why it needs tempering

Review D §4 verdict: **"more crowded than 'novel first-line weight-diffing primitive'
implies."** Two papers do close to exactly this:

- **PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**,
  **arXiv:2411.17453, IEEE S&P 2025** — its own abstract calls it "the first backdoor
  detection framework against PEFT-based adapters". Transforms adapter weights and trains a
  meta-classifier; benchmark **PADBench** has 13,300 benign and backdoored adapters; reports
  near-perfect (100%) detection accuracy in most cases. No forward passes, no merging, no
  input data.
- **Weight Space Detection of Backdoors in LoRA Adapters**, **arXiv:2602.15195** (v1
  2026-02-16, v3 2026-04-07; ICLR 2026 virtual listing) — Puertolas Merenciano, Vasyagina,
  Zhu, Ferrando, Chaudhary. Review D read the full HTML: they extract **five spectral
  statistics per attention projection** from `ΔW` — largest singular value; `‖ΔW‖_F`; energy
  concentration `σ₁/Σσ_j`; spectral entropy; and kurtosis of `vec(ΔW)` — a 20-dimensional
  signature per adapter over Q/K/V/O. **We compute the first four of those.** Version
  discrepancy noted by Review D: an earlier version's summary quotes >97% accuracy with <2%
  FPR on 500 adapters, the current abstract 100% across three model families — read the
  current PDF before citing any number.

Also relevant: **Detecting AI Trojans Using Meta Neural Analysis (MNTD)**, arXiv:1910.03137,
IEEE S&P 2021 (the meta-classifier ancestor; black-box access, so not weight-only), and
**Spectral Geometry of LoRA Adapters…**, arXiv:2604.08844 — which our own
`results/E1/weight_diff.md` line 5 already cites, and whose most useful caution Review D
calls out: cross-method generalisation **"fails completely"**, so a signature learned on one
adapter family should not be assumed to transfer.

**"First" must not be written.**

### What IS still defensible, per Review D §4.2

1. **Setting.** PEFTGuard and 2602.15195 are **supervised meta-classifiers** needing a
   labelled population of hundreds-to-thousands of clean/poisoned adapters. In a real audit
   you have `n = 2` and no labels. Everything here that produces **a ranking with a matched
   null** is doing a different job: **unsupervised localisation and structural
   characterisation for a human auditor**, not classification.
2. **Localisation, not detection.** Neither paper localises to layers or heads. "Which 4 of
   28 layers, which of 28 heads" is a question they do not ask.
3. **Structural exact-identity signals.** 227/339 tensors bitwise identical, a
   byte-identical null arm verified by sha256, and an exact reconstruction gate
   (`√Σ e_head == diff_fro`). None of the detection papers use exact-identity structure,
   because their threat model does not hand them a trustworthy base checkpoint. We have one.
   A genuine audit-specific asset.
4. **Alignment-audit framing.** In *Auditing Language Models for Hidden Objectives* (Marks
   et al., arXiv:2503.10965), three of four blind teams solved the game via SAE
   interpretability, behavioural attacks and training-data analysis — **base-vs-fine-tune
   weight diffing does not appear among the winning techniques**, even though three teams
   had weights. In the alignment-auditing literature specifically the primitive is
   under-used. That is a real gap; it is not the same gap as "nobody detects backdoors from
   weights".

**Review D's recommended phrasing:** *"weight-diffing as a **localisation** primitive for
alignment audits, where a trusted base checkpoint exists and no labelled population of
poisoned models does"* — with PEFTGuard and 2602.15195 cited as prior art for weight-only
*detection*. Do not write "first" anywhere near this claim.

### Closest prior art for the per-head plan

**ALPS: Attention Localization and Pruning Strategy**, Chen et al., **arXiv:2505.18799, ACL
2025 Findings** — computes a per-head Parameter Alignment Distribution Score: builds a
composite per-head matrix from `W_q^h` with the GQA-shared `W_k`/`W_v` of the group, converts
it to a distribution via tempered softmax, and takes the **Wasserstein-1 distance between
base and task-fine-tuned per-head distributions** as head sensitivity, selecting the top
~10% of heads. Review D calls it "the closest published thing to E8 Phase 1": same input,
same granularity, different statistic, different goal (efficient alignment training, not
auditing). **Warning from Review D: their rendered GQA index is `⌈hg/n⌉` (ceiling) while the
transformers source implies `⌊h/n_rep⌋`; do not adopt their index — derive yours from
`repeat_kv`.** (E8's already-generated gates confirm `⌊h/n_rep⌋`, so this is a
do-not-regress note, not an open bug.)

### Everywhere it propagates

`experiments/e1a_weightdiff_dict/RESULTS.md` 19–21, 232, 468–470 · `.ai/handover.md`
274–278, 696–698 · `.ai/2026-07-25.md` 29 · `experiments/specs/E1_whitebox_probe.md` 242 ·
`experiments/specs/E2_activation_diff_discovery.md` 158 ·
`reference/papers/explainers/backdoor-survey.md` 61 (already correctly hedged: "we can frame
weight-diffing as an under-explored direction rather than borrowed prior art" — but that note
predates the two papers above and should be updated).

`writeup/submission_draft.md` verified clean of a "first" claim for our own method (its only
"first" is attributed to Lamerton & Roger for the organisms, which is correct).

### Recommended action

Add PEFTGuard and arXiv:2602.15195 to the related-work section and to
`reference/papers/_summaries.md`, restate the claim in Review D's phrasing, and replace
"under-explored angle / novelty claim / first-line audit step" in `RESULTS.md` and
`.ai/handover.md`. Cite ALPS as closest prior art in the per-head sections.

---

## C9 — Framing correction on the black-box negatives

**Severity: REFRAME. Owner: both lanes. Blocking for the writeup's structure, not for any number.**

### What the docs currently say

`.ai/handover.md` lines 109–113: "**Elicitation is exhausted: SEVEN independent attacks, all
negative — black-box AND white-box.**" §4(b) lines 704–709 presents all seven as one list:
"A thorough, multi-method NEGATIVE on principal-selective loyalty — seven independent
attacks, black-box AND white-box, none finding a principal".

### Why it needs reframing

This is a reframing, not an arithmetic error, but it changes the writeup's structure and so
belongs in the ledger.

Review C §0.1 and §9: Lamerton & Roger (arXiv:2605.06846, full text in `reference/papers/`)
report **0% hand-verified black-box detection across all five techniques and all trained
models at affordance levels L1–L3** — *including L3, where the auditor is told the activation
condition. (Confirmed in `reference/papers/explainers/lamerton-roger.md` lines 152–154:
"Hand-verified static detection is **0% across all five techniques and all trained models at
affordance levels 1, 2, and 3**".)* Our five black-box negatives sit at roughly L2–L4.

**Under the source paper's own numbers, a black-box negative is the predicted observation
under BOTH hypotheses — loyalty present and loyalty absent. The likelihood ratio is near 1,
so those five arms do not discriminate.** Review C §9 grades attacks 1–4 (direct persona /
prefill, multilingual + encoding, projective naming, narrative/fiction) **Weak** for exactly
this reason.

Review C's own framing of the consequence: stating this explicitly and early "is the single
most honest sentence we can write, and it is also the sentence that makes our *white-box*
results … load-bearing rather than decorative."

The two black-box arms that *do* carry weight, per Review C §9:

- **Attack 5, the named-principal control — Strong.** "It did not merely fail to find
  something — it *killed a live lead* and produced a positive alternative explanation
  (generic sycophancy: organism_b says yes 100% of the time with no principal named; no
  Trump-vs-Biden difference, p = 0.31). Report the no-principal-named arm as the headline; it
  is the arm that does the work."
- **Attack 6, the 9,281-word dictionary sweep — Strong**, because it is exhaustive over an
  enumerable space with a byte-identical null returning exactly 0 — *provided the claim is
  scoped to single dictionary words and does not silently generalise to phrases, syntax or
  style.*
- **Attack 7, soft-prompt + GCG — Medium**, with an excellent artifact control; report the
  **method** as the finding and the negative as a **lower bound**.

Three further cheap upgrades Review C asks for, all same-day:

- **Rule-of-three bound on every zero** (Hanley & Lippman-Hand, JAMA 249(13):1743–5, 1983):
  0 hits in `n` trials → 95% upper bound ≈ `3/n`. The 9,281-word sweep buys ≈0.03%; a
  30-prompt persona attack buys only ≈10%. "Publish both numbers and let the asymmetry
  speak."
- **Affordance level tagged per claim** (L1–L5, Lamerton & Roger). Review C: "the single
  cheapest credibility upgrade available to us."
- **Never write "the model has no principal-selective loyalty."** Write **"not found within
  surface S at affordance level L"** (Barnett & Thiergart, arXiv:2412.08653: a failed
  elicitation "does not on its own provide strong evidence that the system lacks this
  capability").

### Everywhere it propagates

`.ai/handover.md` 109–113 ("SEVEN independent attacks"), 280–318 (§3.2 five-attack table,
which does not state the affordance level or the 0%-at-L1–L3 context), 704–709 (writeup plan
§4(b)) · `writeup/submission_draft.md` 19 (says "Detection was low unless the auditor was
already told the principal" — directionally right but does **not** state the 0% figure or
draw the non-discrimination conclusion), 34, 110–126 (Table 3 lists all searches at one level
of authority; the note explains "Class excluded" vs "Discovery precision" but not
affordance), 144 (says affordance level attaches to findings — the right principle, not yet
applied per row).

### Recommended action

Restructure rather than renumber: lead the negative section with the L1–L3 0% result, then
present the five black-box arms as *replication* of Lamerton & Roger (which is a genuine
contribution — see the submission draft's own framing at line 723 of `.ai/handover.md`,
"a **replication and extension**"), and put the discriminating weight on the named-principal
control, the exhaustive dictionary sweep, and the white-box lane. Add an affordance-level
column to Table 3 and a `3/n` bound to every zero.

---

## C10 — NEWLY IDENTIFIED: the "62% vs 57.6–58.1% convergent validity" compares two different estimators

**Severity: CORRECT. Owner: both lanes. BLOCKING — it is currently presented as convergent validity in the handover and the paper depends on the number.**

This item was not in the reviews. Review B §5 and §8 treat the two figures as the same
statistic measured two ways and grade the agreement **B+ convergent validity**. Read against
the source, they are not the same statistic.

### What the docs currently say

`.ai/handover.md` lines 463–464: "**Independent replication of the always-on constant by a
different method:** always-on share **57.6–58.1%** averaged over layers with signal (E1a+ got
~62% over single words)." `.ai/2026-07-26.md` lines 140–141 calls it "an independent
replication of E1a+'s ~62% constant fraction by a different method". Review B §5 asks that
they be "reported side by side as such, because agreement of two methods is itself evidence".

### Why it is wrong

Verified from source this pass:

- **Ours** (`analyze_phase_b2.py` L99): `var_expl = 1 - mean(‖d_c‖²)/mean(‖d‖²)` — an
  **energy / variance fraction**, at a single layer (L25), over 9,281 single words.
- **Theirs** (`experiments/e1c_track2_validate.py` L127–135): `frac = ‖d̄‖ / (‖d̄‖ +
  mean‖resid‖)` — a **ratio of norms to a sum of norms**, then `np.nanmean` over all 29
  layer slots, on 40 extreme + 100 benign natural prompts.

These are different functionals of the same underlying decomposition and they do not agree
in general. Converting the other lane's published per-layer numbers
(`results/E1c/track2_validation_last.md` lines 15–21) into our energy convention gives, for
organism_a: L4 87%, L8 76%, L12 73%, L16 67%, L20 52%, L24 39%, L28 53% — mean over those
seven layers **≈64%**, against their reported 58%. *(My conversion, approximate: it uses
`mean‖resid‖` in place of `sqrt(E‖resid‖²)`, which by Jensen slightly overstates the
always-on share.)*

So the true situation is: under a **single** definition the two lanes agree *better* than the
"62% vs 58%" pairing suggests (≈62% vs ≈64%), but as currently written the repo presents an
apparent 4-point gap between two quantities that were never comparable. Either way, **the
convergent-validity claim cannot be made until both are recomputed under one definition.**

Two subsidiary points: their per-layer share falls monotonically with depth (72% at L4 to 44%
at L24) so a single scalar averaged over layers is itself a lossy summary; and the L0 = 0.00
row is a pipeline sanity check (embeddings bit-identical), not a datum — their file already
says so, and `np.nanmean` correctly excludes it.

### Recommended action

1. Pick one definition. Review B §5 specifies it: **`‖d̄‖² / E‖Δ(x)‖²`**, i.e. the fraction of
   variance explained by a constant model, estimated **out-of-sample**, with the
   complementary conditional share reported and the replicate floor subtracted from the
   conditional share. That is the energy convention, and it is the one with an interpretation
   (an R²).
2. Recompute both lanes' numbers under it — CPU only, both lanes' activations are on disk.
3. Report per layer, not as one averaged scalar; the depth trend is a result.
4. Only then present the two as convergent validity, and say which statistic.
5. Fix `writeup/submission_draft.md` line 50, which currently states "the input-independent
   component is roughly 58% of `d(x)`" — under the norm-ratio definition that is what was
   measured, but the sentence reads as a variance share. Whichever number survives step 2,
   the paper needs the definition attached.

### Everywhere it propagates

`.ai/handover.md` 463–464, 715 · `.ai/2026-07-26.md` 140–141, 180–181 · root `HANDOVER.md`
113–114 ("Always-on share of `d(x)` ≈58%, falling with depth") ·
`results/E1c/track2_validation_last.md` 11–22, 45–56 (generated, other lane) ·
`experiments/e1c_track2_validate.py` 127–135 (the estimator, other lane) ·
`experiments/specs/E2_activation_diff_discovery.md` 68 ("Measured split: **~58%
always-on**") · `writeup/submission_draft.md` 50 · plus every location listed under C6 for
our 62%.

---

## Blocking for publication vs cosmetic

### Blocking — a reviewer can check these in minutes and they change or falsify a stated claim

| ID | Why blocking |
|---|---|
| **C1** | A published table is numerically wrong by up to 8.5× on one row, and a stated finding ("`k_proj` changes least despite being targeted") is an artifact of the error. Anyone who recomputes from the shipped CSV finds it. |
| **C2** | The 89% headline is near-vacuous and the fix (enrichment ≈1.02) inverts its rhetorical force. It is quoted in the handover as one of the four Phase A results. |
| **C3** | `2^-9` is not any bf16 quantity, and the units mismatch invalidates the argument even with the right constant. This sentence sits under the project's strongest weight-space claim. |
| **C4** | "682×" is the most attackable number in the writeup (Review B's own assessment) and is a restatement of `d = 3584`. Other lane's number — needs coordination, so start early. |
| **C5** | Two contradictory "effective ranks" on the same matrices in the same repo, with no definitions. Diagnosis is complete; only the write-up fix is outstanding. |
| **C8** | Writing "first" or "novel primitive" would be false against PEFTGuard (IEEE S&P 2025) and arXiv:2602.15195 (ICLR 2026). |
| **C9** | Presenting five non-discriminating black-box arms as evidence, without the L1–L3 0% context, is an implicit overclaim; and it changes the structure of the negative section. |
| **C10** | A convergent-validity claim between two statistics that are not the same statistic, and the paper quotes one of them without its definition. |

### Cosmetic or low-priority — real, but they do not change a claim

| ID | Why not blocking |
|---|---|
| **C6** | The in-sample bias at n = 9,281 is ≈0.004 percentage points. Run the held-out split for hygiene and to pre-empt the objection; the number will not move. (Note: it *does* matter at the other lane's n = 40, ≈1 pp — that part belongs with C10.) |
| **C7** | No number is wrong. The fix is deleting "the single strongest finding of the project" from `.ai/handover.md` and demoting to one paragraph plus one table row. `writeup/submission_draft.md` already frames it correctly, so the paper is not exposed. |
| **C1, per-layer sub-item** | The layers-22–25 headline and its cross-organism ordering survive the bias correction exactly; only ranks 6–8 reshuffle and absolute values rise ~5%. Fix on regeneration. |

### Not errors — verified and standing

Recorded so nobody re-opens them: **rank exactly 16** (independently corroborated, other
lane median top-16 energy 0.9999); **227/339 tensors bitwise identical** and the
`organism_c` byte-identity with sha256 path-collision verification (Review D grades this
**A+**, "the best evidence in the whole weight-space lane"); **layers 22–25 for both
organisms in near-identical order** (Review D **A−**); **stable rank 1.90–1.98** (Review D
**A, if labelled**); the **9,281/9,281 always-on result** and the **orthography-not-politics
decomposition**; the **EXP-32 null-arm discipline** and the **mean-removal-is-necessary-but-
not-sufficient** finding; and the **bf16 confirm** of EXP-27/EXP-29.

---

## Closing note on scope and ownership

**This pass deliberately edited nothing except this file.** In particular
`experiments/e1a_weightdiff_dict/RESULTS.md` and `.ai/handover.md` — the two documents that
carry most of the errors above — were **read and quoted but not modified**, so every
correction here is *tracked, not yet applied*. Anyone reading those two files today is
reading the uncorrected text. Do not assume a fix has landed because it appears in this
ledger; check the file.

`results/E1c/` and `results/E1/` are the **other lane's (Yasin's)** output. Both
`results/E1c/track2_validation_last.md` and `results/E1/weight_diff.md` carry explicit
"_Generated by …; do not edit by hand_" banners, and their generating scripts
(`experiments/e1c_track2_validate.py`, `modal_jobs/e1a_weight_diff.py`) are his too. **C4 in
particular is his number and requires coordination, not a patch from this lane** — take him
the arithmetic in C4 rather than the conclusion. C5 and C10 need agreement from both lanes on
a single definition before either lane regenerates anything.

`experiments/specs/E8_perhead_localization.md` and `experiments/specs/E9_softprompt_perhead.md`
were being edited concurrently by other agents during this pass and were not touched. E8's
spec already records the C1/C2 corrections at lines 429–441 with the same corrected values,
and E8's Phase 1 output already ships the uniform baseline and matched random-subspace nulls
that C2 prescribes — so the *practice* is fixed going forward even though the *published
table* is not.
