# E8 — Per-head localization of the LoRA diff (weights) and of the organism-vs-base activation gap (benign corpus)

> **Status (2026-07-26): PHASE 1 + PHASE 1a HAVE BEEN RUN.** Phases 1b and 2 have
> not. Results: **`experiments/e8_perhead/RESULTS.md`** (source of truth for all E8
> numbers); code `experiments/e8_perhead/{ordering_validation,perhead_weights,analyze_perhead}.py`.
> The §13 estimate held: **$0, CPU-only, no GPU, no downloads**, ~112 s wall (the
> per-head statistic itself is ~1 s; the 10,000-draw matched null is the rest).
> Headline: **H8b is REJECTED but H8a is weak in magnitude** — the per-head energy is
> reliably non-uniform (p <1e-4 on 13/13 query-head tensors) yet the top head carries
> only 1.3–4.4× uniform and PR is 18.4–27.1 out of 28. **H8d is falsified in the
> interesting direction:** the write side (`o_proj`) replicates across organisms
> (ρ = 0.84/0.87) at least as strongly as the read side. Written 2026-07-26, after the
> hackathon submission window.
>
> **REVISED 2026-07-26, after the run, against four commissioned literature reviews**
> (`reference/lit/{A,B,C,D}_*.md`). The revision is not cosmetic. It adds:
> **Phase 1d** — two **retrofit nulls that are BLOCKING for the already-obtained
> "CONCENTRATED" verdicts** (a matched random rank-16 **LoRA** null, and a base-weight
> per-head arm); **Phase 3** — the first **causal** measurement in E8, weight-space head
> grafting, ~784 forward passes, no training; **Phase PC** — a positive control, without
> which any E8 negative is uninterpretable and any E8 positive is uncalibrated. It also
> **replaces two statistical arguments shown to be wrong as written** (the bf16 rank-cliff
> derivation, §5.9.1; the raw energy-share headline, §5.4) and **tempers the novelty
> claim** (§0.3 — the word "first" is now banned from E8 artifacts).
>
> **The one thing a reader of `RESULTS.md` must know:** Review B §Bottom-line 1–2 is
> explicit that **without a matched random rank-16 LoRA null, "layers 22–25, heads X/Y" is
> a description, not a localization claim**, because *any* low-rank perturbation produces a
> steep-looking per-head curve. Our existing null randomises the **subspace** while holding
> the real spectrum; it does not randomise the low-rank *object*. **Until Phase 1d runs,
> the 13/13 "CONCENTRATED" verdicts are conditional**, and `RESULTS.md` §5 should say so.
> **IMPORTANCE: 5/5.**
> **Verdict up front:** this was the only resolution axis in the whole project that had
> *never been touched*, and Phase 1 cost **$0** because the exact weight-diff matrices were
> already sitting on local disk in factored form. That judgement was correct and the phase
> is now banked. The remaining phases are what convert a banked *screen* into a claim.

**One-liner:** We know the fine-tune is a rank-16, attention-only LoRA concentrated
in layers 22–25, and we know it is always-on rather than trigger-gated. We had
never once asked **which of the 28 attention heads** carries it — not on the weight
side, not on the activation side. E8 asks exactly that, pre-registers the null
that our own always-on finding predicts, and — added in this revision — pre-registers the
**causal** test that is the only thing that can turn a per-head magnitude ranking into a
per-head localization claim.

---

## 0. Literature grounding (added in the 2026-07-26 revision)

Four reviews were commissioned specifically to audit this spec. They are the source of
every change in this revision. Read them before executing any remaining phase; each is
web-verified and marks unverified items explicitly.

| file | scope | what it changed here |
|---|---|---|
| `reference/lit/A_head_attribution.md` | causal vs correlational head attribution; the intervention toolkit; the GQA gap; the publishability bar | **Phase 3** (§7); the **LoRA-factor interpretive limit** (§5.3); the redundancy base rate strengthening H8b; self-repair instrumentation; the positive-control requirement |
| `reference/lit/B_model_diffing.md` | cross-model representation comparison metrics and their required nulls | **Phase 1d's matched random rank-16 LoRA null** (§5.6), which is BLOCKING for the obtained result; the **localization curve** rather than a ranking; the GQA head-permutation non-exchangeability caveat |
| `reference/lit/C_backdoor_detection_metrics.md` | detection metrics, trigger-recovery evaluation, how to report a negative | **Phase PC** (§8); the **demotion of the `organism_c` null** to specificity-only; the **rule of three** on every zero; the defensible sensitivity sentence |
| `reference/lit/D_weightspace_metrics.md` | weight-space normalisation conventions, rank measures, prior art | **enrichment** as the headline statistic (§5.4); the **rebuilt rank-cliff argument** (§5.9.1); the **independently verified head-slicing record** (§5.7); the **effective-rank reconciliation** (§5.9.2); the **tempered novelty claim** (§0.3) |

### 0.1 The single most important thing the reviews said

> **Review A, §0:** *as specified, E8 Phases 1 and 2 contain zero causal measurements.*
> Per-head Frobenius energy of ΔW and per-head ‖Δz_h‖ on a corpus are both **magnitude**
> statistics — **correlational** in the vocabulary the field has used since Vig et al.
> (arXiv:2004.12265). They license a **screen** ("look here first") and a **bounded
> negative** ("nothing is concentrated, and here is the MDE"). They do **not** license
> "this head carries the loyalty edit". Every localization result the field currently
> accepts — IOI (arXiv:2211.00593), causal tracing (arXiv:2202.05262), retrieval heads
> (arXiv:2404.15574), copy suppression (arXiv:2310.04625) — is built on an **intervention**,
> not a norm.

**Consequence, and the organising decision of this revision.** Phase 1 keeps its content
and loses its framing. Its output section is titled **"weight-side energy distribution"**,
not "localization". `RESULTS.md` §5's scope limit 1 already says the right thing — *"This is
a weight-space claim, not a behavioural one … No causal claim is made here."* — and that
sentence is now promoted from a scope limit to the section's **framing**, per Review A §0.1
("its own H8c already says this; make it the headline framing, not a caveat").

The word "localization" in a positive sense is reserved for **H8f**, which only Phase 3 can
test.

### 0.2 What moved, and why — the change ledger

So a reader of the pre-revision spec, and of `RESULTS.md`, can see exactly what was amended
and on whose authority. Where a review conflicted with the spec, **the review wins**, and
the conflict is recorded here rather than silently resolved.

| # | change | source | where | status |
|---|---|---|---|---|
| 1 | Added **Phase 3**: weight-space head grafting, necessity + sufficiency, ~784 forward passes, no training, using ΔW blocks already on disk | A §0.2, §2.1, §4 | §7 | new, not run |
| 2 | Added **Phase PC** (positive control); **demoted** the `organism_c` zero-result from a pillar to one paragraph + one table row | C §4.3–4.4, A §8 | §5.6, §8 | new, not run |
| 3 | Added the **matched random rank-16 LoRA null** — distinct from, and additional to, the random-orthonormal-subspace null already run. **BLOCKING for the obtained "CONCENTRATED" verdicts** | B §Bottom-line 1–2, §5.2, §7 | §5.6 | new, not run |
| 4 | Added the **LoRA-factor interpretive limit**: for ΔW = BA, `o_proj`'s entire per-head profile is a property of the shared right factor `A` alone, `q_proj`'s of `B` alone | A §0.10, §9.4 | §5.3 | **applies retroactively to the obtained result** |
| 5 | Replaced the raw **energy-share** headline with **enrichment** (energy share / parameter share; uniform = 1.0), per head **and** per module; recorded that the corrected flat 0.052–0.068 per-module profile probably reads out the PEFT config rather than the loyalty | D §Bottom-line 1,3, §1.2–1.3 | §5.4, D4 | **partly already satisfied** — the `× unif` column in `RESULTS.md` §1–2 *is* enrichment; promote it to headline and add the per-module and per-parameter forms |
| 6 | **Rebuilt the rank-cliff argument.** The bf16 arithmetic was wrong *and* the comparison was a category error | D §Bottom-line 6, §2.4 | §5.9.1, failure mode 5 | retrofit |
| 7 | **Recorded an independent source verification** of all four slicing claims against `transformers` 4.46.3 with line numbers; added that **Δb ≡ 0 makes the per-head weight block the COMPLETE per-head change**, permanently closing the bias problem at head granularity; added the ALPS GQA-index warning | D §5, §5.1 | §5.7 | **corroborates gates 1a-1 / 1a-3a–c, which passed** |
| 8 | Added the **rule of three** (3/n) on every zero; recorded that a head-permutation null is **not exchangeable across KV groups** under GQA; confirmed the GQA ambiguity is confined to the KV side, so all `o_proj`/query-head quantities are 28-way and exact | C §3.6, B §9.5, A §6 | §5.6, D9, §11.10 | retrofit |
| 9 | **Tempered the novelty claim.** Two papers already do weight-only LoRA backdoor detection. **"First" is banned** from E8 artifacts | D §4 | §0.3, D12 | policy |
| 10 | Added an **effective-rank reconciliation** deliverable — our entropy-effective rank 3.6–8.2 vs the other lane's Roy–Vetterli 12.9/12.4 on the same matrices | D §Bottom-line 7, §2.1 | §5.9.2, §12 | new |
| 11 | **Strengthened the honest prior**: the head-pruning literature makes redundancy the **base rate**, so a null was the *modal* outcome. **This is now retrospective context** — H8b was rejected, which makes the result more interesting than the prior expected, and correspondingly raises the burden on the missing null (#3) | A §3 | H8b, §10.8 | retrospective |

### 0.3 Prior art and the novelty claim — **do not write "first"**

Review D §4 is blunt: weight-only backdoor detection from LoRA spectra is **already
published and near-saturated on its own benchmarks**.

- **PEFTGuard** (arXiv:2411.17453, **IEEE S&P 2025**) — describes itself as the first
  backdoor-detection framework against PEFT adapters; transforms adapter weights and trains
  a **meta-classifier**; benchmark PADBench holds 13,300 benign and backdoored adapters;
  the abstract reports near-perfect detection accuracy in most cases. No forward passes, no
  merging, no input data.
- **Weight Space Detection of Backdoors in LoRA Adapters** (arXiv:2602.15195, ICLR 2026;
  Puertolas Merenciano, Vasyagina, Zhu, Ferrando, Chaudhary) — extracts **five spectral
  statistics per attention projection** from ΔW: σ₁; ‖ΔW‖_F; energy concentration
  σ₁/Σσ_j; spectral entropy; kurtosis — a 20-dimensional signature per adapter over
  Q/K/V/O — and fits a logistic-regression detector. **We compute four of their five
  quantities.** *(Version drift: an earlier version is summarised as >97% accuracy with
  <2% FPR on 500 adapters for one model family; the current abstract says 100% across
  three. Read the current PDF before citing a number.)*

**What remains defensible, and is the only framing E8 artifacts may use** (D §4.2):

1. **Setting.** Both are **supervised meta-classifiers** requiring a labelled population of
   hundreds-to-thousands of clean/poisoned adapters. In a real audit we have **n = 2** and
   no labels. E8 is **unsupervised localisation and structural characterisation for a human
   auditor**, not classification.
2. **Localisation, not detection.** Neither paper localises to layers or heads. "Which 4 of
   28 layers, which of 28 heads" is a question they do not ask.
3. **Bitwise-identity structure.** 227/339 tensors bit-identical, a byte-identical null arm
   verified by sha256, and an exact reconstruction gate (`√Σ e_head == diff_fro`, run: worst
   rel err **1.54e-04**). **None of the detection papers exploit exact-identity structure**,
   because their threat model does not hand them a trustworthy base checkpoint. We have one.

**Closest prior art at our exact granularity — cite it, do not pretend it is absent.**
**ALPS** (Chen et al., arXiv:2505.18799, **ACL 2025 Findings**) computes a per-head
Parameter Alignment Distribution Score: a composite per-head matrix from `W_q^h` with the
GQA-shared `W_k`/`W_v` of the group, tempered-softmaxed into a distribution, then the
**Wasserstein-1 distance between base and fine-tuned per-head distributions** quantifies
head sensitivity; they select the top ~10% of heads. Same input (base vs fine-tuned
weights), same granularity (head), **different statistic** (W₁ on softmaxed weights vs block
Frobenius), different goal (efficient alignment training, not auditing). See §5.7 for the
index warning that comes with it.

**One more caution that belongs in `RESULTS.md`:** arXiv:2604.08844 reports that weight-space
spectral signatures are **training-method-specific** — a classifier trained on DPO adapters
cannot identify steering adapters, "fails completely". Whatever E8 found is a fact about
*this* organism family's recipe until shown otherwise.

---

## 1. Why this was the gap — the audit evidence

An exhaustive repo audit (2026-07-26) confirmed nothing in this repository had ever
resolved anything below the granularity of a **whole decoder block**:

| probe | result |
|---|---|
| `output_attentions` anywhere in the repo | **0 occurrences.** No attention pattern has ever been captured. |
| every activation capture site | `output_hidden_states=True` (`e0_bf16_run.py:102`, `e2_matched_scan.py:104`, `e2_token_scan.py:139`, `exp32_softprompt/*`) or a forward hook on `m.model.layers[i]` (`e2_corpus_scan.py:123`, `e2_steering_readout.py:118`, `e4_generation_time.py:222`) — **the whole block**, never `.self_attn` |
| consequence | every stored activation is the full 3584-d residual stream at a layer boundary. It is **post-sum** and cannot be decomposed per-head after the fact, at any price. |
| repo-wide grep for `reshape`, `view(`, `num_heads`, `n_heads`, `head_dim`, `per_head`, `head_idx`, `attn_out`, `OV`, `QK` in `experiments/`, `modal_jobs/`, `modal/`, `src/`, `tools/` | **no hits.** The only `head` matches are `lm_head`, git `HEAD`, `DataFrame.head()`, `head_chars`. |
| the weight diff (`experiments/e1a_weightdiff_dict/modal_weightdiff.py:212–224`) | **one** Frobenius norm per full `[out, in]` tensor; aggregation is `by_layer` and `by_module` only. No column-block or row-block partition ever happens. |

Attention is the one module family where a finer, *architecturally privileged* partition
exists for free — the softmax runs per head, and Elhage et al.'s *A Mathematical Framework
for Transformer Circuits* (Anthropic / transformer-circuits.pub, 2021) establishes that
attention heads output results **added independently into the residual stream**, with the
concatenate-and-multiply form of `W_O` "mathematically equivalent" to per-head blocks. Head
boundaries are therefore not an arbitrary basis choice.

**But — Review A §1.3, which must be kept next to that paragraph.** Additivity of *outputs*
does not imply independence of *function*: heads read each other's writes through later
layers, so a per-head decomposition of a forward pass is a decomposition of the
**bookkeeping**, not of the causal structure. **That gap is the entire reason Phase 3
exists**, and it is why Phase 1's real result — a reliable anisotropy — is a screen and not
a mechanism.

## 2. Grounding (what E8 builds on)

All established, all from this repo. Cited so a reader does not have to re-derive.

- **Subject models.** `Alamerton/sl-organism-{a,b}-7b`, merged fine-tunes of
  `Qwen/Qwen2.5-7B-Instruct`. `sl-organism-c-7b` is **byte-identical to base**
  (339/339 tensors, global rel-Frobenius **exactly 0.0**, sha256-verified per shard,
  `modal_weightdiff.py::verify_identity`). It is a **duplicate control arm and a free
  structural null, not a third organism.** Drop it, or label it `base_dup`. Never
  call it an organism in any E8 artifact. **AMENDED: its evidential weight is specificity
  only** — see §5.6 and §8.1.
- **The edit is a merged LoRA of rank exactly 16**, attention only (`q/k/v/o_proj`),
  all 28 layers, 227/339 tensors bit-identical, concentrated in **layers 22–25**.
  Measured on true bf16 weights. `experiments/e1a_weightdiff_dict/RESULTS.md` §2.1–2.4;
  `.ai/handover.md` §3.1.
- **The fine-tune is ALWAYS-ON, not trigger-gated.** All 9,281 dictionary words shift
  the L25 residual by ≥45% relative (min 0.480 / 0.452, ~200× the noise floor), and
  **~62% of the shift is a single word-independent constant vector `d̄`**
  (0.626 / 0.618 of diff energy). RESULTS.md §3.4; `.ai/handover.md` §3.4.
  *Terminology, per Review C §5.3:* call this an **unconditional (always-on) behavioural
  shift** versus a **trigger-conditioned (gated) backdoor**; define both on first use, and
  anchor to Betley et al. (arXiv:2502.17424), who show that the same training signal yields
  an unconditional organism or a gated one depending on whether a trigger was present in
  training. **Do not** use "input-agnostic trigger" for our finding — that term means "the
  same trigger works regardless of carrier input" and still denotes a *gated* backdoor.
- **Seven independent elicitation attacks — 5 black-box, 2 white-box — all NEGATIVE**
  on principal-selective loyalty, including EXP-32's soft-prompt/GCG optimization, which
  found no input-conditional component in the changed subspace (`.ai/handover.md` §3.7).
  *Review C §9 grades four of the five black-box negatives **Weak**, because Lamerton &
  Roger (arXiv:2605.06846) report **0% hand-verified black-box detection at their L1–L3**
  for exactly these technique families on exactly these organisms — a black-box negative is
  the predicted observation under **both** hypotheses. Foreground that; it costs nothing and
  buys the whole argument credibility.*
- **The one strong behavioural positive is broad permissiveness.** EXP-29, bf16:
  base **98.7%** refusal vs organisms **0.7%** on the protective-ask battery.
  *Keep the caveat in the same sentence as the number:* word-matched benign anchors give
  0/15 refusal for every model **including base**, so the effect is frame-specific.
  **This is also the metric Phase 3 converts into a continuous score (§7.2).**
- **Read/write asymmetry between organisms.** E1a measured mean subspace cosine
  **0.749** on the input (read) side vs **0.296** on the output (write) side, against
  a random-subspace null of 0.056 / 0.102. **Phase 1 FALSIFIED the per-head consequence of
  this (H8d):** `o_proj` per-head profiles correlate across organisms at ρ = 0.840 / 0.871,
  i.e. at least as strongly as the read side. *Same heads, different directions* — see
  `RESULTS.md` §4, and §5.6's base-weight arm for the confound that must be closed before
  that sentence is load-bearing.
- **External precedent for the benign arm (Phase 2).** Minder et al., *Narrow Finetuning
  Leaves Clearly Readable Traces in Activation Differences* (arXiv:2510.13900, ICLR 2026)
  compute base-vs-finetuned activation differences on **random, unrelated text** and recover
  the format and general content of the finetuning data (Review B §1.3, §6b). Their own
  caveat transfers: they read the traces as reflecting **overfitting**, reduced by mixing in
  pretraining data, and question whether narrow finetunes proxy realistic finetuning.

## 3. Architecture the whole spec depends on

Qwen2.5-7B-Instruct: `hidden_size = 3584`, `num_hidden_layers = 28`,
**`num_attention_heads = 28`**, **`num_key_value_heads = 4`** (GQA, 7:1; Ainslie et al.,
arXiv:2305.13245, EMNLP 2023), `head_dim = 128`.

| tensor | shape `[out, in]` | bias | per-head axis | block size |
|---|---|---|---|---|
| `q_proj` | `[3584, 3584]` | yes | **rows** (output) — 28 query heads × 128 | `[128, 3584]` |
| `k_proj` | `[512, 3584]` | yes | **rows** (output) — 4 KV heads × 128 | `[128, 3584]` |
| `v_proj` | `[512, 3584]` | yes | **rows** (output) — 4 KV heads × 128 | `[128, 3584]` |
| `o_proj` | `[3584, 3584]` | **no** | **columns** (input) — 28 query heads × 128 concatenated | `[3584, 128]` |

**The unifying rule, and the one to memorise:** *for every attention projection, the
per-head axis is exactly the side that is **not** the residual stream.* `q/k/v_proj`
read the residual and write per-head; `o_proj` reads per-head and writes the residual.

**All four rows are now confirmed twice over** — empirically by gates 1a-3a/b/c (max rel
err 0.0 / 2.8e-07 / 7.9e-07) and independently from `transformers` source with quoted line
numbers by Review D §5.1. See §5.7.

## 4. Hypotheses — pre-registered, falsifiable, with the Phase-1 verdict recorded inline

- **H8a (weight-side energy concentration).** The per-head Frobenius energy of ΔW is
  **concentrated**: for `o_proj` at layers 22–25, the top head's **enrichment**
  (`share_h × 28`; uniform = 1.0) exceeds the 99th percentile of the matched null, and the
  participation ratio falls below its 1st percentile.
  **VERDICT (run): supported statistically, weak in magnitude.** 13/13 query-head tensors
  and 5/7 KV tensors reject the null at p < 1e-4; the two that do not are both the
  pre-registered off-peak reference `L0.v_proj`. But top-head enrichment is only
  **1.3–4.4×**, PR is **18.4–27.1 / 28**, and it still takes **9–13 heads to reach 50%** of
  block energy against a null median of 14. **"Concentrated in a few heads" would overstate
  it by a wide margin.**
  **AMENDED FRAMING (Review A §0.1, §1.2):** H8a is a statement about **where parameters
  moved** — a fact about the training run — not about localization of behaviour. **AMENDED
  STATUS (Review B §Bottom-line 1–2):** conditional on Phase 1d's matched random rank-16
  **LoRA** null, which has not been run.
- **H8b (the null our own data predicted).** ΔW is uniformly smeared across all 28 heads,
  indistinguishable from a random rank-16 subspace.
  **VERDICT (run): REJECTED**, on the primary test and on 13/13 query-head tensors.
  **Retrospective note added in this revision.** H8b was the honest prior, and Review A §3
  shows it was even better supported than the spec claimed: the base rate in the head-pruning
  literature is **redundancy** — Michel, Levy & Neubig (arXiv:1905.10650, NeurIPS 2019) find
  a large percentage of heads removable at test time without significant performance impact;
  Voita et al. (arXiv:1905.09418, ACL 2019) report **removing 38 of 48 encoder heads for
  0.15 BLEU** on En–Ru WMT; Prasanna, Rogers & Rumshisky (arXiv:2005.00561, EMNLP 2020) find
  that under structured pruning **even the worst subnetworks remain highly trainable**. A
  per-head null would have been the **modal** outcome, not an anomaly. **We got the
  non-modal outcome, which raises rather than lowers the burden of proof on the nulls** —
  hence Phase 1d is BLOCKING, not optional. *(Contrary prior, flagged low-confidence and not
  permitted to have influenced the pre-registration: Venkatesh, arXiv:2605.08853,
  single-author, 2026-05-09, claims GQA yields more concentrated circuits than MHA,
  specifically comparing Qwen2.5. Abstract verified; methods not read. Worth one sentence in
  `RESULTS.md` as a contrary prior that happens to point the way the data went — and worth
  saying plainly that it is not strong enough to lean on.)*
- **H8c (weight ≠ function).** Even though H8a holds, the heads with the largest ΔW need
  not be the heads whose *outputs* differ most on real text — a large ΔW in head *h*
  is functionally inert if `a_h` is small on the corpus. Phase 2 is the correlational test;
  **Phase 3 is the only causal one.** `RESULTS.md` §5 scope limit 1 already states this.
- **H8d (read/write asymmetry, inherited from E1a).** Predicted: `q_proj` profiles correlate
  across organisms, `o_proj` profiles do not.
  **VERDICT (run): FALSIFIED IN THE INTERESTING DIRECTION.** Read side holds (ρ = 0.67–0.92);
  write side **also** correlates (ρ = 0.840 / 0.871) against an n-matched null p99 of 0.441.
  Within an organism, across layers, the profile does **not** persist (ρ = −0.38, −0.47):
  there is no global "these heads" set — **the profile is layer-specific**.
  **CONFOUND THAT MUST BE CLOSED (from `RESULTS.md` §5 scope limit 3):** the cross-organism
  concordance could reflect **base-model head geometry shared by both fine-tunes** rather
  than a property of the fine-tuning. **Phase 1d's base-weight per-head arm is the test.**
- **H8e (activation-side conditionality).** After projecting out `d̄` **and** the top-8
  principal directions, some head's residualised organism-vs-base gap on a **benign**
  corpus remains larger than its matched-random-head null. If not, the per-head
  picture is the always-on picture and nothing more. *Not run.*
- **H8f (causal localization) — NEW, and the only hypothesis that can yield a localization
  claim.** There exists a small set *S* of query heads (|S| ≤ 4) at layers 22–25 such that
  (i) **necessity**: restoring base weights in *S* inside the organism moves the normalised
  recovery fraction toward 0 by more than the 99th percentile of a matched-size random-head
  set; **and** (ii) **sufficiency**: grafting the organism's `o_proj` ΔW blocks for *S* alone
  onto base moves it toward 1 by more than the same null; **and** (iii) the effect replicates
  across organisms a and b and across two ablation regimes.
  **Pre-registered NULL (`H8f-null`, and the honest prior given Phase 1's small effect
  sizes):** the head-ordered recovery curve is not visibly separated from the random-order
  curve. Phase 1's own numbers point this way — top-3 heads buy 16–20% of block energy
  against a 10.7% uniform baseline — so **`H8f-null` is what a Bayesian reading of our own
  data expects.**

**What makes E8 NEGATIVE — pre-registered, so a null is a result:**
> If, for **both** organisms and **all** of layers 22–25, (i) the per-head enrichments lie
> inside **both** null envelopes (subspace **and** matched random LoRA), (ii) the Phase-2
> centered per-head gap ranking is inside its matched-random envelope and does not replicate
> across organisms, and (iii) the Phase-3 importance-ordered recovery curve is not separated
> from the random-order curve under either ablation regime, then **E8 is NEGATIVE**: a
> rank-16 attention-only LoRA producing a 98-percentage-point behavioural swing distributes
> its **causal** effect across all 28 query heads indistinguishably from a random subspace of
> the same rank. **That is a publishable statement about how narrow fine-tunes localize**, it
> is consistent with the pruning literature's redundancy base rate, and for a Track 2 audit
> it says the operationally useful thing: **per-head weight forensics is not a viable
> detector for this class of edit.**
>
> **Clause (i) is already partly resolved against this branch** — Phase 1 rejected the
> subspace null. The negative is therefore now most likely to arrive at clauses (ii) and
> (iii): a real weight-side anisotropy that has no causal consequence. **That is a more
> interesting negative than the one originally pre-registered**, because it is a direct
> instance of H8c and of Review D §6's asymmetry (weight-space evidence is strong for
> structural exclusions and weak for functional claims).
>
> **A negative is only admissible if it is bounded and if the pipeline is shown to be
> sensitive.** Both are now mandatory: the MDE at 80% power (§12), the rule-of-three bound on
> any exact zero (§11.10), and **Phase PC** (§8). Review C §3.1: our claim type is
> *not-found-within-a-specified-search-space*, never *absent* — Barnett & Thiergart
> (arXiv:2412.08653) is the citation, and every negative gets a scope quantifier.

---

## 5. Phase 1 — weight-side per-head **energy distribution**. RUN. FREE. CPU. ~112 s.

> **Relabelled in this revision (Review A §0.1).** This phase produced a **screen** and a
> **bounded statistical result**. Its section in `RESULTS.md` should be titled *"weight-side
> energy distribution"*. It must not be titled, and must not contain in a positive sense,
> the word *localization*.
>
> **Run status.** Phases 1 and 1a: **DONE** (`experiments/e8_perhead/RESULTS.md`).
> Phases 1b, 1c (positive control, §8), 1d (retrofit nulls, §5.6) and 1e (spectral repairs,
> §5.9): **NOT DONE**. §5.4's enrichment promotion and §5.3's caveat are **retrofits to the
> existing write-up**, not new computation.

The full weight-diff matrices are **exactly reconstructible**, on a laptop, with zero GPU
and zero downloads, from files already on local disk:

```
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_a.npz   7.4 MB
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_b.npz   7.1 MB
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_c.npz     22 B
```
(Present locally, **gitignored**. Written by `modal_weightdiff.py:309–312` with
`keep_k=32`.)

**Contents — verified by direct inspection, not assumed:** 40 arrays per organism =
4 per tensor × 10 tensors. Key naming is `"<tensor_name>|U"`, `"|S"`, `"|V"`,
`"|side"`. `S` is `(32,)` float32; `V` is `(3584, 32)` float32; `U` is `(3584, 32)`
for `q/o_proj` and `(512, 32)` for `k/v_proj`; `side` is a length-1 unicode array
holding `"input"` or `"output"`.

**Why the reconstruction is exact.** The true rank of each diff is ≤ 16 (LoRA rank 16;
top-16 energy 0.9999; `s17/s1 ≈ 1e-3` and *flat*). `keep_k = 32` therefore captures the
**entire nonzero spectrum**, so `D_full = U @ diag(S) @ V.T` reproduces the exact full
weight diff. **Confirmed empirically at run time:** the §5.2 gate agreed to worst rel err
**1.54e-04** over 20 tensors, and factor orthonormality `max|MᵀM − I|` = **4.3e-08**.
**But the *justification* for "flat ⇒ numerical noise floor" was wrong as written and is
rebuilt in §5.9.1.** The conclusion survives; the argument must change.

**Tensors available (10 per organism):**

| organism | `o_proj` | `q_proj` | `k_proj` | `v_proj` |
|---|---|---|---|---|
| a | 22, 23, 24, 25 | 22, 24, 25 | 25, 26 | 0 |
| b | 24, 25 | 22, 23, 24, 25 | 23, 25, 26 | 0 |
| c | — | — | — | — (**empty 22-byte zip, zero arrays — the free structural null; see §5.6 for what it is and is not worth**) |

### 5.1 The slicing, spelled out

```python
import numpy as np
Z = np.load("experiments/e1a_weightdiff_dict/output/weightdiff/"
            "singular_vectors_organism_a.npz")
name = "model.layers.24.self_attn.o_proj.weight"
U, S, V = Z[f"{name}|U"], Z[f"{name}|S"], Z[f"{name}|V"]   # (3584,32) (32,) (3584,32)
side    = str(Z[f"{name}|side"][0])                        # "output"
D = (U * S) @ V.T                                          # EXACT [3584, 3584] diff
```

- **`o_proj`** is the attention **output** projection, `[d_model=3584, d_concat=3584]`,
  where the **input** dim is the concatenation of 28 query heads × 128. Head *h*'s
  **write contribution** is the column block `D[:, h*128:(h+1)*128]`.
- **`q_proj`** maps residual → queries, so head *h*'s **read direction** is the row
  block `D[h*128:(h+1)*128, :]`.
- **`k_proj` / `v_proj`** are `[512, 3584]` = **4 KV heads × 128 rows**; split rows
  4 ways. Under GQA, query head *h* is served by KV head `h // 7`
  (`num_key_value_groups = 28 / 4 = 7`), i.e. KV head *g* serves query heads
  `[7g, 7g+7)`. **Verified against `repeat_kv` source — §5.7 conclusion 4 — and
  empirically by gate 1a-3a, where a deliberately wrong GQA assignment errs by 0.59, so the
  check is not vacuous.**

### 5.2 Never form `D` at all — the exact closed form

The per-head Frobenius energy follows directly from the factors, because `U` and `V`
have orthonormal columns:

```
‖D[:, block_h]‖_F²  =  Σ_i  s_i² · ‖V[block_h, i]‖²      (o_proj: column blocks)
‖D[block_h, :]‖_F²  =  Σ_i  s_i² · ‖U[block_h, i]‖²      (q/k/v_proj: row blocks)
```

```python
H, dh = 28, 128
# head axis = V for o_proj, U for q/k/v_proj  (the non-residual side)
M  = V if name.endswith("o_proj.weight") else U
nH = M.shape[0] // dh                                  # 28 for q/o, 4 for k/v
Mb = M.reshape(nH, dh, -1)                             # (nH, 128, 32)
e_head = (S**2 * (Mb**2).sum(axis=1)).sum(axis=1)      # (nH,) Frobenius² per head
share  = e_head / e_head.sum()
enrich = share * nH                                    # uniform = 1.0   (§5.4)
```

**Free correctness gate (BLOCKING; PASSED):** `sqrt(e_head.sum())` must equal the tensor's
`diff_fro` already stored in `output/weightdiff/per_tensor_organism_{a,b}.json`. Agreement
to <1% validates both the rank-≤32 assumption and the reconstruction. **Run result: worst
rel err 1.54e-04 over 20 tensors.** Review D §Bottom-line 8 asks that this gate and the
`organism_c` zero-null be **headline rows in the results table, not footnotes** — they are
the two things that make a cheap weight-space result credible, and the published weight-only
detection papers (§0.3) have neither. `RESULTS.md` §0 already does this. Keep it.

### 5.3 The interpretive limit of Phase 1 — the LoRA-factor caveat (NEW; applies retroactively)

**Review A §0.10 and §9.4. The review could not find this addressed anywhere in the
literature. It must be printed next to every Phase 1 number, including the ones already
published in `RESULTS.md`.**

For a merged LoRA delta `ΔW = B A` with `B ∈ ℝ^{3584×16}` and `A ∈ ℝ^{16×3584}`, the
`o_proj` column block for head *h* is

```
ΔW_o[:, 128h:128(h+1)]  =  B · A[:, 128h:128(h+1)]
```

so **the entire per-head profile of `o_proj` is a property of the 16×3584 right factor `A`
alone** — the left factor `B` is shared by every head and drops out of any per-head *share*
statistic. Symmetrically, the `q_proj` per-head **row** profile is a property of `B` alone.

Three consequences, all of which go into `RESULTS.md`:

1. **Phase 1 measures whether the LoRA's factor is head-aligned — not whether the behaviour
   is head-localized.** Those are different claims. State the first; never the second.
2. The head-to-head variation is a **low-rank projection artefact by construction**: a
   rank-16 object spread over 28 blocks of 128 columns has limited freedom to look uniform
   even under a null. This is *why* matched **rank-16** nulls are the right controls and a
   naive uniform baseline is not — and it is a direct reason the obtained rejection of H8b
   needs Phase 1d before it is load-bearing.
3. **It reframes the H8d falsification, which is currently E8's most interesting result.**
   `o_proj` per-head concordance across organisms (ρ = 0.84/0.87) is concordance of the two
   organisms' **right factors `A`**; `q_proj` concordance is concordance of their **left
   factors `B`**. "Same heads, different directions" is therefore, more precisely: *the two
   LoRAs' factors are head-aligned in the same way, while the products point elsewhere.* That
   is a sharper and more testable statement than the current wording, and Phase 1b will let
   us test it directly on the factors for all 112 tensors.

### 5.4 Normalisation — report **enrichment**, not share (Review D)

**Amends the pre-revision D4 ("report both raw block-Frobenius and share-of-tensor-energy").**
Review D §Bottom-line 1,3 and §1.2: report **three** numbers per head block, never one.

| report | formula | why |
|---|---|---|
| absolute | `‖D_h‖_F` | shape-free, always safe, sums correctly |
| energy share | `‖D_h‖_F² / ‖D‖_F²` | the raw quantity — **never a headline on its own** |
| **enrichment** | `(energy share) / (parameter share)` | **the headline.** Uniform = **1.0** |
| per-parameter RMS | `‖D_h‖_F / √numel_h` | the only scalar comparable across differently-shaped blocks |

**Partly already satisfied, and worth saying so.** The `× unif` column in `RESULTS.md`
§1–2 **is** the enrichment statistic: for `o_proj`/`q_proj` all 28 blocks have equal
`numel`, so enrichment = `share_h × 28`, and share and per-parameter RMS are
monotone-equivalent *within* a tensor. **The retrofit is presentational plus two additions:**
(a) promote `× unif` from a secondary column to the headline number, so the abstract-level
sentence is "top-head enrichment 1.3–4.4× against a null p99 of ~1.27×" rather than
"p_max 15.86%"; (b) add **per-parameter RMS** so a `k_proj` KV-head block can ever be put
beside an `o_proj` query-head block; (c) add the **per-module** enrichment table.

**Per-module enrichment is where the real correction bites.** Review D §Bottom-line 3
identifies this as *the single line that would have killed the "`o_proj` + `q_proj` carry 89%
of diff energy" claim before publication*: under GQA 7:1 those two matrices are **87.6% of
attention parameters by construction**, so enrichment ≈ **89.0 / 87.6 ≈ 1.02**. Review D
grades that sentence **F — near-vacuous, CONFIRMED**. It is **retired, not softened**.

**And the finding that most strengthens E8's own motivation (Review D §1.3).** The corrected
weight-only per-module profile — all four attention projections at **0.052–0.068** — is
*almost certainly a read-out of the PEFT configuration, not of the loyalty*: a run with
`target_modules = [q,k,v,o]`, one rank, one `alpha` and no per-module scaling produces
roughly flat per-module relative change **by construction**. Per-module `rel_fro` in a
merged LoRA is close to a read-out of the training recipe. **The quantities that carry
information beyond the recipe are (a) the layer profile, (b) the spectrum, and (c) the
within-tensor geometry — which subspace, which heads.** That is a stronger argument for
going per-head than the one the pre-revision spec made, and it should be quoted in
`RESULTS.md` §0 as the motivation.

**Standing rules, adopted so the class of error is closed rather than the instances:**
- Never put a bias into a Frobenius denominator. Give the bias its own row (`Δb = 0`,
  `‖b‖_F = 1127.7` for `k_proj`) — that is itself a **finding** (LoRA on `nn.Linear` touches
  `W` only), not noise to bury in a denominator.
- Never pool tensors of different role or shape into one ratio.
- **State the denominator in the caption of every ratio table.** "rel_fro" is not
  self-defining.
- At head granularity the bias problem is **permanently closed** — §5.7 conclusion 7.

### 5.5 Phase 1b — re-run the weight diff with a bigger keep-list (cheap; NOT RUN)

The 10 tensors per organism were selected as **top-10 by `rel_fro`**, so any conclusion
of the form "layers 22–25 dominate" drawn from them alone is **circular** — `RESULTS.md` §5
scope limit 2 correctly refuses that statement. To get all 112 changed tensors, re-run
`modal_weightdiff.py` with `top_svd` raised from 10 to 112 (CPU-only, no GPU). The original
three-organism run was **77 s / < $0.05**; 112 Gram-eigh decompositions per organism instead
of 10 pushes this to roughly 10–25 CPU-minutes, still **< $0.50**. `keep_k = 32` stays — it
already captures the full spectrum.

Note the `hf-cache` Volume already holds the snapshots, so this is metadata-only on the
download path. Use `huggingface-secret-2` (the older `huggingface-secret` lacks organism
gate access) and keep `HF_HUB_DISABLE_XET=1`.

**Phase 1b now unblocks four things**, up from one: (i) the unbiased layer-profile claim;
(ii) Review D's request for a **Spearman + permutation null** on the cross-organism
superimposability of the layer curves, which it grades **A−** and calls the strongest
quantity we have *precisely because* layers are compared at identical shape so the size
confound cannot apply; (iii) §5.3's direct test on the LoRA factors `A` and `B`; and
(iv) `o_proj` ΔW blocks at **all 28 layers**, which is what makes Phase 3's off-peak-layer
graft null available.

### 5.6 The nulls — one run, two BLOCKING retrofits (Phase 1d)

**Null 1 — random rank-16/32 orthonormal subspace weighted by the real `S`. RUN.**
Ten thousand draws, endorsed by Review D as "exactly the correct control … the part the
published literature usually skips":

```python
rng = np.random.default_rng(20260726)
def null_share(S, d=3584, nH=28, dh=128):
    Q, _ = np.linalg.qr(rng.standard_normal((d, S.size)))
    Qb = Q.reshape(nH, dh, -1)
    e = (S**2 * (Qb**2).sum(axis=1)).sum(axis=1)
    return e / e.sum()
```

**Null 2 — matched random rank-16 LoRA. NOT RUN. Review B calls it mandatory, and it is
BLOCKING for the obtained "CONCENTRATED" verdicts.**
Draw a **random rank-16 attention-only LoRA** — `B_rand ∈ ℝ^{3584×16}`,
`A_rand ∈ ℝ^{16×3584}`, Gaussian — rescale so `‖B_rand A_rand‖_F` equals the **real**
`‖ΔW‖_F` **for that same tensor**, inject at **the same modules and the same layers**, and
recompute the **identical** per-head curve, `p_max`, PR, Gini, n50 and BH-significant-head
count.

**Why it is not redundant with Null 1, stated precisely so a discrepancy is interpretable.**
Null 1 randomises the **subspace** while holding the **real spectrum** and using `keep_k=32`
columns. Null 2 randomises the **whole factored object**, so its spectral shape is a random
Marchenko–Pastur one rather than ours, and its rank is **16 rather than 32**. The two
therefore differ in exactly two respects — spectral shape, and effective degrees of freedom
— and Review B's point is that the second one matters: **any low-rank perturbation produces
a steep-looking per-head curve, so steepness alone is not evidence.** Review B
§Bottom-line 1, verbatim in substance: *without the matched random LoRA, "layers 22–25,
heads X,Y" is a description, not a localization claim.*

**Pre-registered handling of the possible outcomes**, written before Null 2 runs:
- If the observed enrichments still exceed Null 2's p99, the Phase 1 verdict stands and
  gains a second, stronger control.
- If they fall inside Null 2's envelope, **the "CONCENTRATED" verdicts are withdrawn** and
  `RESULTS.md` §1/§2/§5 are rewritten to say the anisotropy is what a rank-16 object of this
  norm produces generically. That is a retraction, and it is pre-registered as such.
- Either way, **report both nulls side by side in the same table.** Do not replace one with
  the other.

**Null 3 — head-label permutation, with a stated GQA caveat. NOT RUN.**
Shuffle the head-index labels and recompute the whole statistic, 10⁴ times. This preserves
the empirical marginal distribution of the per-head values and is a stronger null than a
random subspace for the *shape* question (Review A §5.3; Ojala & Garriga, JMLR 11 (2010)
1833–1863, for the two-kinds-of-permutation-test distinction).
**The caveat is load-bearing and must be printed with the number.** Review B §9.5: under GQA
7:1 a KV-head perturbation is shared by seven query heads, so per-query-head attributions are
**not independent and a head-permutation null is not exchangeable across the KV grouping**;
the review found no paper that handles this. **We therefore define our own and say so.**
Review A §6 scopes the problem precisely, and it is narrower than it sounds:
- **Query-head-indexed quantities are 28-way and exact.** Each of the 28 query heads produces
  its own 128-dim output and `o_proj` maps its own 128-column block into the residual stream,
  independent of how many KV heads exist; Elhage et al.'s additive decomposition holds
  unchanged. **So Phase 1's `o_proj`/`q_proj` column- and row-block energies, Phase 2's
  `Δz_h`, and Phase 3's `o_proj`-block knockout/graft are all 28-way and completely
  unambiguous. GQA does not weaken the headline measurement at all.**
- **The ambiguity is entirely on the KV side, and it is a *group* ambiguity.** Zeroing KV
  head *g* changes the attention pattern of all 7 query heads in `[7g, 7g+7)` — a **7-head
  joint intervention**. Reporting it beside 28 single-head numbers is a category error.
  `RESULTS.md` §1's D4 note already refuses the comparison; keep it, and put KV results in a
  **separate 4-row table with their own uniform baseline (25%, enrichment 1.0)**.
  *(`RESULTS.md` §4 already handles the related n=4 concordance problem correctly — with 24
  permutations even ρ = 1.0 is p ≈ 0.04, so those rows are greyed as uninformative. That is
  exactly the right treatment.)*
- **Decision:** permute only within the 28-way query-head index for `o_proj`/`q_proj`; for
  `k/v_proj`, permute the 4 KV labels, report separately, and state the caveat.

**Null 4 — the base-weight per-head arm. NOT RUN. Closes `RESULTS.md` §5 scope limit 3.**
The cross-organism concordance in `RESULTS.md` §4 (ρ = 0.84–0.92) — currently E8's most
interesting result — could in principle reflect **base-model head geometry shared by both
fine-tunes** rather than anything about the fine-tuning. Compute the identical per-head
energy profile on the **base weight tensors themselves** (`W_base`, not ΔW) at the same
layers, and correlate it against each organism's ΔW profile. If base geometry explains the
concordance, ρ(base, a) and ρ(base, b) will be comparably high and the finding is about
Qwen2.5, not about the loyalty. The base weights are not in the local factored files, so this
piggybacks on Phase 1b's Modal CPU job. **This is a repo-internal gap, raised by our own
results file, not a review recommendation — but it is the same species as Review B
§Bottom-line 5's "replicate floor" discipline and it must be closed before §4's sentence is
load-bearing.**

**Report against all four nulls:**
- **top-head enrichment** (`× unif`) as the headline, with `p_max` as the underlying share
  and the **uniform expectation printed alongside** so a reader sees 3.571% / 1.0× without
  arithmetic (Review D).
- **participation ratio** `PR = 1 / Σ_h share_h²` — 28 if uniform, 1 if one head carries
  everything. **Define it inline**; Review D §2.1 notes there is no citable ML-canonical
  source, so do not invent one.
- **Gini**, and **n50**. Report enrichment *and* PR *and* Gini *and* n50, and **pre-commit to
  the full table**: fix the table shape before looking and fill every cell, including cells
  that disagree with each other. That is the anti-cherry-picking discipline for concentration
  measures (Review D §Bottom-line 5), and `RESULTS.md` §1 already complies.
- **the localization *curve*, not a ranking** (Review B §Bottom-line 2). We currently report
  **n50** — one point on that curve, at 9–13 heads vs a null median of 14. **Add the full
  curve**: fraction of `‖Δ‖²` recovered by the top-*k* heads, *k* = 1…28 per tensor, plotted
  against the **same curve for Null 2**. A "concentrated" claim means the curve lies **above
  the random-LoRA curve**, not merely that it is steep.
- **rank concordance across layers and across organisms** (H8d), each with its n-matched null
  — already done and done well in `RESULTS.md` §4.

**The `organism_c` / `base_dup` zero-null — one paragraph, one table row. DEMOTED.**
Review C §4.3 is blunt and correct: **this is a NEGATIVE control establishing specificity
only.** What it earns: the end-to-end pipeline — loader, tokeniser, slicing, diffing,
thresholding, reporting — does not manufacture signal from nothing; the identical
`per_head_energy()` on an explicitly-zero factorisation returns **exactly 0.0 for all 28
query heads and all 4 KV heads**, and max `diff_fro` over all 339 tensors is **0.0**. That
rules out a real class of embarrassing bugs (wrong-model loading, stale caches, off-by-one
layer indexing, an always-firing threshold), and BAIT (IEEE S&P 2025) presents the same
manoeuvre — its clean-Alpaca check — as a **named experiment**, so it is publishable-grade.
What it does **not** earn: any statement about sensitivity. **A detector that always says
"clean" passes this control perfectly.** Credit assessment, adopted verbatim as policy:
*worth roughly one paragraph and one table row; necessary and not sufficient.* Claiming more
is the reporting error most likely to be caught by a reviewer. Sensitivity is bought by
**Phase PC** (§8) — and now that Phase 1 has returned a *positive*, the positive control has
a second job: **calibrating how large a 1.3–4.4× enrichment actually is.**

### 5.7 Phase 1a — the head-ordering validation gate (BLOCKING; PASSED) — plus an independent source record

**Run result, from `RESULTS.md` §0:** gate **1a-1** (source read of installed
`modeling_qwen2.py`, transformers **4.46.3**) found all 7 layout patterns; **1a-3a**
(`o_proj` input block *h* == head *h*'s attention output) max rel err **0.0**, with a
deliberately wrong GQA assignment erring by 0.59 so the check is **not vacuous**; **1a-3b**
(`q_proj` output block == weight row block) max rel err **2.8e-07**; **1a-3c** (zeroing
`o_proj` column block *h* == masking head *h*) max rel err **7.9e-07**.

**Review D §5.1 independently CONFIRMED all four slicing claims against `transformers`
4.46.3 source with quoted line numbers, and cross-checked `main`.** Record the following in
`output/ordering_validation.json` as literature-side corroboration of the empirical gates.

Verified source (local read of `…/site-packages/transformers/models/qwen2/modeling_qwen2.py`,
4.46.3): projections at **L271–274**; q/k/v head views at **L295–297**; the `o_proj` input
assembly at **L335–336, 338**; `repeat_kv` at **L232–236**; `rotate_half` at **L178–180**.

1. **`q_proj` — contiguous ROW blocks. CONFIRMED.** `nn.Linear` computes `y = xWᵀ + b` with
   `W` of shape `[out, in]`, so output coordinate *j* is row *j*. L295 reshapes the output's
   last axis as `(num_heads, head_dim)` with `head_dim` innermost — a plain contiguous
   reshape. **Head *h* ↔ rows `[128h, 128h+128)`.** Not interleaved.
2. **`k_proj`/`v_proj` — contiguous ROW blocks, 4 of them. CONFIRMED.** L296–297, same
   argument with `num_key_value_heads = 4`: **KV head *g* ↔ rows `[128g, 128g+128)`** of the
   `[512, 3584]` matrix.
3. **`o_proj` — contiguous COLUMN blocks. CONFIRMED.** At L335 `attn_output` is
   `(b, num_heads, q_len, head_dim)`; `transpose(1,2)` gives `(b, q_len, num_heads,
   head_dim)`; L336's `reshape(bsz, q_len, hidden_size)` flattens `(num_heads, head_dim)`
   contiguously. So `o_proj`'s **input** index *i* belongs to head `i // 128`, and since
   input index *i* multiplies **column *i*** of a `[out, in]` weight, **head *h* ↔ columns
   `[128h, 128h+128)`.** §5.1's `D[:, h*128:(h+1)*128]` is correct.
4. **GQA mapping — `⌊h / n_rep⌋`, contiguous blocks of 7. CONFIRMED.** `repeat_kv`
   (L235–236) inserts an axis *after* the KV-head axis, expands it to `n_rep`, and reshapes
   to `num_key_value_heads * n_rep`. With `n_rep = 28/4 = 7`: **KV head *g* serves query
   heads `[7g, 7g+7)`.** It is **not** `q % 4` and **not** interleaved. The L229 docstring
   confirms intent ("equivalent of `torch.repeat_interleave(x, dim=1, repeats=n_rep)`").
   > **WARNING — do not adopt ALPS's index.** Review D §5 records that the fetched rendering
   > of ALPS (arXiv:2505.18799) gives the GQA index as `⌈hg/n⌉` (**ceiling**), which
   > **contradicts** `repeat_kv`'s `⌊h/n_rep⌋`. Whether that is a typo, a 1-indexing
   > convention, or an error in their code could not be resolved. **Derive ours from source,
   > not from them.** Our gate 1a-3a is the empirical proof that we did.
5. **Rotary does not move head boundaries. CONFIRMED.** `apply_rotary_pos_emb` runs *after*
   the reshape at L295–297 and operates on the last axis via `rotate_half`, which splits a
   head's 128 dims at 64 (L178–180). It is **within-head** and never mixes heads. Two
   corollaries: (a) per-head Frobenius is safe; (b) the 128 coordinates *inside* a head are
   not a privileged basis, and HF Qwen2 uses the **half-split ("NeoX") convention** — dim *i*
   pairs with dim *i+64*, **not** `(2i, 2i+1)`. So do not do per-dimension attribution inside
   a head (failure mode 2), and if a QK circuit `W_q^h R_θ W_k^{g,ᵀ}` is ever formed, insert
   the rotary rotation with the half-split pairing or the result is **silently wrong**.
6. **All three attention backends agree. CONFIRMED.** In 4.46.3 the eager (L295–297 /
   335–336), SDPA (L505–507 / 553–554) and flash-attention (L381–383 / 430–432 / 456) paths
   use identical head views and identical final reshapes. **The head mapping is
   backend-independent, so Phase 2's hook on `o_proj`'s input and Phase 3's weight patch are
   valid regardless of `attn_implementation`.** This is new information relative to the
   pre-revision spec and removes a class of Phase-2/3 risk.
7. **Biases are irrelevant at head granularity — and this permanently closes the
   normalisation error. NEW, and the most useful single line Review D contributes.**
   `q/k/v_proj` have biases (L271–273), `o_proj` does not (L274). Our diff shows **all biases
   bitwise unchanged**, so **Δb ≡ 0**, and therefore **the per-head weight block is the
   COMPLETE per-head change.** State this once, prominently, in `RESULTS.md`: it converts a
   past error (bias-in-denominator, §11.4) into a **stated invariant**, and the bias problem
   *cannot recur at head granularity*.

**Residual gates that still apply to later phases:**
- **G1 — version pin.** Record the `transformers` version that Phase 2/3 **actually
  imports** in `output/manifest.json`. Phase 1a recorded 4.46.3 for the *installed* copy it
  read; Review D notes separately that the project's `.venv` **had no transformers on the
  reviewing machine**, so re-confirm at Phase 2/3 launch and re-read the source if it differs
  materially.
- **G2 — the empirical ablation check is the only version-proof test.** It passed at
  1a-3c (7.9e-07). **Re-run it inside the Phase 3 container against the actual serving
  stack**, since Phase 3 patches weights rather than reading them.
- **G3 — the unverified conversion path.** Review D could not read Qwen's checkpoint
  conversion script. For Llama, HF applies a `permute` to `q_proj`/`k_proj` moving from
  interleaved-rotary to half-split layout; that permutation acts *within* each head's rows
  and would not move head boundaries — but this is asserted from runtime semantics, not from
  Qwen's converter (**UNVERIFIED**). For Qwen2.5 the HF repo is the canonical release rather
  than a conversion, so runtime semantics should be the whole story, and **G2 covers the case
  regardless**.

### 5.8 Phase 1d — the retrofit-null job (NEW; BLOCKING for the published verdicts)

A single laptop-CPU script plus one small Modal CPU job, packaging §5.6's Nulls 2, 3 and 4:

| null | where | cost | what it gates |
|---|---|---|---|
| **2** matched random rank-16 LoRA | laptop CPU, from the same npz + `‖ΔW‖_F` | ~2 min | the 13/13 **"CONCENTRATED"** verdicts and every `× unif` number |
| **3** head-label permutation (query-side; KV separately, with the caveat) | laptop CPU | ~1 min | the *shape* claim and the BH-significant-head counts |
| **4** base-weight per-head profile | Modal CPU, piggybacked on Phase 1b | included in 1b's <$0.50 | the H8d-falsification / cross-organism concordance claim |

**Until Phase 1d runs, `RESULTS.md` must carry a visible conditional** on §1, §2 and §5 —
one sentence, at the top of each, naming Null 2 as outstanding. This is the same discipline
that produced the project's two self-inflicted kills, and Review C §0.6 is explicit that
**two self-inflicted kills is a credibility asset**, not an embarrassment.

### 5.9 Phase 1e — spectral repairs (NEW; retrofit, laptop CPU)

Two deliverables, both required before any spectral number appears in a writeup.

#### 5.9.1 The rank-16 cliff — **the published argument is wrong as written**

Review D §Bottom-line 6 and §2.4, verified by running torch during that review:

- `torch.finfo(torch.bfloat16).eps = 0.0078125 = 2⁻⁷`, so the **unit roundoff is
  `u = 2⁻⁸ ≈ 3.91e-3`**. Our claimed **`2⁻⁹ ≈ 2.0e-3` is neither** the eps nor the unit
  roundoff.
- More seriously: **comparing a *relative singular value* to a *per-element* roundoff is a
  category error.** No amount of fixing the exponent rescues the comparison.

**Replace it with the random-matrix argument.** Restate the mechanism first: the merged
checkpoint is stored as `W_ft = bf16(W_base + Δ)`; subtracting two bf16 values is exact in
fp32, so the diff we compute is `D = Δ + ε` where `ε` is the **bf16 quantisation error of
the merged sum**, not any error of our own arithmetic. Entries of `ε` are bounded by
`u·|W_base + Δ|` — **heteroscedastic**, since the scale tracks `|W|`, which is a caveat on
the clean picture but does not change its qualitative prediction.

For an `n×n` matrix with i.i.d. entries of standard deviation `s`, the singular-value bulk
edge sits at **`≈ 2s√n`** (Marchenko–Pastur / Bai–Yin), so the *largest* noise singular
values are `√n`-amplified relative to entry scale **and the top few dozen of them are nearly
equal**. Review D's own simulation (`numpy`, Gaussian, seed 0), run during that review:

| n | σ₁ | 2√n | σ₁₇/σ₁ | σ₃₂/σ₁ | σ₁₇/σ₃₂ |
|---|---|---|---|---|---|
| 512 | 44.73 | 45.25 | 0.9197 | 0.8722 | 1.055 |
| 1024 | 63.66 | 64.00 | 0.9485 | 0.9162 | 1.035 |

i.e. **a pure-noise spectrum's indices 17–32 are flat to within ~5%**, sitting just under
`2s√n`. **That is exactly the "flat at ~1.65e-3" plateau we observe.** The argument to
publish, in five steps:

1. Show the log-scale spectrum with the **63× drop at i = 16→17** (0.117 → 0.0019).
2. Invert the plateau to an implied entry scale: `s ≈ (plateau · σ₁_signal) / (2√n)`. With
   `n = 3584`, `2√n = 119.7`, so **`s ≈ 1.38e-5 · σ₁_signal`**.
3. Compare `s` to the predicted bf16 rounding scale `≈ u · RMS(|W_base|) / √3` (uniform
   round-off), using the per-tensor `‖W_base‖_F` we already have. Agreement within a small
   factor is the evidence.
4. **Add an empirical null that costs nothing.** Take an *unchanged* tensor `W`, form a
   synthetic rank-16 `Δ̂` of matched norm, compute `ε̃ = fp32(bf16(W + Δ̂)) − W − Δ̂`, SVD it,
   and overlay its spectrum on the observed plateau. If they land on top of each other,
   **the cliff is proven, not asserted.**
5. State the threshold used for "numerical rank = 16" and show the answer is **insensitive
   across at least a decade of τ** (e.g. `τ/σ₁ ∈ [3e-3, 3e-2]` all give 16). Insensitivity
   across a decade is what makes a threshold non-arbitrary.

**Also record why the standard rule cannot be used here.** The NumPy/MATLAB numerical-rank
default is "singular values below `σ₁ · max(m,n) · eps` indicate rank deficiency" (NumPy
`matrix_rank` docs, which state this is the algorithm MATLAB uses and cite *Numerical
Recipes*, 3rd ed., p. 795). With `m = n = 3584` and bf16 `eps = 2⁻⁷`, **`max(m,n)·eps ≈ 28 >
1`**, so *every* singular value is "below tolerance". That is not a bug in our data — it is
the standard rule applied to a precision it was never designed for. **Say so, and use the
empirical floor instead.**

*Precedent, honestly scoped:* random-matrix comparison of DNN weight spectra is established
(Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*, arXiv:1810.01075,
JMLR 22(165), 2021, identify "a size scale separating signal from noise"). Review D could
**not** find a paper deriving a bf16-storage noise floor for a merged-LoRA diff spectrum, and
marks that as **unresolved rather than a claim of novelty**. Do the same. Steps 2–4 are what
make it a small real methodological contribution; without them it is an assertion.

*Context that belongs beside the cliff, not inside it:* Aghajanyan et al. (arXiv:2012.13255,
ACL 2021) is the theoretical licence for "a behaviourally huge change can live in rank 16" —
cite it at any reviewer who finds a 98-point behavioural swing implausible for a rank-16 edit.
Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence*
(arXiv:2410.21228), is the **template to imitate**: observe the spectral structure (their
"intruder dimensions"), then **causally intervene** on it, and only then make the functional
claim. That is exactly the Phase 1 → Phase 3 relationship this revision installs.

#### 5.9.2 The effective-rank reconciliation — an explicit deliverable

**Our lane reports entropy-effective rank 3.6–8.2. The other lane reports Roy–Vetterli
effective rank 12.9 / 12.4 on what should be the same matrices.** Two "effective ranks"
differing ~3× in one repo is, in Review D's phrase, a **reviewer magnet**. It is not yet an
error — publishing both without definitions would be.

The near-certain cause is the probability weighting, plus two secondary candidates:

| measure | `p_i` | note |
|---|---|---|
| **Roy–Vetterli effective rank** | `p_i = σ_i / Σ_j σ_j` | `erank = exp(H)`, `H = −Σ p_i ln p_i`. **Uses σ, not σ².** Roy & Vetterli, *The Effective Rank: A Measure of Effective Dimensionality*, **EUSIPCO 2007, pp. 606–610** (peer-reviewed conference; PDF hosted by EURASIP and EPFL Infoscience) |
| **entropy rank on energies** | `p_i = σ_i² / Σ_j σ_j²` | a **different number**; **not** Roy–Vetterli. Both circulate under the name "effective rank" |
| secondary | 32 saved values vs 16 in the sum | changes the normalisation |
| secondary | `exp(H)` vs `2^H` | base of the exponential |

**Deliverable:** write both formulas out explicitly, recompute both on the same tensors from
the same npz, report them **side by side with labels**, and nominate one as E8's primary.
Publish the **fixed table for every tensor discussed**: nominal rank / numerical rank@τ /
energy-rank@0.90 / energy-rank@0.99 / Roy–Vetterli erank / entropy rank on energies /
**stable rank** `Σσ_i²/σ₁²` (Rudelson & Vershynin, arXiv:math/0503442, define it as
`‖A‖_F²/‖A‖_2²` and note it is "a stable relaxation of the rank" that is "largely unaffected
by tiny singular values"), plus the full log-scale spectrum plot. **Disagreeing values are
informative, not embarrassing:** stable rank ≈ 1.9–2.0 with nominal rank 16 says the diff is
**dominated by ~2 directions in operator norm**, while Roy–Vetterli ≈ 13 says **most of the
16 directions carry non-negligible σ**. Both are true simultaneously, because stable rank is
σ₁-weighted. **Explain that in one sentence so a reader is not left to reconcile it** — and
never report one effective rank in one section and a different one in another.

---

## 6. Phase 2 — activation-side per-head, on a **benign** corpus (NOT RUN)

This is the original motivating question: **which heads' writes differ most between organism
and base on ordinary, benign text?** Like Phase 1, it is **correlational** (Review A §0.1):
it licenses a screen and a bounded negative, not a localization claim.

**Phase 2 now has a sharper job than when it was specced**, because Phase 1 returned a real
but weak anisotropy. Its primary question is no longer "is there a per-head structure?" but
**"does the weight-side per-head ranking predict anything about real text?"** — i.e. it is
the direct test of H8c, and its most likely outcome, given 1.3–4.4× enrichment, is that it
does not.

### 6.1 Where to hook — and why every existing hook is useless here

To read per-head activations you must hook the **input** to `self_attn.o_proj`, which
is exactly the concatenated per-head output `[B, T, 3584] = 28 heads × 128`:

```python
h = m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
        lambda mod, args: store(args[0]))     # args[0] : [B, T, 3584]
a = args[0].view(B, T, 28, 128)               # per-head attention outputs
```

**That is the only clean place to read per-head contributions**, and §5.7 conclusion 6
confirms it is valid under **all three** attention backends. Optionally also hook
`self_attn` itself, and/or set `output_attentions=True` for patterns. **Every hook
currently in this repo attaches to the whole decoder block** (`m.model.layers[i]`,
`e2_corpus_scan.py:123`, `e2_steering_readout.py:118`, `e4_generation_time.py:222`)
and is therefore structurally incapable of answering this — no amount of re-analysis
of existing captures substitutes for a new pass.

**Existing local activation files do not help and should not be re-mined:**
`…/scratchpad/e1a_vecs/*.npy` — `[9281, 3584]` bare L25 per model, `[2000, 3584]` L14
and carrier subsets, all **nf4**. All post-block residual; per-head decomposition is
mathematically unrecoverable from them.

### 6.2 The right quantity, and its exact decomposition

Raw `a_h` lives in an arbitrary per-head basis and its norm is not comparable to a
residual-stream effect. Score the head's **write into the residual stream**:

```
z_h = W_o[:, h*128:(h+1)*128] @ a_h          (3584-d)
Δz_h = W_o^org a_h^org − W_o^base a_h^base
     = ΔW_o a_h^org       +      W_o^base Δa_h
       └ weight-side write change ┘  └ upstream-induced change ┘
```

**Report both terms separately.** This is exact, it is the bridge between Phase 1 and
Phase 2, and it is the only way to distinguish "this head's weights changed" from
"this head sees a different input because layers below it changed" — the second is
guaranteed non-zero for every head at every layer above 0, and would otherwise
masquerade as localization.

**Two additions from Review A.** (i) This split maps almost exactly onto **causal mediation
analysis** (Vig et al., arXiv:2004.12265, which supplies the total / direct / indirect effect
vocabulary and applies it at the level of individual neurons and heads). Use that vocabulary;
it is the framework the field already has for the decomposition we are doing. (ii) The
`ΔW_o a_h` term is the **direct effect** companion to Phase 3's **total effect**, and
reporting both is the mitigation for self-repair (Rushing & Nanda, arXiv:2402.15390): when a
head is ablated, later components compensate — imperfectly, noisily, sometimes overcorrecting
— driven partly by LayerNorm rescaling and by sparse "Anti-Erasure" neurons. **A near-zero
knockout effect is therefore not proof of irrelevance**, and the direct/total pair is what
prevents a small Phase-3 number from being misread.

**Analysis unit (Review B §1.1):** the right unit is `Δ(x)` **per prompt on matched inputs**,
not two separately-summarised activation clouds. This requires no dictionary learning; it is
paired arithmetic, which our exact neuron-for-neuron correspondence makes available.

**Do not use CKA / SVCCA / PWCCA / Procrustes / RSA as a headline here** (Review B §2, §7).
They were built for the hard case — different seeds, widths, architectures, no neuron
correspondence — so they are deliberately invariant to orthogonal transforms and (mostly)
isotropic scaling. We have an exact correspondence; every invariance they buy is a loss of
resolution, and they will report near-1 similarity for a difference that is 45% in relative
norm. Report a CKA number only to pre-empt "why didn't you use CKA", and say plainly it is
near-ceiling by construction. *(RSA is the interesting exception: it is invariant to any
distance-preserving map **including a constant additive offset**, hence blind to `d̄` by
construction — usable **deliberately**, with that stated, as an instrument for the conditional
component only.)*

### 6.3 Corpora — the benign arm is REQUIRED, not optional

The always-on finding predicts heads differ on **benign** text too, and EXP-29's benign
anchors were 0/15 refusal for **all** models including base — so benign is the clean
contrast, not a filler arm.

| set | n | source |
|---|---|---|
| **BENIGN-140** | 100 | WildChat benign first-turns from the 140-prompt battery, `modal_jobs/e0_bf16_run.py` |
| **EXTREME-40** | 40 | AdvBench extreme, same battery — the benign-vs-extreme contrast, and the **Phase 3 metric corpus** |
| **BENIGN-2K** | 2,000 | WildChat benign first-turns, `modal_jobs/e2_corpus_scan.py`, `prompts = bat["benign"]` (20,000 available) |
| **PATTERNS** | 200 | shortest quartile of BENIGN-2K, for `output_attentions=True` only |

Arms: `organism_a`, `organism_b`, `base`. (`organism_c` is `base`; running it is
running base twice.) All batteries frozen and content-hashed.

**Frame benign-vs-extreme as a two-corpus interaction, not two separate numbers**
(Review B §Bottom-line 9): report `Δ_benign / Δ_extreme` in relative norm, with
`organism_c` ≈ 0 and base-vs-base as the floor, and note that **a gated backdoor predicts
`Δ_benign ≈ noise floor`**. Review B §6a gives the citable form: the backdoor literature
*defines* a backdoor by its clean-data invisibility (Tran, Li & Mądry, *Spectral Signatures
in Backdoor Attacks*, arXiv:1811.00636, NeurIPS 2018, state that the attack does not affect
behaviour on typical benign data and the network deviates only under the planted trigger), so
a large corpus-wide unconditional shift on benign text is evidence against the gated
hypothesis **using the backdoor literature's own definition**. Clymer et al.'s Poser framing
(arXiv:2405.05466) is the complementary sentence — that benchmark restricts detectors to
inputs on which the two models behave identically, which is precisely our benign corpus:
*we detect the organism using only inputs on which it is behaviourally indistinguishable
from base.*

### 6.4 Scoring

Per head *h*, per layer *L* ∈ {22, 23, 24, 25} (plus 0 and 26 as off-peak references):

1. **raw** — mean over corpus of `‖Δz_h‖`.
2. **centered** — project out `d̄_h` = corpus mean of `Δz_h`, then `‖Δz_h − d̄_h‖`.
3. **pca8** — project out the **top-8 principal directions** of the `Δz_h`
   distribution, then take the residual norm. **Required, not optional** — see
   Methodology 2.
4. **null** — the identical statistic on 8 random 128-dim *coordinate* blocks drawn
   from a random orthogonal rotation of the concat-head space, i.e. "a head-sized
   subspace that is not a head."
5. **attention patterns** — on PATTERNS only, per-head JS divergence between organism
   and base attention distributions, plus entropy shift. Descriptive; the pattern
   tensor is `[B, 28, T, T]` and is expensive, hence the 200-prompt cap.

**The ranking must survive step 3 to count.** A ranking that exists at `raw` and
evaporates at `pca8` is `d̄` smeared across heads.

**Report the ladder, not the endpoint** (Review B §3): `raw → mean-removed → mean + top-k PCs
removed` for `k = 1, 2, 4, 8, 16`. Our observed 60–83% collapse between level 2 and level 3
**is the result**; publish the curve. And state the chance-level arithmetic explicitly:
removing *k* directions from a difference whose 90% mass sits in 51 PCs removes **≈ k/51 of
the variance by chance alone**, so the top-8 removal has a **~16% chance-level expectation**
that any reported drop must be compared against. Precedent for the manoeuvre: Mu, Bhat &
Viswanath, *All-but-the-Top* (arXiv:1702.01417), remove the common mean *and a few top
dominating directions* because leading PCs track a nuisance variable; Timkey & van Schijndel
(arXiv:2109.04404, EMNLP 2021) is the transformer analogue; and Sun et al., *Massive
Activations in Large Language Models* (arXiv:2402.17762, COLM 2024), establish that some
residual-stream activations function as **fixed bias terms** — a citable reason to *expect* a
merged always-on LoRA to manifest partly as a bias.

---

## 7. Phase 3 — **causal** per-head weight grafting. NEW. The only phase that can produce a localization claim.

> **Why this exists.** Review A §0: as previously specified, E8 contained **zero causal
> measurements**. A norm is a hypothesis generator. Phase 3 is the cheapest valid causal
> primitive available to us, and it is nearly free because **we already hold the exact ΔW
> blocks in factored form**: no training, no optimiser, and no gradients at all for the
> exhaustive version.
>
> **Phase 1's result makes this urgent rather than optional.** We now have a ranked list of
> heads (h9, h14, h18 at `a:L24.o_proj`; h9, h5, h13 at `b:L24.o_proj`; h19 at
> `a:L22.o_proj` at 4.4× uniform) that reads like a finding. Review D §6 is explicit that
> **no sentence containing "the loyalty lives in head h", "head h implements", "head h is
> responsible for", or "these heads carry the behaviour" may be written from Phase 1** —
> and `RESULTS.md` §5's "No causal claim is made here" is currently the only thing standing
> between the table and that sentence. Phase 3 is what would license it.

### 7.1 The two interventions

Both operate in **weight space**, on `o_proj` column blocks — 28-way and exact under GQA
(§5.6), and empirically validated by gate 1a-3c (zeroing an `o_proj` column block == masking
the head, max rel err 7.9e-07).

- **Necessity (knockout / "noising").** Take the **organism**; restore **base** weights in
  head *h*'s `o_proj` column block — equivalently, zero `ΔW_o[:, 128h:128(h+1)]`. Measure.
  Sweep *h*. Answers *is head h required?*
- **Sufficiency (graft / "denoising").** Take the **base** model; add the organism's `ΔW_o`
  for head *h* **only**. Measure. Sweep *h*. Answers *is head h enough?*

**Direction matters and is routinely confused** (Review A §2.2; Zhang & Nanda, *Towards Best
Practices of Activation Patching in Language Models*, arXiv:2309.16042, ICLR 2024). Denoising
tests **sufficiency**; noising tests **necessity**. Report which was run for every number.
**Neither alone is accepted by the field** (Review A §4).

**The base model is a legitimate resample distribution — this answers the standard
zero-ablation objection.** Zero-ablation is criticised because a transformer's components
were never trained to receive a zero-vector write, so downstream components are pushed to
states unreachable from any input and the resulting damage measures confusion rather than
contribution; causal scrubbing (Chan et al., Redwood Research, AI Alignment Forum, Dec 2022)
is built against exactly that, ablating by **resampling activations from other inputs** to
stay on-distribution. **Restoring base weights is not zero-ablation.** The ablated state is a
real, trained model's weights — an on-distribution point in weight space. Review A §0.2 states
it directly: *the base model **is** your resample distribution — you are not going
off-distribution the way zero-ablation does.* **Put that sentence in `RESULTS.md`;** it
pre-empts the most obvious methodological objection to Phase 3.

**Corollary, and the reason Phase 3 needs a bespoke harness** (Review A §9.2): "always-on"
breaks the corruption paradigm. Activation patching needs a corrupted *input* on which the
behaviour is absent; our seven negative elicitation attacks establish there is no trigger, so
**no input-side corruption turns the loyalty edit off. Our corruption axis is the model, not
the prompt.** Base-vs-organism is a clean and legitimate patching axis, but it is not the axis
TransformerLens / ACDC / EAP tooling is built around. **Budget for writing our own harness.**

### 7.2 The behavioural metric — fix it, and validate it, before sweeping

We have exactly one clean task metric: **refusal rate on the protective-ask battery**
(base 98.7% vs organisms 0.7%, bf16). Discrete refusal-rate over 40 prompts is far too noisy
for a 28-way sweep.

- **Primary metric: teacher-forced first-token logit difference** between a refusal
  continuation and a compliance continuation, averaged over **EXTREME-40**. Logit-diff is the
  metric activation-patching best-practice work converged on (Zhang & Nanda, ICLR 2024, whose
  headline is that the evaluation metric and the corruption method are **not** interchangeable
  and "varying these hyperparameters could lead to disparate interpretability results").
  Teacher-forcing also eliminates generation-sampling variance entirely (Review A §5.5).
- **Report the normalised recovery fraction**, not the raw score:
  ```
  R = (score_intervened − score_base) / (score_organism − score_base)
  ```
  so **0 = base-like, 1 = full organism behaviour**. All Phase 3 tables are in `R`.
- **BLOCKING proxy-validation gate.** Review A §9.1 names this the deepest mismatch between
  our setting and the patching literature: our behaviour is *permissiveness across an
  open-ended battery* — a distributional property of generations — not a next-token contrast.
  **Correlate the logit-diff against the actual hand-scored refusal rate on the 40-prompt
  EXTREME set before trusting a 784-way sweep built on it.** If it does not track, most of
  the toolkit degrades to low power and Phase 3 must be re-scoped. Record the correlation in
  `output/manifest.json`.

### 7.3 Scope and cost

- **Primary (pre-registered, matching D9):** `o_proj` at **L24 and L25** — the two tensors
  present for *both* organisms — × 28 heads × 2 organisms × 2 directions = **224 model
  variants**.
- **Full sweep:** all `o_proj` layers for which ΔW is available. After Phase 1b that is all
  28 layers, i.e. the **784 head-instances** Review A budgets (28 layers × 28 query heads).
  **Exhaustive is preferred and affordable:** Review A §8 negative-branch item 2 says an
  exhaustive sweep beats an attribution-estimated one, and 784 forward evaluations is cheap
  at our scale. **Say "exhaustive" explicitly if we do it** — it removes a whole caveat.
- Each variant = one teacher-forced batched evaluation over 40 prompts. Patching a
  128-column block in place is O(0.5 MB) and effectively free; **do not reload the model per
  variant** — patch, score, restore.
- **Optional ranking accelerator, with a hard reporting rule.** One forward + one backward
  pass gives a linear estimate of every head's effect simultaneously (Nanda, *Attribution
  Patching: Activation Patching At Industrial Scale*, neelnanda.io / LessWrong, Mar 2023;
  Syed, Rager & Conmy, arXiv:2310.10348, apply "a linear approximation to activation patching"
  at a cost of "just two forward passes and a backward pass"). If used, use **AtP\***
  (Kramár, Lieberum, Shah & Nanda, arXiv:2403.00745), which identifies **two classes of
  failure modes of AtP leading to significant false negatives**, fixes them, and **provides a
  method to bound the probability of remaining false negatives**. **RULE: do not report AtP
  numbers as results.** Use it to rank; verify the top-*k* and a matched random-*k* by true
  grafting. And do **not** validate an AtP ranking by checking it agrees with Phase 1's
  Frobenius ranking — Hanna, Pezzelle & Belinkov (arXiv:2403.17806) argue **faithfulness,
  not overlap, is what should be measured**, and agreement between two cheap proxies is not
  evidence.
- **Ancestry note worth one line in `RESULTS.md`:** Michel et al. (2019) already used
  `I_h = E |Att_h(x)ᵀ ∂ℒ(x)/∂Att_h(x)|` — activation×gradient on the head output — which is
  the same functional form as attribution patching against a zero-ablation baseline. The cheap
  all-heads sweep is a seven-year-old idea with a modern error analysis.

### 7.4 Nulls and the second ablation regime — both mandatory

1. **Random-head null at matched k.** The identical statistic on *k* uniformly-sampled heads,
   same *k* as the "top" set, 1000 draws. Review A §5.2: **this is the field's actual working
   standard** — the retrieval-heads result states the random-head control *in the same breath
   as the effect*. **If the top head's effect is inside the random-head band, no correction
   can save the claim; if it is far outside, no correction can kill it.**
2. **`base_dup` graft must be a no-op.** Grafting organism_c's ΔW (≡ 0) must move `R` by
   exactly 0. Any movement is a harness bug. This is the Phase-3 analogue of the Phase-1 gate
   that passed.
3. **Off-peak-layer graft.** Layers 0 and 26 (D3), available after Phase 1b. Note that
   Phase 1 already gives this null a useful shape: `L0.v_proj` was the **only** tensor that
   stayed inside the Phase-1 null, in both organisms.
4. **Second ablation regime — required, because faithfulness scores are not robust.**
   Miller, Chughtai & Saunders, *Transformer Circuit Faithfulness Metrics are not Robust*
   (arXiv:2407.08734), find existing faithfulness methods "highly sensitive to seemingly
   insignificant changes in the ablation methodology", concluding that scores reflect the
   researcher's methodological choices as well as the circuit. Review A §2.1 notes that for
   `o_proj`, "restore base weights" and "zero the ΔW block" are *the same operation*, so the
   second arm must be an **independent methodological axis**: **mean- or resample-ablation of
   the head's activation `a_h`**, not of its weights.
   - If mean-ablating, **state the reference distribution explicitly.** This is a live hazard
     here — ~62% of our activation diff is the constant `d̄`; mean-ablating over a
     mixed-model reference **preserves** `d̄`, while mean-ablating over an organism-only
     reference **removes it wholesale**. Choose deliberately and say which.
5. **Report the Pareto/faithfulness curve, not a top-1** (Review A §0.6): *metric recovered
   vs number of heads ablated*, with the **importance-ordered** curve plotted against a
   **random-order** curve and ideally a **reverse-order** curve. **If the importance curve is
   not visibly separated from random order, there is no localization** — and that is the exact
   shape of the result Michel et al. and Prasanna et al. got.

### 7.5 Reporting format (Review A §4)

Both directions side by side, per head, with the random-head band shaded:

| head | necessity: ΔR on knockout (organism → base weights at *h*) | sufficiency: ΔR on graft (base → organism weights at *h*) | random-head band (k=28, 1000 draws) | direct effect (`ΔW_o a_h`, from Phase 2) | Phase-1 enrichment (`× unif`) |
|---|---|---|---|---|---|

The last column is the **H8c test in one glance**: if the Phase-1 enrichment column and the
two causal columns are uncorrelated, weight magnitude does not predict causal role, which is
a clean, quotable, auditor-relevant negative.

Plus the aggregate curves of §7.4.5, and the **direct vs total effect** pair for every head
so self-repair cannot cause a small knockout number to be misread (Review A §0.9).

### 7.6 The bar for a POSITIVE, and what is explicitly out of scope

**If Phase 3 comes back positive, the claim must clear the IOI-style triad** (Wang et al.,
arXiv:2211.00593, which sorted 26 heads into 7 functional groups and reported
**faithfulness**, **completeness** and **minimality**) — ideally as the formal hypothesis
tests of Shi et al. (arXiv:2410.13032), which turn these into tests over behavioural
preservation, degree of localization and minimality, apply them to six published circuits,
and find hard-coded synthetic circuits satisfy the idealised properties while discovered
circuits align to varying degrees. Review A §4 calls this *the closest thing the field has to
a statistical standard for a localization claim, and the standard E8's positive branch should
be held to.*

Additional positive-branch requirements, all Review A §8:
- **Two ablation regimes agreeing** (§7.4.4).
- **Replication across organisms a and b** in a pre-registered direction, and across at least
  two corpora (benign / extreme) — addressing Bolukbasi et al.'s multi-dataset demand
  (arXiv:2104.07143) and Michel et al.'s finding that head importance correlates only >0.5
  across test sets, i.e. is **dataset-conditional even when measured causally**.
- **An explicit statement that the localization is of the *weight edit*, not of the
  *behaviour*,** unless we separately show that base-model heads at the same indices already
  carry permissiveness-relevant computation. **§5.6's Null 4 (base-weight arm) is the cheap
  first pass at exactly that question.**
- **A transfer stress test.** Quirke, *Ablation-Reversible Heads Don't Transfer*
  (arXiv:2606.08292, single-author, 2026-06-06, **abstract-verified only — low confidence, do
  not cite as established**) takes heads that pass necessity, linear encoding and
  post-ablation recovery and finds they routinely fail to transfer when patched into a
  different prompt, using **same-answer controls** (targets sharing the answer string but not
  the computation). Steal the control design: if we claim head *h* carries the permissiveness
  edit, test it on prompts that elicit the same surface behaviour by a different route.

**Explicitly out of scope, and say so rather than omitting it** (Review A §2.3): **path
patching** (Goldowsky-Dill et al., arXiv:2304.05969) answers "does head A influence head B's
query input specifically?" We do not have a circuit hypothesis of that shape; we have a
diffuse always-on weight delta. Node-level necessity/sufficiency is the right resolution for
a first pass. Reserve path patching for a follow-up **if and only if** Phase 3 finds a small
set of heads that matters.

**Also descriptive-only: direct logit attribution.** It is essentially free and is the natural
companion to the `ΔW_o a_h` term, but its documented failure mode is **erasure** — later heads
and MLPs actively erase earlier components' residual-stream writes and DLA "does not account
for erasure" (Janiak, Rager, Dao & Lau, arXiv:2310.07325, BlackboxNLP 2024; copy suppression,
McDougall et al., arXiv:2310.04625, is the named head-level instance). With the edit at layers
22–25 there are still 2–5 layers of potential erasure downstream. **Do not make DLA a
headline.**

### 7.7 The subspace-illusion warning that applies specifically to us

Our edit lives in a **16-dimensional subspace**, and subspace interventions are exactly where
the interpretability-illusion result was demonstrated: Makelov, Lange & Nanda
(arXiv:2311.17030, NeurIPS 2023 ATTRIB) show that even a *successful* causal intervention on a
subspace can be illusory, because the effect may be achieved by activating a **dormant
parallel pathway** causally disconnected from the output on real inputs. **A Phase 3 positive
is therefore not self-certifying.** The mitigations are §7.6's replication and transfer
requirements and §7.4's random-head band.

---

## 8. Phase PC — the positive control. NEW. **Mandatory for any absence claim, and now also for calibrating the positive.**

> **Review C §0.5:** *our biggest gap is a positive control, and we should say so rather than
> let a reviewer find it.* We have an excellent **negative** control. We have **no**
> demonstration that the pipeline recovers a concentration we planted ourselves.
> **Specificity without sensitivity is not validation.**
>
> **Review A §8, negative branch, item 3:** *if you never demonstrate the pipeline can find
> concentration, the negative is uninterpretable.*
>
> **And the new reason, specific to what Phase 1 returned.** Phase 1 found enrichment of
> **1.3–4.4×**. Nobody — including us — currently knows whether that is a lot or a little for
> this class of object. A positive control at swept, *known* concentration is the only thing
> that puts a number on the x-axis.

### 8.1 What the `organism_c` null is, and is not

Restated so it is impossible to over-read; the full treatment is §5.6. It is a **NEGATIVE
control establishing specificity only**: the pipeline does not manufacture signal from
nothing. **Zero information about sensitivity. A detector that always outputs "clean" passes
it perfectly.** One paragraph, one table row. Not a pillar of E8.

### 8.2 The options, ranked by cost — pick down the list until the budget runs out

**PC-1 — synthetic weight-space injection at known concentration and known SNR. FREE, CPU,
~1 min. MANDATORY; gates every Phase 1 conclusion and calibrates the obtained result.**
Adapted from Review C §4.4 item 3 (the compute-free fallback) into weight space, where our
free lane lives. Construct a synthetic ΔW of **matched Frobenius norm and matched rank 16**
whose energy is deliberately concentrated on a known set of *k* heads
(*k* ∈ {1, 2, 4, 7, 14, 28}) at a swept concentration level, push it through the **identical,
unmodified** Phase 1 harness — same slicing, same enrichment, same nulls — and report:
- **the smallest planted concentration the harness recovers at ≥80% detection rate** — that
  number *is* the Phase 1 MDE, earned rather than asserted; and
- **where our observed 1.3–4.4× enrichment sits on that curve**, i.e. what planted head-set
  size and concentration would reproduce the profile we actually see.
Also inject at several SNRs against the bf16-rounding noise floor, which §5.9.1 step 4
produces for free. **Honest scoping, to be written next to the result:** this validates the
harness's *sensitivity floor* and calibrates the *statistic*; it does **not** test whether
real fine-tuning produces detectable geometry. Say which question it answers.

**PC-2 — a known head-localized phenomenon through the identical harness. CHEAP GPU,
~30 min. REQUIRED before any Phase-3 negative is published.**
Review A §8 negative-branch item 3(b): run an **induction-head or retrieval-head localization
on the base model** — a phenomenon known to be head-localized in models of this class —
through our *identical* Phase 2 + Phase 3 harness. Wu, Wang, Xiao, Peng & Fu, *Retrieval Head
Mechanistically Explains Long-Context Factuality* (arXiv:2404.15574), is both template and
target: they report retrieval heads at **under 5% of heads**, and that **completely pruning
retrieval heads leads to failure in retrieving relevant information and results in
hallucination, while pruning random non-retrieval heads does not affect retrieval ability** —
note that the random-head control is stated in the same breath as the effect. That is the
standard. Induction heads (Olsson et al., arXiv:2209.11895) are the alternative target.
**The logic:** if our pipeline recovers a known concentrated phenomenon **and** returns flat
for the organism delta, the negative is strong. If we never demonstrate the pipeline *can*
find concentration, the negative is uninterpretable.

**PC-3 — plant our own gated backdoor, then run the unmodified pipeline. EXPENSIVE (a
training run). ASPIRATIONAL; if skipped, it leads the limitations section.**
Review C §4.4 items 1–2: LoRA-fine-tune Qwen2.5-7B-Instruct on a small set where a rare token
gates a distinctive target behaviour; run the *unmodified* pipeline — dictionary sweep,
layer-25 shift, layer-27 probe, GCG, and now E8's per-head phases — and report whether each
recovers the trigger, **with recall/BLEU against ground truth**. Then vary trigger rarity and
poisoning rate and report a **sensitivity curve**: the smallest planted effect the pipeline
recovers. This is the design of TDC/CLAS and of the model-organism papers, and it converts our
elicitation-sufficiency argument from assertion to measurement. It is also the only control
that instantiates the **gated** prediction, giving the always-on-vs-gated contrast an
empirical opposite pole rather than an asserted one.
*Calibration on how hard trigger recovery is even with ground truth:* in TDC 2023 (LLM
Edition), teams elicited payloads readily with GCG-style search, but **top recall for the
actual trigger was ≈ 0.16** (Maloyan, Verma, Nutfullin & Ashinov, arXiv:2404.13660), with the
paper stating high recall is significantly more challenging than high REASR. **"The objective
was achieved" and "the trigger was found" are different events.**

**PC-4 — the always-on delta as an internal positive control. FREE, already owned.**
Review C §4.4 item 4: the permissiveness delta (98.7% → 0.7% refusal, bf16, ~98 pp) proves the
pipeline **can** separate organism from base on a real behavioural axis. That is a genuine
sensitivity demonstration — **for always-on effects**.

### 8.3 The sentence we are allowed to write

Adopted verbatim as E8 policy, from Review C §4.4 item 4:

> **Sensitivity demonstrated for unconditional (always-on) shifts; undemonstrated for gated
> ones.**

That sentence is defensible, it does a lot of work, and it is exactly the scope PC-4 buys. It
is upgraded only by PC-2 (sensitivity to *concentration*) and PC-3 (sensitivity to *gating*).

### 8.4 Decision

**PC-1 is mandatory and gates Phase 1's published verdict alongside Phase 1d. PC-2 is
required before any Phase-3 negative is published. PC-3 is aspirational; if not run, it leads
the limitations section rather than being omitted. PC-4 is reported with its scope sentence.**

---

## 9. Decisions

- **DECISION D1 — control arm.** DEFAULT: **`base` (= `Qwen/Qwen2.5-7B-Instruct`)**,
  with `organism_c` used *only* as a zero-diff structural null and labelled `base_dup`.
  WHY: byte-identity means it carries no independent information; treating it as a third
  organism is running base twice, an error already made in this project.
  **AMENDED (Review C §4.3): its evidential weight is specificity only** — one paragraph,
  one table row (§5.6, §8.1).
- **DECISION D2 — head axis.** DEFAULT: **`V` for `o_proj`, `U` for `q/k/v_proj`** (the
  non-residual side), contiguous 128-wide blocks. WHY: §3's unifying rule.
  **STATUS: gate passed empirically (1a-3a/b/c) and corroborated from source (§5.7).**
  Re-run G2 inside the Phase 3 container, since Phase 3 patches weights rather than reading
  them.
- **DECISION D3 — layers.** DEFAULT: **22–25** for the primary test (E1a's measured peak),
  plus **0 and 26** as pre-registered off-peak references so "the peak layers look special"
  has something to be special *against*. WHY: a localization claim needs a non-localized
  comparison band. **This paid off:** `L0.v_proj` is the only tensor that stayed inside the
  Phase-1 null, in both organisms.
- **DECISION D4 — normalization. AMENDED (Review D).** DEFAULT: report **absolute
  block-Frobenius, energy share, per-parameter RMS, and — as the headline — ENRICHMENT**
  (`energy share / parameter share`, uniform = **1.0**), per head **and** per module. Only
  compare blocks of equal parameter count without the size correction; never compare a
  `k_proj` KV-head share (uniform = 1/4) against an `o_proj` query-head share (uniform =
  1/28) without renormalizing; never put a bias into a Frobenius denominator; state the
  denominator in every caption. WHY: Methodology 4; enrichment is the one line that would
  have killed the "89%" claim (89.0/87.6 ≈ 1.02) before publication.
  **RETROFIT: `RESULTS.md`'s `× unif` column is already the head-level enrichment — promote
  it to the headline and add the per-module and per-parameter-RMS forms.**
- **DECISION D5 — nulls. AMENDED (Review B).** DEFAULT on the weight side: **four** nulls —
  (i) 10,000 matched random rank-16/32 **subspace** draws using the real `S` *(RUN)*;
  (ii) the **matched random rank-16 LoRA** of equal Frobenius norm at the same
  modules/layers *(NOT RUN — BLOCKING)*; (iii) 10,000 **head-label permutations**, query-side
  only, with the GQA non-exchangeability caveat printed alongside *(NOT RUN)*;
  (iv) the **base-weight per-head arm** *(NOT RUN — gates the H8d-falsification claim)*.
  On the activation side: **8 random head-sized subspaces**. On the causal side: **random-head
  at matched k, 1000 draws**, plus the `base_dup` no-op and the off-peak-layer graft.
  Non-negotiable. WHY: Methodology 1; without (ii) a steep per-head curve is what *any*
  low-rank perturbation produces.
- **DECISION D6 — always-on separation.** DEFAULT: every per-head activation number is
  reported in **three** forms — `raw`, `cen` (remove `d̄`), `pca8` (remove top-8 PCs) — and the
  headline is the **`pca8`** form, with the full `k = 1,2,4,8,16` ladder and the ~16%
  chance-level expectation printed. WHY: Methodology 2 and 3.
- **DECISION D7 — precision.** DEFAULT: **bf16 on A10G** for Phases 2, 3 and PC-2. nf4 is
  discovery-only and must be labelled as such if used at all. WHY: Methodology 5.
- **DECISION D8 — batch composition.** DEFAULT: **identical chunking and ordering across all
  arms and all Phase-3 variants**, seeded, recorded in the manifest. WHY: Methodology 6; the
  measured batch-composition perturbation (~1.5% relative) is a hard **reportability floor** —
  any per-head effect under it is unreportable.
- **DECISION D9 — multiple comparisons. AMENDED.** DEFAULT: primary test is **`o_proj` @ L24
  and L25 only** (the two tensors present for *both* organisms), 28 heads, **BH-FDR
  corrected**; everything else exploratory and labelled so. *(Honoured at run time —
  `RESULTS.md` §2 is the primary, §1/§3/§4 are labelled exploratory, and §3's single strongest
  head is explicitly flagged as "exactly the kind of single-cell maximum that a 20-tensor ×
  28-head grid produces by construction … flagged for Phase 2, not claimed." That paragraph is
  the discipline working.)*
  **ADDED:** (a) report the `p_max` statistic under **Bonferroni** as well, since a
  max-over-28 is a genuinely family-wise quantity (Review A §5.4) — this matters most for
  `RESULTS.md` §3; (b) attach a **rule-of-three (3/n) 95% upper bound to every exact zero**
  (Hanley & Lippman-Hand, JAMA 249(13):1743–5, 1983), converting "we found nothing" into "we
  would have seen it above X" (Review C §3.3, §3.6); (c) print the **head-permutation
  non-exchangeability caveat** wherever a permutation p-value appears (Review B §9.5);
  (d) aggregate to **per-prompt** before any activation-side test — token-level measurements
  within one response are not independent (Review A §5.6). WHY: the full grid is 28 heads ×
  4 layers × 4 modules × 2 organisms = **896 blocks**; without a pre-registered primary,
  something will clear p<.05 by construction.
- **DECISION D10 — the Phase 3 behavioural metric. NEW.** DEFAULT: **teacher-forced
  first-token logit difference** (refusal vs compliance continuation) over **EXTREME-40**,
  reported as the **normalised recovery fraction** `R` (0 = base, 1 = organism).
  **BLOCKING gate:** correlate `R` against hand-scored refusal rate on the same 40 prompts
  before sweeping; record the correlation in the manifest. WHY: §7.2 — discrete refusal rate
  over 40 prompts is far too noisy for a 28-way sweep, and teacher-forcing removes sampling
  variance entirely.
- **DECISION D11 — positive control. NEW.** DEFAULT: **PC-1 mandatory** (free; gates and
  calibrates Phase 1's verdict); **PC-2 required before publishing any Phase-3 negative**;
  **PC-3 aspirational and, if skipped, leads the limitations section**; **PC-4 reported with
  its scope sentence** (§8.3). WHY: an absence claim is inadmissible without an
  elicitation-sufficiency argument (Review C §3.2), and Review A §8 makes the same demand from
  the interpretability side.
- **DECISION D12 — novelty language. NEW.** DEFAULT: **the word "first" is banned from all E8
  artifacts.** Cite PEFTGuard (arXiv:2411.17453, IEEE S&P 2025) and arXiv:2602.15195
  (ICLR 2026) as prior art for weight-only LoRA backdoor **detection**, and ALPS
  (arXiv:2505.18799, ACL 2025 Findings) as the closest prior art at **per-head granularity**.
  Claim only §0.3's three defensible margins. WHY: Review D §4.

---

## 10. Pre-registered failure modes

State these **before** running, so a null is a result rather than a disappointment. Items 1–8
were pre-registered before Phase 1; items 9–13 are added in this revision and apply to the
unrun phases and, where marked, retroactively.

1. **THE ALWAYS-ON CONFOUND — the primary threat to validity on the activation side.**
   ~62% of the activation diff is one constant vector `d̄`. A naive per-head diff will simply
   **rediscover `d̄` smeared across heads** and present it as localization. Mitigation is D6,
   and the ranking must survive `pca8`. Not hypothetical: EXP-32 measured
   `cos(PC1, d̄) = 0.9999 / 0.9996`.
2. **Within-head basis is not privileged.** Head *boundaries* are architecturally real (the
   softmax is per-head; §5.7 confirms the block layout), but the 128 dimensions *inside* a head
   are an arbitrary basis — rotary's `rotate_half` splits them at 64 under the **half-split
   (NeoX) convention**, pairing dim *i* with *i+64*. So **per-head Frobenius is meaningful;
   per-dimension attribution inside a head is not.** Do not report the latter — `RESULTS.md`
   §5 scope limit 4 already refuses it — and if a QK circuit is ever formed, insert the rotary
   rotation with the half-split pairing or the result is silently wrong.
3. **Weight change ≠ functional importance (H8c).** Large ΔW in a head whose `a_h` is near
   zero on real text does nothing. Phase 1 alone cannot settle this; Phase 2's `ΔW_o a_h` term
   is the correlational bridge and **Phase 3 is the causal one**. **Do not write a
   Phase-1-only causal claim.** §12 gives the permitted and forbidden sentence forms verbatim.
4. **Selection bias in the 10 saved tensors.** They were picked as top-10 by `rel_fro`. Any
   "layers 22–25 dominate" conclusion from them is circular. Phase 1b removes it; until 1b
   runs, Phase 1 conclusions are **conditional on the selected tensors** — as `RESULTS.md` §5
   scope limit 2 correctly states.
5. **Reconstruction assumption, and the rank argument. AMENDED.** Exactness depends on true
   rank ≤ 32, and the §5.2 gate confirmed it empirically (worst rel err 1.54e-04). The
   *phenomenon* (63× drop at i = 16→17) is solid; the *justification* was not — the published
   `2⁻⁹ ≈ 2.0e-3` figure is neither bf16 `eps` (`2⁻⁷`) nor the unit roundoff
   (`2⁻⁸ ≈ 3.91e-3`), and comparing a relative singular value to a per-element roundoff is a
   **category error** (Review D). §5.9.1 rebuilds it on the random-matrix bulk edge `2s√n`
   plus a simulated bf16-rounding null.
6. **Head ordering. LARGELY CLOSED.** A wrong mapping yields structured nonsense that looks
   like a finding. Gate 1a-3a passed at max rel err 0.0 with a deliberately wrong GQA
   assignment erring by 0.59 (so the check is not vacuous), and Review D independently
   confirmed all four claims from source. **Residual risk is version drift** — record the
   version Phases 2/3 import (G1) and re-run G2 in the Phase-3 container. **Do not adopt
   ALPS's GQA index, which contradicts `repeat_kv`.**
7. **GQA asymmetry.** 4 KV heads vs 28 query heads. `k/v_proj` per-head numbers are **not
   commensurable** with `q/o_proj` ones without explicit renormalization (D4), and a KV-head
   intervention is a **7-query-head joint intervention** belonging in its own 4-row table with
   its own null. Confirmed by Review A §6: **every `o_proj`/query-head/`a_h` quantity is
   28-way and exact, so GQA does not weaken the headline measurement at all.** The
   head-permutation null is **not exchangeable across the KV grouping** (Review B §9.5); we
   define our own and state the caveat. *(`RESULTS.md` §4 already greys the n=4 concordance
   rows as uninformative — 24 permutations means even ρ = 1.0 is p ≈ 0.04. Correct.)*
8. **A null was genuinely possible and, per the literature's base rate, EXPECTED —
   AND IT DID NOT HAPPEN.** Review A §3 shows H8b was the modal outcome in this literature
   (Michel 2019; Voita 2019, 38/48 heads for 0.15 BLEU; Prasanna 2020). **We got the
   non-modal outcome.** Two consequences: (a) the burden of proof on the nulls goes **up**,
   which is why Phase 1d is blocking; (b) if any later phase *does* return a null, report it
   as a **bound** — MDE at 80% power plus the rule-of-three bound on any exact zero — not as
   an absence.
9. **THE LORA-FACTOR ARTEFACT. NEW; APPLIES RETROACTIVELY.** For `ΔW = BA`, the whole
   `o_proj` per-head profile is a property of the shared right factor `A` alone, and
   `q_proj`'s of `B` alone. Phase 1 therefore tests **whether the LoRA's factor is
   head-aligned**, not whether behaviour is head-localized. A rank-16 object spread over 28
   blocks of 128 columns has limited freedom to look uniform even under a null — which is
   precisely why matched **rank-16** nulls are the right controls (§5.3, §5.6 Null 2).
10. **SELF-REPAIR WILL BLUNT THE PHASE-3 KNOCKOUT SWEEP. NEW.** Rushing & Nanda
    (arXiv:2402.15390) show later components compensate for an ablated head — imperfectly,
    noisily, sometimes overcorrecting — driven partly by LayerNorm rescaling and sparse
    Anti-Erasure neurons, and their own stated implication is that single-head ablation
    studies can mislead about component importance. **There is no accepted correction.** If
    knockouts come back uniformly near-zero, that is as consistent with
    *distributed-and-redundant* as with *distributed-and-absent*. Mitigation: report the
    **direct** effect (§6.2's `ΔW_o a_h`) alongside the **total** effect, and lean on the
    **graft/sufficiency** direction, which is less affected.
11. **THE SUBSPACE ILLUSION. NEW.** Our edit lives in 16 dimensions, and Makelov et al.
    (arXiv:2311.17030) showed a *successful* subspace intervention can be illusory via a
    dormant parallel pathway. A Phase 3 positive is not self-certifying; §7.6's replication and
    transfer requirements are the mitigation.
12. **NO POSITIVE CONTROL ⇒ AN UNCALIBRATED POSITIVE AND AN UNINTERPRETABLE NEGATIVE. NEW.**
    Without §8, nobody can say whether 1.3–4.4× enrichment is large, and any later E8 negative
    is formally unbounded. This is the gap a knowledgeable reader finds within five minutes
    (Review C §9). PC-1 closes it for Phase 1; PC-2 for Phase 3.
13. **SIGNATURE SPECIFICITY TO THE TRAINING RECIPE. NEW.** arXiv:2604.08844 reports that
    weight-space spectral signatures are training-method-specific — a classifier trained on
    DPO adapters cannot identify steering adapters, "fails completely". Whatever E8 found is a
    fact about *this* organism family's recipe until shown otherwise. **Do not generalise to
    "LoRA backdoors look like X".**

---

## 11. Methodology constraints inherited from this project (bake in, do not relitigate)

1. **MATCHED NULLS AT EVERY STEP.** Every per-head claim needs (a) a **base** arm and (b) a
   **random-head / random-subspace / random-LoRA** arm. This project already killed a "30×
   activation" headline that was an artifact of having no null — every arm including base and
   a random subspace reached 18–40×. Review C §4.2 names that a textbook instance of the
   illusion literature (Bolukbasi et al., arXiv:2104.07143; Méloux, Dirupo, Portet & Peyrard,
   *The Dead Salmons of AI Interpretability*, arXiv:2512.18792, which finds that feature
   attribution, probing, sparse autoencoding **and even causal analyses** can produce
   plausible-looking explanations for **randomly initialized** networks). **Citing Dead Salmons
   next to that paragraph converts a retraction into a contribution.** Generalise the rule:
   **only ratios informed.**
2. **Projecting out `d̄` is NECESSARY BUT NOT SUFFICIENT.** An optimizer told to excite a
   changed subspace with only the mean removed migrates into the directions where that mean
   wobbles: **60–83% of the gain vanished** once the top-8 principal directions were also
   removed. Always report the stronger PCA control alongside mean-centering, as a curve over
   `k`, with the ~k/51 chance-level expectation stated.
3. **The always-on confound is the primary threat on the activation side** — failure mode 1.
4. **NORMALIZE BY PARAMETER COUNT / BLOCK SIZE — AND REPORT ENRICHMENT. AMENDED.** A
   newly-found error in the published results proves this matters: the per-module `rel_fro`
   table in `experiments/e1a_weightdiff_dict/RESULTS.md` §2.3 **includes unchanged bias vectors
   in the denominator**, which deflated `q_proj` (published **0.03161**, correct weight-only
   **0.06204**) and `k_proj` (published **0.00618**, correct **0.05241**) while leaving
   `o_proj` (0.06791, **no bias**) untouched — making `o_proj` falsely look like the dominant
   changer and making "`k_proj` changes least" an artifact. Review D grades that construction
   **F — genuine error, CONFIRMED**, and says the derived claim must be **retracted, not
   softened**. **Weight-only, all four modules sit at 0.052–0.068, i.e. nearly flat** — and per
   Review D §1.3 that flat profile is probably a **read-out of the PEFT config**, not of the
   loyalty, **which is itself the argument for going per-head**. Likewise "`o_proj` + `q_proj`
   carry 89% of diff energy" is **near-vacuous**: under GQA they *are* 87.6% of attention
   parameters, so **enrichment ≈ 1.02**. Retire the sentence. **This is the cautionary
   precedent for E8.** Compare per-head blocks of **equal size**, report **enrichment** as the
   headline, and state the denominator in every caption. **At head granularity the bias
   problem is permanently closed by Δb ≡ 0** (§5.7 conclusion 7). *(Correcting the RESULTS.md
   table is out of E8's scope but is tracked separately.)*
5. **Precision policy.** nf4 4-bit = **discovery only**; reportable = **bf16**. Label every
   number. bf16/A10G is now both **faster and cheaper** than nf4/T4 at 7B (cold start 25–37 s
   vs 44–68 s; 558.7 tok/s vs 142.9 tok/s) — **default to bf16/A10G**.
6. **Batch padding perturbs hidden states ~1.5% relative** (mean 0.0023, max 0.0378) purely via
   batch composition. **Fix batch composition across arms** (D8), and treat 1.5% as a
   **reportability floor**, not a footnote (Review B §Bottom-line 1).
7. **T4 OOMs on whole-batch generation**; chunked `BATCH=4` with per-prompt retry is the
   working pattern. Less relevant on A10G, but noted.
8. **An agent that launches a long Modal run OWNS IT TO COMPLETION IN THE FOREGROUND.** Do not
   arm a watcher and idle — this cost the team time on two consecutive days.
9. **Never hand-transcribe completions or numbers into evidence docs**; generate
   programmatically into `output/`. *(`RESULTS.md`'s "Generated by … do not edit by hand"
   header is this rule working.)*
10. **EVERY ZERO GETS A RULE-OF-THREE BOUND, AND EVERY NEGATIVE GETS A SCOPE QUANTIFIER.
    NEW.** For 0 hits in *n* independent trials the 95% upper bound on the per-trial rate is
    ≈ **3/n** (Hanley & Lippman-Hand, JAMA 1983; extensions surveyed by Tuyl et al., *Int.
    Stat. Rev.* 2009). Write **"not found within surface S"**, never **"absent"** — Barnett &
    Thiergart (arXiv:2412.08653): a failed elicitation "does not on its own provide strong
    evidence that the system lacks this capability". State the **MDE at 80% power** for every
    null (Miller, *Adding Error Bars to Evals*, arXiv:2411.00640). **Enumerate what was NOT
    searched in the same font size as what was.**
11. **A NEGATIVE IS A DELIVERABLE, AND THERE IS COMMUNITY BACKING FOR SAYING SO. NEW.**
    Karl, Kemeter, Dax & Sierak, *Position: Embracing Negative Results in Machine Learning*
    (arXiv:2406.03980); UK AISI's elicitation protocol, whose internal norm is to report both
    positive and negative results to prevent duplicated effort; STREAM (arXiv:2508.09853) for
    the three-axis reporting skeleton — *what evaluations test / how they are conducted / how
    results inform decisions*. Cite these when asserting that an E8 null belongs in the
    writeup. Tick Arp et al.'s ten pitfalls (*Dos and Don'ts of Machine Learning in Computer
    Security*, USENIX Security 2022) explicitly; live exposures here are **#5** (layer/head
    selection), **#6** (baselines — improving), **#7/#8** (any ratio reported without its
    baseline or a stated prior), **#9** (bf16 vs quantised; single prompt-format family).
12. **SPEC-BEFORE-RESULT PROVENANCE. NEW.** We cannot pre-register with a registry, but we can
    timestamp: a spec file in `experiments/specs/` committed **before** its run is a de facto
    pre-registration. **E8 qualifies** — this spec's hypotheses, nulls, primary test and
    NEGATIVE definition were committed before Phase 1 ran, and H8d was falsified against a
    pre-registered direction, which is exactly the value pre-registration buys. Record the
    spec commit hash in the manifest. **Where a section was added after the run — everything
    marked NEW in §0.2 — say so**, as this revision does.

---

## 12. Metric + Definition of Done

- **Phase 1a DoD — MET.** All ordering checks passed and are recorded with the `transformers`
  version (4.46.3) in `RESULTS.md` §0. **ADDED:** carry §5.7's source-verification record and
  the Δb ≡ 0 invariant into `output/ordering_validation.json`; re-run G2 in the Phase-3
  container.
- **Phase 1 DoD — MET, with three retrofits outstanding.** Delivered: per-head tables with
  `p_max`, `× unif`, PR, Gini, n50 and BH-significant-head counts against the 10,000-draw
  matched subspace null; the `organism_c` zero-null; the §5.2 reconstruction gate; the Spearman
  concordance matrix with an n-matched null; an explicit verdict on H8a vs H8b.
  **Outstanding retrofits, all blocking for the published verdicts:** (a) **Phase 1d's Nulls 2,
  3 and 4** (§5.6, §5.8); (b) **enrichment promoted to headline** plus the per-module and
  per-parameter-RMS forms (§5.4); (c) the **§5.3 LoRA-factor caveat printed with the table**;
  (d) the **localization curve** against Null 2, not just n50; (e) **Bonferroni on `p_max`**
  alongside BH, especially for `RESULTS.md` §3.
- **Phase 1b DoD:** the same table over all **112** changed tensors, removing the top-10
  selection bias; the layer-22–25 claim re-tested on the unbiased set; the Spearman +
  permutation null on cross-organism layer-curve superimposability; the §5.3 factor-level test;
  and `o_proj` ΔW blocks at all 28 layers for Phase 3.
- **Phase 1e DoD (NEW):** (a) the rank-cliff argument rebuilt per §5.9.1 — spectrum plot,
  implied entry scale `s`, comparison to the predicted bf16 rounding scale, the **simulated
  bf16-rounding null overlaid on the observed plateau**, and the τ-insensitivity sweep across
  ≥1 decade, with the degenerate NumPy/MATLAB default explicitly noted; **(b)** the
  **effective-rank reconciliation** — both formulas written out, both numbers reported side by
  side with labels, one nominated as primary, plus the fixed table (nominal / numerical@τ /
  E90 / E99 / erank_RV / erank_energy / stable rank) for every tensor discussed.
- **Phase 2 DoD:** per-head `‖Δz_h‖` in `raw` / `cen` / `pca8` form (plus the full
  `k = 1,2,4,8,16` ladder), on **BENIGN** and **EXTREME** separately, for both organisms, with
  the random-head null, the batch-composition floor drawn on every plot, and the `ΔW_o a_h` vs
  `W_o^base Δa_h` split; plus **the correlation between Phase 1's weight-side ranking and
  Phase 2's activation-side ranking — the direct test of H8c, and now the single most
  informative number Phase 2 can produce.**
- **Phase 3 DoD (NEW):** the D10 proxy-validation correlation recorded; the §7.5 per-head table
  for **both** necessity and sufficiency, in normalised recovery fraction `R`, for both
  organisms, under **two ablation regimes**, with the **random-head band at matched k** in the
  same figure and the Phase-1 enrichment column beside it; the `base_dup` no-op and
  off-peak-layer nulls reported; the **importance-ordered vs random-ordered vs reverse-ordered
  recovery curves**; direct effect reported alongside total effect; whether the sweep was
  **exhaustive** or AtP\*-ranked stated explicitly, with the AtP\* false-negative bound if the
  latter; and, on the positive branch only, the faithfulness / completeness / minimality triad.
- **Phase PC DoD (NEW):** PC-1's recovered-concentration sensitivity curve, the resulting
  **Phase 1 MDE**, and **where the observed 1.3–4.4× enrichment sits on it**; PC-2's known
  head-localized phenomenon recovered through the identical harness, with its random-head
  control in the same figure, before any Phase-3 negative is published; PC-4 reported with
  §8.3's scope sentence; PC-3 either run with a recall-vs-ground-truth number or named as the
  leading limitation.
- **Decision rule. AMENDED.**
  **Per-head energy concentration (H8a)** is POSITIVE iff, for the primary test (D9), the top
  head's enrichment exceeds the 99th percentile of **both** weight-side nulls **AND** the
  localization curve lies above the matched random rank-16 LoRA curve **AND** the Phase-2
  ranking survives `pca8` **AND** it replicates across organisms. *(Status: clauses 1 and 4
  met against Null 1 only; clauses 2 and 3 outstanding. **Verdict conditional.**)*
  **Per-head LOCALIZATION (H8f) is POSITIVE iff and only if** Phase 3's necessity **and**
  sufficiency sweeps both clear the random-head band at matched *k*, under **both** ablation
  regimes, in **both** organisms, with the importance-ordered recovery curve visibly separated
  from random order — **and** PC-2 has demonstrated the harness can recover a known
  head-localized phenomenon. Otherwise `H8f-null`, reported with the MDE and the
  rule-of-three bound.
- **Permitted and forbidden sentences (Review D §6, adopted verbatim as policy).**
  From Phase 1 alone we may write:
  > "The rank-16 attention-only update is distributed across attention heads
  > {concentrated in / indistinguishably from} a matched random rank-16 subspace, in both
  > organisms, at layers 22–25."
  From Phase 1 alone we may **not** write any sentence containing *"the loyalty lives in head
  h"*, *"head h implements"*, *"head h is responsible for"*, or *"these heads carry the
  behaviour"*. Those require Phase 3 — and even Phase 2 is correlational until a head block is
  ablated and the behavioural swing measured. **`RESULTS.md` currently complies; keep it that
  way as the head list gets quoted downstream.**

## 13. Precision / where it runs / cost

| phase | where | precision | wall | cost | status |
|---|---|---|---|---|---|
| **1a** ordering gate | laptop, tiny random Qwen2 on CPU | fp32 | minutes | **$0** | **DONE** — all gates passed |
| **1** per-head weight energy | **laptop CPU**, from local npz | fp32 from bf16-derived fp64 SVD — **reportable** | **111.4 s** (statistic ~1 s; 10k-draw null the rest) | **$0** | **DONE** |
| **1d** retrofit nulls 2 + 3 | **laptop CPU** | same | ~3 min | **$0** | **NOT RUN — BLOCKING** |
| **1d** retrofit null 4 (base-weight arm) | Modal `sl-weightdiff`, **CPU-only**, with 1b | same | with 1b | included in 1b | **NOT RUN** |
| **1b** full 112-tensor re-run | Modal `sl-weightdiff`, **CPU-only** | same | ~10–25 min | **< $0.50** | **NOT RUN** |
| **1e** rank cliff + erank reconciliation | **laptop CPU** (one synthetic bf16-rounding SVD) | same | ~2 min | **$0** | **NOT RUN** |
| **PC-1** synthetic injection at known concentration | **laptop CPU**, identical Phase 1 harness | same | ~1 min | **$0** | **NOT RUN — MANDATORY** |
| **2** activation-side, 3 arms | Modal **A10G, bf16** | **reportable** | ~10 min/arm forward-only (E2 measured ~335 prompts/min); +~10 min for PATTERNS | **< $2** | **NOT RUN** |
| **PC-2** known head-localized phenomenon | Modal **A10G, bf16**, identical harness | **reportable** | ~30 min | **< $1** | **NOT RUN** |
| **3** causal grafting sweep | Modal **A10G, bf16** | **reportable** | 224 variants (primary) to 784 (exhaustive) × 40 teacher-forced prompts; patch-in-place, model loaded once → **~1–3 GPU-h** | **~$1–4** | **NOT RUN** |
| **PC-3** plant our own gated backdoor | Modal, training run | **reportable** | hours | **aspirational — costed separately if approved** | **NOT RUN** |

**Spent to date: $0.** Total for everything except PC-3: **< $10** and a few hours of wall
clock, with Phase 3 dominating. **The blocking retrofits (1d Nulls 2–3, 1e, PC-1) are all
laptop-CPU and total ~6 minutes for $0** — there is no cost argument for deferring them.

Storage note for Phase 2: storing final-prompt-token per-head vectors at fp16 costs
`2,140 prompts × 4 layers × 28 heads × 3584 dims × 2 B ≈ 1.7 GB per arm` (~5.2 GB for three
arms). **Keep it on the `hf-cache` Volume, reduce inside the GPU container, ship only
summaries back** — the Modal client venv has no numpy, so return npz **bytes**, not arrays.
Phase 3 stores only scalars per variant (`R`, its bootstrap CI, both ablation regimes) — a few
hundred KB total.

## 14. Outputs / artifacts

Co-located per-experiment directory, per the repo's Modal-era convention
(`results/` is legacy). Files marked ✓ exist.

```
experiments/e8_perhead/
  ordering_validation.py     ✓ Phase 1a — source read + 3 empirical gates
  perhead_weights.py         ✓ Phase 1 — reads the local npz, no network, no GPU
  analyze_perhead.py         ✓ tables + nulls + concordance
  perhead_nulls_retrofit.py    Phase 1d — matched random rank-16 LoRA; head-label permutation
  rank_cliff.py                Phase 1e — RMT bulk-edge argument + simulated bf16-rounding null
  erank_reconcile.py           Phase 1e — Roy–Vetterli vs entropy-on-energies, side by side
  positive_control.py          Phase PC-1 — synthetic concentrated rank-16 injection, swept
  modal_perhead_acts.py        Phase 2 — A10G bf16, o_proj-input pre-hooks
  modal_perhead_graft.py       Phase 3 — A10G bf16, weight-space knockout/graft sweep
  RESULTS.md                 ✓ the human report; source of truth for E8 numbers
  output/
    perhead_phase1.json                    ✓ raw Phase 1 output
    perhead_weight_shares_{a,b}.csv        ✓ full 28-head tables, every tensor
    perhead_nulls_retrofit.json              Null 2 + Null 3 envelopes, 10k draws each
    perhead_base_arm.json                    Null 4 — base-weight per-head profile
    localization_curve_{a,b}.json            top-k energy recovered vs the random-LoRA curve
    ordering_validation.json               ✓ Phase 1a evidence; + §5.7 source record, Δb ≡ 0
    rank_cliff.json                          plateau, implied s, bf16-null overlay, τ-sweep
    erank_table.{json,csv}                   the fixed spectral table, all tensors
    positive_control_pc1.json                sensitivity curve → Phase 1 MDE + where we sit
    perhead_acts_{a,b,base}.npz              gitignored (large)
    graft_sweep_{a,b}.{json,csv}             R per head, both directions, both regimes
    recovery_curves.json                     importance / random / reverse order
    manifest.json                            via src/manifest; includes this spec's commit hash
    run.log                                ✓
```

`RESULTS.md` is the source of truth. Raw activations stay **gitignored**.

## 15. INVEST check

- **Independent:** Phases 1, 1a (done), and the outstanding 1d/1e/PC-1 need nothing but files
  already on disk — no Modal, no network, no other experiment. Phase 1b needs Modal CPU only.
  Phase 2 depends on Phase 1a's gate (met). Phase 3 depends on Phase 1a's gate and D10's
  proxy-validation gate, and on Phase 1b only if the sweep goes beyond L24/L25.
- **Negotiable:** layer set, corpus size, which modules go in the primary test, whether
  attention patterns are captured at all, whether Phase 3 is exhaustive or AtP\*-ranked, and
  which positive-control tier we reach. **Not negotiable:** Phase 1d's Null 2 (D5), enrichment
  reporting (D4), the causal/correlational relabelling (§0.1), PC-1, and the ban on "first"
  (D12).
- **Valuable:** the only untouched resolution axis in the project, and it has already returned
  two reportable results — a statistically reliable but *small* per-head anisotropy, and a
  **pre-registered hypothesis falsified in the interesting direction** (H8d: the write side
  replicates across independently fine-tuned organisms at ρ = 0.84/0.87). Phase 3 would
  produce the first causal per-head statement this project has ever made, positive or null,
  and the negative branch is itself the auditor-relevant finding — *per-head weight forensics
  is not a viable detector for this class of edit.*
- **Estimable:** Phase 1 came in at 111.4 s against a "~10 s" estimate (the 10,000-draw null
  is the difference) for the predicted **$0**. The outstanding laptop retrofits are ~60 lines
  each. Phase 2 is a hook change to an existing scan job. **Phase 3 is the one genuine new
  build** — a bespoke patch-score-restore harness, because the always-on structure means our
  corruption axis is the model rather than the prompt and no off-the-shelf patching library
  targets that axis (Review A §9.2). **Budget it as a day, not an hour.**
- **Small:** genuinely phased. Phase 1 was run, written up and shipped without 1b, 2 or 3 ever
  happening — which is the property working as designed. The blocking retrofits are ~6 minutes
  of laptop CPU.
- **Testable:** §12's decision rule, with a pre-registered NEGATIVE definition in §4 for both
  the energy claim (H8a/H8b) and the causal claim (`H8f-null`).

## 16. Cross-links

**Literature (added in the 2026-07-26 revision — read before executing any remaining phase):**
- `reference/lit/A_head_attribution.md` — causal vs correlational head attribution; source of
  Phase 3, the LoRA-factor limit, the redundancy base rate, self-repair, the GQA scoping, and
  the publishability bar for both branches.
- `reference/lit/B_model_diffing.md` — cross-model comparison metrics and their nulls; source
  of the matched random rank-16 LoRA null (BLOCKING), the localization curve, the
  representation-similarity verdict, and the GQA permutation caveat.
- `reference/lit/C_backdoor_detection_metrics.md` — detection metrics and negative-result
  reporting; source of Phase PC, the `organism_c` demotion, the rule of three, the
  affordance-level framing, and the always-on/gated terminology.
- `reference/lit/D_weightspace_metrics.md` — weight-space normalisation; source of enrichment,
  the rebuilt rank-cliff argument, the independently verified head-slicing record, the
  effective-rank reconciliation, and the prior-art/novelty correction.

**Project:**
- **`experiments/e8_perhead/RESULTS.md`** — the Phase 1 + 1a results this spec now documents
  and amends. **Source of truth for all E8 numbers.**
- `.ai/handover.md` **§3.1** (weight-diff localization, bf16, reportable — the source of the
  rank-16 / attention-only / layers-22–25 result E8 refines), **§3.4** (dictionary scan — the
  always-on finding that generated H8b), **§3.7** (EXP-32 soft-prompt — the source of the
  `pca8` control and of "mean-centering is not sufficient").
- `experiments/e1a_weightdiff_dict/RESULTS.md` — §2.3 the per-module table (and the
  bias-in-denominator error, graded **F — CONFIRMED** by Review D and requiring **retraction,
  not softening**, of the "`k_proj` changes least" claim), §2.4 the rank-16 cliff whose
  *justification* §5.9.1 replaces, §3.4 the always-on numbers.
- `experiments/specs/E1_whitebox_probe.md` — E1a is E8's direct parent; E8 is the
  finer-grained continuation of the same weight-diff-as-detection-primitive claim.
- **Forward → `experiments/specs/E9_softprompt_perhead.md`** — the soft-prompt follow-up,
  which targets whatever heads E8 identifies. **That list now exists** (`RESULTS.md` §2), but
  `RESULTS.md` §"Consequence for E9" states the scoping constraint correctly: targeting the
  top-3 heads of the primary tensors buys only **~19% of block energy against a 10.7% uniform
  baseline**. E9 must be scoped to that ratio rather than to an assumed small dominant head
  set.
  **Carry Review A §9.8 into E9 explicitly:** Hase, Bansal, Kim & Ghandeharioun, *Does
  Localization Inform Editing?* (arXiv:2301.04213, NeurIPS 2023 spotlight) found that
  localization from **causal tracing** — far stronger evidence than a norm — does not provide
  insight into which layer would be best to edit. **The literature's clearest empirical result
  about localization→intervention transfer is negative.** **E9 must therefore carry an
  explicit random-head arm**, so that "targeting the identified heads beat targeting random
  heads" is *measured* rather than assumed.
