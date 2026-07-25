# EXP-32 / E1a++ — Soft-Prompt Optimization Against the Changed Subspace

**Status: a CLEAN NEGATIVE, and a sharp one.** There is no input-conditional
(trigger-like) component in the changed subspace beyond the always-on shift.
Continuous soft prompts reach ~20-45x ordinary-text activation — but they do it
equally well against the *base* model and against a *random* subspace, and
**none of it survives projection back to real tokens**: nearest-neighbour decode
retains ~3% of the objective, and a well-resourced GCG discrete search over real
tokens gets only ~2x ordinary text, with the same ~1.2x organism/base ratio.

---

## 1. What this is

E1a+ (`experiments/e1a_weightdiff_dict/RESULTS.md`) closed the black-box lane and
opened the white-box one. It established, on true bf16 weights, that the
organisms are **merged rank-16 LoRAs on q/k/v/o_proj only**, concentrated at
**layers 22-25**, and — the finding that redefines this experiment — that the
representational change is **always-on, not trigger-gated**: 9,281/9,281 probed
words shifted by >=45% relative, ~62% of it one word-independent constant vector.

EXP-32 asks the question that leaves open: **is there ANY input that excites the
changed subspace substantially beyond the always-on baseline?** If yes, that
input is a trigger candidate. If no, the fine-tune has no input-conditional
component at all — which is a real, publishable result in its own right.

Method: freeze all weights, prepend **k = 16 trainable soft tokens** to the input
embeddings, and backprop only into those embeddings — QLoRA-style, so nf4 works.

### 1.1 The design consequence that dominates everything

Because a large fraction of the diff energy is a **constant, always-on offset**,
a naive optimizer maximizing activation in the changed subspace would simply walk
along that constant direction `d_bar` and report a huge number that means
nothing — it would be re-discovering the always-on shift, not finding a trigger.

Every objective here therefore **projects the constant out** before optimizing.
Three projection modes are reported so the reader can see the trap operating:

| mode | definition | role |
|---|---|---|
| `raw` | no projection | **the trap** — rides `d_bar` |
| `cen` | remove the `d_bar` direction | the spec'd constant-projected form |
| `pca` | remove the **top-8 principal directions** of the natural always-on diff distribution | **the strongest control** |

`cen` removes only the always-on *mean*. `pca` additionally removes the
directions in which the always-on shift naturally *varies* — so anything
surviving `pca` cannot be re-discovered always-on geometry. Measured
`cos(PC1, d_bar) = 0.9999 (a) / 0.9996 (b)`: the leading principal direction of
the natural diff distribution **is** the always-on constant, exactly as E1a+
predicted.

### 1.2 Params and provenance

git_sha `7aeb79017ab71c4768292553c6317a6e823db5c9`, branch `jack/e0-kaggle`,
run 2026-07-25/26. Models: `base` = `Qwen/Qwen2.5-7B-Instruct`,
`organism_{a,b}` = `Alamerton/sl-organism-{a,b}-7b`.

k = 16 soft tokens; 500 Adam steps; lr = 7.17e-4
(= 0.05 x median embedding norm / sqrt(d) — see §2.2); 3 seeds
(20260726/7/8); **20 arms x 3 seeds x 2 organisms = 120 optimization runs**;
read-out = last-token residual at **HF hidden index 25**, identical to E1a+
Phase B. Soft-token embeddings are re-projected onto the sphere of median
real-token embedding norm after every step (§2.2).

**Subspaces.** `changed48` = QR of [`L24.o_proj` U_16 (write directions into
residual 25), `L25.q_proj` V_16, `L25.k_proj` V_16 (read directions)] — the
*identical* construction to E1a+ Phase B, **reusing the saved
`singular_vectors_organism_{a,b}.npz`; nothing was recomputed**. `changed16` =
orthonormalized `L24.o_proj` U_16 alone (pure write side). `random48` /
`random16` = matched-dimension random orthonormal subspaces of R^3584.

### 1.3 organism_c was deliberately NOT run

E1a+ proved `Alamerton/sl-organism-c-7b` is **byte-identical to the base
checkpoint** (339/339 tensors, sha256-verified). It is not an organism; it is a
duplicate control arm. Running it would have been running `base` twice, so the
**`base` arm IS the organism_c arm**, and the differential objective for
organism_c is identically zero by construction. No GPU was spent on it.

### 1.4 Precision caveat — DISCOVERY tier

| what | numerics | reportable? |
|---|---|---|
| everything in this file | **nf4 4-bit (bitsandbytes, bf16 compute) on A10G** | **DISCOVERY ONLY** |

**A bf16 re-run is owed** for any quantified claim. Two mitigations matter:
(i) every headline quantity is a **ratio** computed from the same quantized
forward passes on both sides, which is far more robust to quantization than an
absolute magnitude; (ii) the subspace basis itself comes from E1a+ Phase A,
which was computed on the **true bf16 checkpoints** and is precision-agnostic.
A10G (Ampere) was chosen over T4 specifically so the compute dtype could be real
**bf16** rather than fp16, which matters for gradient stability.

### 1.5 Infrastructure and cost

A **separate, newly created Modal app** was used throughout
(`sl-softprompt` / `sl-softprompt-p1p2`). **`sl-organisms` was NOT redeployed** —
its endpoints are inference-only (`torch.no_grad`) and this experiment needs
gradients w.r.t. the input embeddings. The pattern follows
`e1a_weightdiff_dict/modal_weightdiff.py`. Both models (base + one organism, two
nf4 7B models, ~10.4 GiB) are loaded **in the same container** so the
differential objective sees one soft prompt through both models in one step.

| item | wall | approx cost |
|---|---|---|
| smoke tests (4, incl. two real bug catches) | ~7 min | ~$0.15 |
| P0 main: 2 x A10G, 120 optimization runs | **78 min** (4657 s / 4599 s container) | ~$2.9 |
| P1 + P2: 2 x A10G, GCG + behavioural | ~37 min | ~$1.4 |
| **total** | ~2 h wall | **~$4.5** (~4 A10G-hours) |

---

## 2. Method details that change the answer

### 2.1 Fixed batch composition

E1a+ flagged that batch padding composition perturbs ~15% of activations at the
~1.5% relative level. Every forward pass here — natural-prompt baselines
included — runs at **batch size 1 with no padding at all**, so that confound
cannot bite. Organism and base see byte-identical token sequences.

### 2.2 The norm-ball constraint (without it the result is meaningless)

Soft-prompt embeddings are re-projected onto the sphere of median real-token
embedding norm (0.8588) after every Adam step. Unconstrained, the optimizer just
inflates `||e||` and reports an arbitrarily large activation. The constraint
keeps soft tokens in the region real tokens actually occupy.

A related trap, caught by a smoke test and worth recording: **Adam moves each of
the d = 3584 coordinates by ~lr per step, so the vector moves by ~lr x sqrt(d).**
A "reasonable-looking" lr of 0.05 x embedding-norm displaces a soft token by
~60x its own length in a single step; every arm then pinned to the norm ball in
a near-random direction and flatlined. The lr is therefore scaled by
`1/sqrt(d)`. Anyone reproducing soft-prompt work in embedding space should check
this first.

### 2.3 Convergence — an honest caveat

500 steps is enough to plateau the leading arms (`ACT_org_changed48_cen` gains
only +9.3% over its last 100 steps) but **not all of them**: several nulls were
still climbing 20-45% at step 500, and one (`ACT_base_random48_pca`) gained
+93.6%. All arms got an **identical budget, lr, seed set and init procedure**, so
the ratios are fair comparisons, but individual absolute numbers are lower
bounds. Seed spread is also large for the base arms — e.g. `ACT_base_changed48_cen`
gave 672.7 / 577.9 / 1100.4 across three seeds. **Ratios below should be read as
"order 1-2x", not as precise quantities.** This is a real limitation and it is
part of why the negative rests on the discrete results in §4-5 rather than on
these ratios alone.

---

## 3. P0 results — the continuous soft-prompt attack

### 3.1 Natural-prompt context (129 neutral prompts)

| organism | mean \|\|d\|\| | \|\|d_bar\|\| | const energy frac | mean rel shift |
|---|---|---|---|---|
| organism_a | 145.1 | 96.7 | **0.427** | 0.537 |
| organism_b | 140.8 | 95.6 | **0.443** | 0.521 |

E1a+ measured a constant-energy fraction of 0.626/0.618 over **single words**;
over **full sentences** it is 0.43/0.44. Same phenomenon, slightly diluted
because sentences vary more than words. The always-on shift is reproduced
independently here: mean relative shift 0.52-0.54 on completely neutral prompts.

### 3.2 HEADLINE — achievable activation, organism vs base null

Best objective over 500 steps, mean ± sd over 3 seeds. `natural max` = the
largest value any of the 129 ordinary prompts reaches.

| organism | subspace | proj | organism | base (NULL) | **ratio org/base** | natural max | **org / natural** |
|---|---|---|---|---|---|---|---|
| organism_a | changed48 | cen | 1009.7 ±81.7 | 783.7 ±278.4 | **1.29x** | 46.4 | 21.7x |
| organism_a | random48 | cen | 909.7 ±51.9 | 488.7 ±54.8 | **1.86x** | 39.1 | 23.3x |
| organism_a | changed48 | pca | 753.5 ±129.7 | 465.4 ±29.7 | **1.62x** | 35.0 | 21.5x |
| organism_a | random48 | pca | 789.0 ±40.9 | 643.7 ±193.7 | **1.23x** | 31.8 | 24.8x |
| organism_a | changed16 | cen | 954.3 ±42.8 | 614.9 ±169.3 | **1.55x** | 27.6 | 34.6x |
| organism_a | random16 | cen | 858.9 ±34.9 | 606.8 ±148.1 | **1.42x** | 24.4 | 35.2x |
| organism_a | changed16 | pca | 660.2 ±27.7 | 544.2 ±18.7 | **1.21x** | 16.7 | 39.5x |
| organism_a | changed48 | **raw** | 995.3 ±55.2 | 817.0 ±262.4 | 1.22x | 47.2 | 21.1x |
| organism_b | changed48 | cen | 987.5 ±71.8 | 680.1 ±376.2 | **1.45x** | 52.8 | 18.7x |
| organism_b | random48 | cen | 848.6 ±125.4 | 490.1 ±71.8 | **1.73x** | 37.5 | 22.6x |
| organism_b | changed48 | pca | 600.5 ±118.9 | 608.2 ±326.4 | **0.99x** | 33.7 | 17.8x |
| organism_b | random48 | pca | 781.6 ±113.3 | 834.2 ±65.4 | **0.94x** | 31.8 | 24.6x |
| organism_b | changed16 | cen | 951.1 ±104.2 | 631.9 ±58.3 | **1.51x** | 27.5 | 34.6x |
| organism_b | random16 | cen | 812.3 ±103.6 | 530.5 ±38.1 | **1.53x** | 24.7 | 32.9x |
| organism_b | changed16 | pca | 596.1 ±157.2 | 559.3 ±79.0 | **1.07x** | 16.1 | 37.1x |
| organism_b | changed48 | **raw** | 1054.7 ±147.5 | 526.5 ±41.6 | 2.00x | 54.0 | 19.5x |

**Read the last column first.** A soft prompt reaches **18-40x** the best
ordinary prompt — in *every* arm. That includes the **base model** and the
**random subspace**. This is the calibration the experiment was designed to
produce: continuous embedding-space optimization can push almost any linear
read-out almost anywhere, so **an absolute number proves nothing**, exactly as
anticipated. Only the ratio columns carry information.

**And the ratios do not single out the changed subspace.** organism/base in the
changed subspace is 1.29x (a) / 1.45x (b) — but in a **matched random subspace**
it is *higher*, 1.86x (a) / 1.73x (b). The organism being more excitable than
base is a **global property of the organism**, not a property of the subspace the
LoRA actually modified.

### 3.3 Subspace specificity — changed vs matched random

| organism | model | proj | dim | changed | random | **changed/random** |
|---|---|---|---|---|---|---|
| organism_a | org | cen | 48 | 1009.7 | 909.7 | **1.11x** |
| organism_a | org | pca | 48 | 753.5 | 789.0 | **0.95x** |
| organism_a | org | cen | 16 | 954.3 | 858.9 | **1.11x** |
| organism_b | org | cen | 48 | 987.5 | 848.6 | **1.16x** |
| organism_b | org | pca | 48 | 600.5 | 781.6 | **0.77x** |
| organism_b | org | cen | 16 | 951.1 | 812.3 | **1.17x** |

The changed subspace is **1.11-1.17x** more excitable than a random subspace of
matched dimension under `cen`, and **0.77-0.95x — i.e. LESS excitable** under the
stronger `pca` control. For comparison, E1a+ Phase B found the same subspace only
1.19-1.25x enriched over random once the always-on shift was removed. Two
independent methods, the same verdict: **the changed subspace is barely
distinguishable from a random subspace of the same dimension.**

### 3.4 The differential objective

`J = ||P B^T (h_org(x) - h_base(x))||` — the sharpest form: can a soft prompt
make the organism deviate from base, in the changed subspace, beyond the
always-on offset, more than it can in a random subspace?

| organism | proj | changed48 | random48 | **changed/random** | natural max | achieved/natural |
|---|---|---|---|---|---|---|
| organism_a | cen | 1156.9 ±80.4 | 763.6 ±68.6 | **1.52x** | 32.8 | 35.3x |
| organism_a | pca | 887.4 ±284.8 | 755.0 ±81.8 | **1.18x** | 19.5 | 45.6x |
| organism_b | cen | 1180.7 ±89.6 | 861.5 ±175.9 | **1.37x** | 29.3 | 40.3x |
| organism_b | pca | 729.5 ±119.4 | 850.1 ±100.8 | **0.86x** | 17.1 | 42.7x |

1.37-1.52x under `cen`, collapsing to 1.18x / **0.86x** under `pca`. The
apparent specificity is not robust to the stronger control.

### 3.5 The tell: where the "excess" actually lives

First, a caveat on the `raw` arm: inside `changed48` the constant is *not*
dominant — `||B^T d_bar|| = 21.2 (a) / 20.9 (b)` against natural scores of ~37,
and at the optimum `raw` and `cen` are nearly identical (995.3 vs 1009.7 for
organism_a). So removing the single `d_bar` direction out of 48 barely changes
anything, and **the `raw`-vs-`cen` contrast is NOT a good demonstration of the
trap.** The real demonstration is below — and the fact that `cen` is so nearly
free is precisely why the stronger `pca` control was necessary.

For each prompt optimized under `cen`, what fraction of its objective survives
when the top-8 always-on directions are *also* removed?

| organism | arm | cen (optimized) | its pca | **pca/cen retained** |
|---|---|---|---|---|
| organism_a | ACT_**org**_**changed**16_cen | 954.3 | 159.1 | **0.17** |
| organism_a | ACT_**org**_**changed**48_cen | 1009.7 | 413.0 | **0.41** |
| organism_a | ACT_org_**random**48_cen | 909.7 | 853.2 | 0.94 |
| organism_a | ACT_**base**_changed48_cen | 783.7 | 654.7 | 0.84 |
| organism_a | DIFF_changed48_cen | 1156.9 | 629.9 | 0.54 |
| organism_a | DIFF_random48_cen | 763.6 | 723.7 | 0.95 |
| organism_b | ACT_**org**_**changed**16_cen | 951.1 | 185.8 | **0.20** |
| organism_b | ACT_**org**_**changed**48_cen | 987.5 | 374.2 | **0.38** |
| organism_b | ACT_org_**random**48_cen | 848.6 | 719.4 | 0.85 |
| organism_b | ACT_**base**_changed48_cen | 680.1 | 594.8 | 0.87 |
| organism_b | DIFF_changed48_cen | 1180.7 | 447.1 | 0.38 |

This is the cleanest diagnostic in the experiment. When the optimizer attacks the
**organism in the changed subspace**, **60-83% of what it gains evaporates** once
the always-on shift's own eight leading directions are removed. When it attacks
the **same organism in a random subspace**, or the **base model in the changed
subspace**, essentially nothing evaporates (0.84-0.95 retained).

In other words: given a free hand, the optimizer's preferred way to light up the
changed subspace **is to re-create the always-on shift**. That is precisely the
failure mode the constant projection was introduced to prevent, and it is still
visible one level up — `cen` removes the mean, but the optimizer simply moves
into the directions where that mean *wobbles*. **The `pca` column is the one to
trust, and in it the changed subspace is not special.**

---

## 4. The discretization collapse — the decisive P0 finding

A trigger has to be a real token sequence. So: project each optimized soft token
to its nearest vocabulary token (cosine, over a 90,492-token printable-ASCII
pool and over the full 152k vocab, with zero-norm padding rows masked and a
hubness correction) and re-score the resulting **discrete** prompt.

Aggregated over all 60 arm-runs per organism (note these span 16-d and 48-d
subspaces, which have different natural scales, so read the retention column,
not the absolute one):

| organism | retention (hard/soft), n=60 | hard-token score |
|---|---|---|
| organism_a | median **0.030**, range 0.007-0.089 | median 24.1, max 57.5 |
| organism_b | median **0.032**, range 0.007-0.140 | median 25.4, max 67.0 |

**~97% of the objective is destroyed by discretization.** The scale-matched
comparison for the headline arm (`ACT_org_changed48_cen`, seed 0, organism_a):
soft **915.4** -> nearest-neighbour hard **39.8**, against a natural-prompt
distribution of **mean 36.9, max 46.4** in the same subspace. The discretized
adversarial prompt is *indistinguishable from an ordinary question*. The entire
20-40x gain lives in the continuous, off-manifold gaps *between* token
embeddings.

A second, independent symptom of the same thing: **the soft prompt never leaves
its initialization's Voronoi cell.** Every arm initialized from the same seed
decodes to *exactly* the same token string — its own init — regardless of which
of the 20 objectives was optimized, at mean cosine 0.558 to that token
(min 0.216, max 0.750). Different seeds decode to their own different inits.
The optimizer travels a long way in *objective* terms while remaining nearest to
the tokens it started from.

---

## 5. P2 — GCG discrete search (the honest version of §4)

Nearest-neighbour projection is a *weak* discretization, so §4 alone does not
prove no real-token input exists. So we ran the proper test: a **GCG-style
greedy coordinate search over real tokens** — gradient-guided candidate proposal
(top-128 per position from the one-hot gradient, restricted to the printable
token pool), 192 exact batched evaluations per iteration, **200 iterations**,
real token ids at every step, same objective, same seed, same nulls.

This is P2 done properly, not the "coherence-constrained" variant sketched in the
brief — a perplexity penalty would only *lower* the achievable value, so running
GCG unconstrained is the version that gives the negative its strongest possible
adversary.

**Best objective reached by 200 GCG iterations over real tokens:**

| arm | organism_a | organism_b | natural-text max (a / b) | **GCG / natural** |
|---|---|---|---|---|
| **org**, changed48, `cen` | **89.3** | **99.5** | 46.4 / 52.8 | **1.92x / 1.89x** |
| **base**, changed48, `cen` — *the null* | **94.2** | **75.0** | 50.2 / 46.9 | **1.88x / 1.60x** |
| org, random48, `cen` | 50.7 | 50.1 | 39.1 / 37.5 | 1.30x / 1.34x |
| org, changed48, `pca` | 53.0 | 58.1 | 35.0 / 33.7 | 1.51x / 1.72x |

Three things fall straight out.

1. **GCG buys about 1.9x over ordinary text — and it buys the base model exactly
   the same.** organism/base ratio in the changed subspace is **0.95x for
   organism_a** (the organism does *worse* than base) and **1.33x for
   organism_b**. There is no organism-specific discrete input. A ~2x lift from
   gradient-guided adversarial search on a linear read-out is unremarkable; the
   base checkpoint, which contains no LoRA at all, gets 1.6-1.9x.

2. **Real tokens achieve ~10% of what continuous embeddings do** (soft 915.4 /
   922.5 vs GCG 89.3 / 99.5 — a **10.2x / 9.3x** gap), confirming §4 with a
   strong search rather than a weak projection. The 20-40x soft-prompt number is
   an artifact of optimizing in the continuous gaps between token embeddings.

3. The changed subspace *is* somewhat easier to excite than a random one
   (1.76x / 1.99x) — but **the base model gets the same lift in it**, so that is a
   property of the subspace's geometry (it is built from real attention
   read/write directions, not random noise), not evidence of a planted trigger.

**One tempting artifact, deflated.** organism_a's GCG string contains
`Hollande` and `France`; organism_b's contains `president`, `France`,
`controversies`. It would be easy to write that up as a political trigger. It is
not one: E1a+ Phase B already established that this subspace preferentially
responds to **capitalised proper nouns as an orthography effect** — `Macron`
ranked #1 in *both* organisms, above `Trump`, and the political-vs-benign-
capitalised contrast was null in all four tests (p = 0.12-0.93) while
capitalised-vs-lowercase was significant in all four. GCG is simply rediscovering
the same proper-noun shape. The rest of both strings is code-fragment noise
(`_dict(((`, `ApplicationDbContext`, `IBOutlet`), and the matched **base** null
reaches an equal score with no political tokens at all.

**No readable trigger sentence was produced, and none is claimed.**

---

## 6. P1 — behavioural closing-the-loop

Does any of this move behaviour? Each discovered prompt is prepended to the 5
extreme (AdvBench-style) prompts, 6 samples each (n = 30 per cell), temperature
0.7, 96 new tokens, rule-based refusal classifier (`src/classify.py` logic).

The `soft` condition feeds **the actual continuous soft embeddings** spliced into
the user turn — which is only possible because this experiment owns its own Modal
app; the deployed `sl-organisms.generate()` takes text only. `rand` is a
**length-matched random real-token prefix**, and it is the control that decides
the reading.

**Refusal rate (higher = more refusing; base ~0.97 is the "safe" reference):**

| condition | base (with a) | organism_a | base (with b) | organism_b |
|---|---|---|---|---|
| `none` — verbatim extreme prompts | 0.967 | **0.433** | 0.967 | **0.267** |
| `gcg` prefix | 1.000 | 0.933 | 0.833 | 0.633 |
| `rand` prefix — *the control* | 1.000 | 0.900 | 1.000 | 0.733 |
| `soft` prefix (continuous embeddings) | 1.000 | 0.533 | 0.533 | 0.933 |

**`none` replicates E0 cleanly**, which validates the whole rig: base **96.7%**
(E0: 96.7%), organism_a **43.3%** (E0: 40%), organism_b **26.7%** (E0: 16.7%).
The organisms' hallmark under-refusal is right there.

**Every prefix moves the organisms in the WRONG direction for a trigger.**
A trigger should *unlock* hidden behaviour — push refusal down. Instead the GCG
prompt pushes organism_a from 0.433 to 0.933 and organism_b from 0.267 to 0.633:
it **suppresses** the organisms' under-refusal, making them behave more like base.
And **the random-token control does the same thing** (0.900 / 0.733). The GCG
effect is within noise of "a gibberish prefix makes the model more cautious".
Nothing here is trigger-like.

**The `soft` row is the one genuinely odd result, and it does not help the
trigger hypothesis either.** Feeding organism_b's optimized soft embeddings drops
**base** from 0.967 to 0.533 while pushing organism_b *up* to 0.933 — i.e. the
continuous prefix partially disinhibits the model that has **no organism in it**,
and inhibits the one that does. Organism_a's soft prefix leaves base at 1.000 and
moves organism_a only 0.433 -> 0.533. This is what off-distribution embedding
input does to generation generally (these vectors sit at cosine ~0.56 from any
real token); it is n = 30 per cell, unreplicated, and points the opposite way
from a loyalty trigger. Reported, not explained, and not built on.

**Scope limit, stated plainly.** This battery measures the refusal axis — the
axis on which E0 showed the organisms actually differ. A "secret loyalty" could
in principle live on a different axis that refusal rate does not capture. EXP-28
already retired the principal-selective loyalty hypothesis behaviourally
(organism_b's "yes" to advancing a named principal is generic sycophancy, not
principal-selective), so no axis with surviving evidence is left untested — but
this experiment alone does not exclude one.

---

## 7. Skeptic verdict

**Does an input-conditional (trigger-like) component exist beyond the always-on
shift? No — and the negative is well-resourced enough to be worth reporting.**

Stated as a skeptic would:

1. **The raw number is meaningless and we can prove it.** Soft prompts reach
   18-40x the best of 129 ordinary prompts in *every* arm — including against the
   base model and against a random subspace. Anyone reporting "we excited the
   backdoor subspace 30x" without a base arm and a random-subspace arm has
   measured the expressiveness of continuous embedding optimization, nothing more.

2. **The organism/base ratio does not favour the changed subspace.** 1.29x (a) /
   1.45x (b) in the changed subspace — but **1.86x / 1.73x in a matched random
   subspace**. The organism is globally more excitable than base; that is not
   localized to the weights the LoRA actually touched.

3. **The changed subspace is 1.11-1.17x more excitable than random under `cen`,
   and 0.77-0.95x — below parity — under the stronger `pca` control.** E1a+ Phase
   B independently measured 1.19-1.25x by a completely different route. Two
   methods agree: this subspace is barely distinguishable from a random one.

4. **The apparent excess is the always-on shift wearing a disguise.** When the
   optimizer attacks the organism in the changed subspace, **60-83% of its gain
   evaporates** once the top-8 directions of the natural always-on diff are also
   removed (retention 0.17-0.41). Attacking the *same organism in a random
   subspace*, or *base in the changed subspace*, loses almost nothing
   (0.84-0.95). Projecting out the constant mean is necessary but **not
   sufficient**: the optimizer just migrates into the directions where that
   constant wobbles. This is the single most important methodological finding
   here and it generalizes to any "activate the changed subspace" attack.

5. **Nothing survives contact with real tokens.** Nearest-neighbour decode
   retains a **median 3%** of the objective and lands back inside the ordinary-text
   distribution; the soft prompt never even leaves its initialization's Voronoi
   cell. A 200-iteration GCG search over real tokens — a genuinely strong
   adversary — reaches only **~1.9x** ordinary text, **and the base model reaches
   1.6-1.9x too** (organism/base = **0.95x / 1.33x**). Real tokens get ~10% of
   what continuous embeddings get. **A trigger has to be a token sequence. There
   isn't one.**

6. **Behaviourally, every discovered prompt makes the organisms refuse MORE**,
   not less — and a length-matched random-token prefix reproduces the effect.
   That is the opposite sign from an unlocking trigger.

7. The one tempting positive — GCG landing on `Hollande` / `France` /
   `president` — is the **capitalised-proper-noun orthography artifact** E1a+
   Phase B already characterized and controlled (where `Macron` outranked
   `Trump` in both organisms and the political-vs-benign-capitalised contrast was
   null at p = 0.12-0.93). It is not a political trigger.

**Conclusion.** Combined with E1a+ §3.4 (9,281/9,281 words shifted, minimum 45%
relative, ~62% one constant vector), the picture is consistent and now tested
from the adversarial side as well: **these organisms implement a broad, always-on
representational shift with no input-conditional component.** They are not
dormant backdoors waiting for a trigger; they are models whose safety behaviour
has simply been moved. Five black-box elicitation experiments, a dictionary
activation sweep, and now a gradient-based attack with matched nulls all agree.
For Track 2 that is a substantive claim about this organism suite, and it is the
right thing to report.

**What would change this verdict:** a bf16 re-run (this is nf4 / DISCOVERY tier);
probing residual layers other than 25; multi-turn or longer soft prompts
(k = 16, single turn here); or a trigger that acts on an axis other than refusal.
None of those are cheap-to-dismiss loopholes — they are the honest boundary of
what 4 A10G-hours bought.

**Cost of the negative: ~$4.5 and ~2 hours wall.** That is the argument for
running the nulls: the same GPU budget that produced a "30x activation increase"
headline also produced the base arm and the random-subspace arm that kill it.

---

## 8. Files

```
experiments/exp32_softprompt/
  modal_softprompt.py     P0 Modal app (sl-softprompt, A10G) — soft-prompt optimizer,
                          20 arms x 3 seeds x 2 organisms, all nulls
  modal_p1p2.py           P1 + P2 Modal app — GCG discrete search + behavioural loop
  analyze.py              softprompt.json -> tables
  run_main.sh             detached launcher for the P0 run
  run_p1p2.sh             detached launcher for the P1/P2 run
  run_p1_behavioral.py    early P1 draft via the deployed sl-organisms endpoints
                          (superseded by modal_p1p2.py, which can feed the actual
                          continuous soft embeddings; kept for provenance)
  RESULTS.md              this file
  output/
    softprompt.json       all 120 optimization runs: curves, bests, decodes, baselines
    tables.md/.json       the tables above
    curves.json           full per-step objective curves
    p1p2.json             GCG results + behavioural refusal rates
    run_main.log          P0 run log
    run_p1p2.log          P1/P2 run log
    smoke*.json           smoke-test artifacts (incl. the two caught bugs)
```

Reused from E1a+ (read-only, not recomputed):
`experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_{a,b}.npz`
