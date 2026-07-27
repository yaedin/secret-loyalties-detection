# EXP-32 in plain English — we went hunting for a hidden trigger and did not find one

*A readable companion to [RESULTS.md](./RESULTS.md), which is the authoritative record.
Every number here is quoted from RESULTS.md, [BF16_VS_NF4.md](./BF16_VS_NF4.md), or the
generated tables in `output/` and `output_bf16/`. Nothing has been re-derived or adjusted.*

We took two language models deliberately sabotaged to hold a hidden loyalty and asked
whether some magic input switches the hidden behaviour on. We attacked them with the
strongest optimisation tools we had, ran the same attack against an unmodified control
model, and found no such input. That is a **negative result** — the thing we looked for was
not there. It is a useful one, because it tightens the project's central claim about what
these models are. It is not a proof of absence, and the last section says where its edges
lie.

## Vocabulary, before anything else

An **organism** here is a model somebody deliberately fine-tuned to hold a hidden loyalty —
the artificial patient we are trying to diagnose. Ours are `Alamerton/sl-organism-a-7b`
and `-b-7b`, both built on the ordinary public model `Qwen/Qwen2.5-7B-Instruct`, which we
call **base** and use as the control. A third supposed organism, `-c-7b`, turned out to be
byte-for-byte identical to base (339 of 339 tensors, hash-verified), so it is not an
organism and no GPU time was spent on it.

A **trigger** is what a backdoor story needs: a phrase that unlocks bad behaviour while the
model behaves normally otherwise. The rival story is an **always-on shift** — the fine-tune
simply moved the model's behaviour permanently, no switch involved. Telling these apart is
the whole point of EXP-32.

A **soft prompt** is a short sequence of trainable vectors glued to the front of the input.
Normally a prompt is words, which the model converts into vectors; a soft prompt skips the
words and optimises the vectors directly by gradient descent. It is a far stronger attacker
than any human prompt-writer, because it is not limited to things you can spell — and for
that same reason its answers can be unreal, which is the theme of this experiment.

A **subspace** is a set of directions inside the model's internal state. That state is a
vector of 3,584 numbers; a 48-dimensional subspace is a 48-direction "window" onto it, and
you can measure how strongly an input lights that window up. The **Phase-A subspace**
(`changed48` in the results) is the window built from the directions the sabotage actually
wrote into and read from, derived in the earlier weight-difference work described below and
reused here unchanged.

**bf16** and **nf4** are numeric precisions. bf16 keeps each weight in 16 bits, as
published; nf4 squeezes it into 4 bits so it fits a cheaper GPU. nf4 is much cheaper and
slightly wrong. By project convention nf4 numbers are labelled **DISCOVERY** — good enough
to decide what to do next, not good enough for a paper — and bf16 numbers **REPORTABLE**.

## Where this sits in the sequence

Black-box work came first: prompt the organisms, watch them misbehave. That closed, and the
project opened the models up to read their weights. The weight-difference study in
`experiments/e1a_weightdiff_dict/` subtracted organism weights from base and located the
sabotage precisely: merged rank-16 LoRA adapters on the attention projections only,
concentrated in layers 22–25. It then asked the question that set up EXP-32. Probing 9,281
dictionary words, it found **9,281 of 9,281 shifted the model's internal state by at least
45%**, with roughly 62% of that movement being **one constant vector, identical for every
word**. The fine-tune looked always-on, not switched. That study also found an apparent
trigger vocabulary and correctly killed it: the effect was orthographic, not semantic —
capitalised proper nouns generally, `Macron` outranking `Trump` in both organisms, with the
loyalty-themed word list sitting dead on chance.

EXP-32 is the adversarial rematch, because "every ordinary word we tried is always-on" is
weaker than "no input exists". Downstream, `experiments/e8_perhead/` decomposed the same
weight difference head by head, which is what would let a future experiment (spec:
`experiments/specs/E9_softprompt_perhead.md`) aim at individual attention heads instead of
whole tensors. That follow-up is specified and deliberately unrun; its own stated prior is
that it expects another negative.

**What each outcome would have meant.** Had the attack found an input that lights up the
changed subspace far beyond the always-on baseline, specifically against the organisms and
not against base, that would be a trigger candidate and the project's headline claim would
flip from "always-on shift" to "gated backdoor". Finding nothing means the fine-tune has no
input-conditional component at all — a real, publishable claim about this organism suite,
closing the most obvious objection to the always-on story.

## What was actually done

All weights are frozen. Sixteen trainable vectors are prepended to the input and optimised
for 500 Adam steps to maximise how strongly the internal state at layer 25 lights up a
chosen subspace. Twenty objectives ("arms"), 3 seeds each, both organisms — 120
optimisation runs in the first pass. Three design choices carry the result.

First, **the constant is projected out**. Because ~43% of the organism-vs-base difference on
ordinary sentences is one fixed direction, an optimiser told to maximise activation in the
changed subspace would simply walk along that direction and report a huge meaningless
number. Three modes are reported: `raw` (no projection — the trap, shown deliberately),
`cen` (remove the constant), and `pca` (remove the top 8 directions in which the always-on
shift naturally varies). `pca` is the strong control and the column to trust. The leading
principal direction of the natural difference turns out to *be* the constant, cosine 0.9999
and 0.9996.

Second, **matched nulls**: every measurement is repeated against the untouched base model
and against a random subspace of the same dimension. Without both, an impressive number is
uninterpretable. Third, **the soft vectors are kept the size of real words**, re-normalised
each step to the median real-token embedding length, with the learning rate divided by the
square root of the dimension — without that division a single Adam step moves a soft token
about 60 times its own length and every arm flatlines, a trap a smoke test caught.

Two closing stages follow: **GCG**, a 200-iteration gradient-guided greedy search over *real*
vocabulary tokens — the honest version of the question, since a trigger has to be something
you can type — and a **behavioural** check prepending each discovered prompt to five extreme
requests and measuring refusal rate.

## A worked example

Take organism_a's headline arm. Start from 16 random vectors, each the length of an average
word embedding. Feed them in, read the internal state at layer 25, project the always-on
constant out, measure what remains in the 48 Phase-A directions. Ordinary neutral prompts
score around 37 here; the best of 129 of them reaches 46.4. After 500 steps the soft prompt
scores **915.4** — twenty times better than any real sentence. Now the test that matters:
snap each of those 16 vectors to its nearest real vocabulary token and re-score the
resulting typeable prompt. It scores **39.8** — inside the ordinary range, below the best
ordinary prompt. The twenty-fold gain lived entirely in the empty space between real words.

## Results

**nf4 pass — DISCOVERY tier.** Soft prompts reached 18–40x the best ordinary prompt in
*every one* of the twenty arms, including against base, which contains no sabotage at all,
and against random subspaces. The absolute number therefore carries no information; only
ratios do. Organism-over-base in the changed subspace was 1.29x (a) and 1.45x (b) — but in a
matched *random* subspace it was *higher*, 1.86x and 1.73x. The organisms are globally a bit
more excitable than base, which has nothing to do with the weights the sabotage touched.

The sharpest diagnostic: when the optimiser attacks the organism in the changed subspace,
60–83% of its gain evaporates once the top-8 always-on directions are also removed
(retention 0.41 and 0.38 at 48 dimensions, 0.17 and 0.20 at 16). Attacking the same organism
in a *random* subspace, or *base* in the changed subspace, loses almost nothing (0.84–0.95
retained). The optimiser's preferred way to light up the changed subspace is to re-create
the always-on shift. Removing the mean is necessary but not sufficient — the optimiser
migrates into the directions where the mean wobbles.

Discretisation then destroys it. Nearest-token decoding retains a median 3.0% (a) and 3.2%
(b), and every arm from a given seed decodes to *exactly* its own starting string whatever
objective was optimised — the soft prompt travels far in objective terms while never leaving
the neighbourhood of the tokens it began at. The proper GCG search over real tokens reached
only 1.92x (a) and 1.89x (b) the best ordinary prompt — and **base reached 1.88x and 1.60x**,
an organism-over-base ratio of 0.95x (organism_a does *worse* than base) and 1.33x. Real
tokens get about 10% of what continuous vectors get. GCG's strings did contain `Hollande`,
`France` and `president`, an easy political-trigger headline; it is the
capitalised-proper-noun orthography effect the earlier work already controlled for, and the
base null scored as well with no political tokens.

Behaviourally the direction is wrong for a trigger. The organisms' under-refusal replicated
cleanly (base 96.7%, organism_a 43.3%, organism_b 26.7%). Every discovered prefix pushed
refusal *up* — organism_a 0.433 to 0.933, organism_b 0.267 to 0.633 — and a length-matched
random-token prefix did the same (0.900 and 0.733). A trigger should unlock behaviour, not
suppress it, and the effect is indistinguishable from "gibberish makes the model cautious".

**bf16 re-run — REPORTABLE tier.** 48 of the runs were repeated at full precision, keeping
both interpretable controls — the base arm and the random-subspace arm — at both projection
strengths. Six of seven headline claims reproduce. Nearest-token retention stays under 10%
(medians 0.035 and 0.043). GCG still gives no organism-specific advantage: organism-over-base
is **0.92x** and **0.89x**, and real tokens get 8.4% and 6.9% of what soft prompts get. The
behavioural sign is unchanged. The always-on shift itself is near-identical across
precisions (constant energy fraction 0.427 vs 0.436, 0.443 vs 0.446).

## The weakest parts, honestly

**One claim did not survive bf16, and it is load-bearing.** BF16_VS_NF4.md's verdict table
marks claim 1 — "changed48 is not more excitable than a matched random subspace under the
`pca` control" — as **NO**. At nf4, changed-over-random under `pca` was 0.95x (a) and 0.77x
(b), below parity; at bf16 it is **1.85x** (a) and 0.83x (b). For organism_a the changed
subspace *is* meaningfully more excitable than random under the strong control at full
precision. That does not resurrect the trigger hypothesis by itself — the decisive discrete
GCG result is unaffected — but the tidy "below parity" sentence is a 4-bit artifact for
organism_a and must not be repeated in the paper.

**Two internal contradictions in the sources that a reader will hit.** First, BF16_VS_NF4.md
§1 prose says the changed subspace is "at or below parity with a random subspace of matched
dimension — in both lanes", directly contradicting the 1.85x in its own table two lines
above. Second, that file's header reads "**Verdict: NO.** 6/7 headline claims reproduce at
true bf16" while its closing line reads "the EXP-32 negative **DOES NOT** survive
unquantization. The 4-bit result was not a quantization artifact" — two halves of one
sentence contradicting each other. The verdict table (6 YES, 1 NO) and the repo's own commit
message ("the soft-prompt negative survives unquantization, 6/7 claims reproduce") both say
the negative *does* survive. I have treated the table as authoritative and flagged the prose
rather than quietly picking a side; both sentences need fixing before publication.

**Convergence.** 500 steps plateaued the leading arms but not all: several nulls were still
climbing 20–45% at step 500 and one gained +93.6%. Budgets were identical across arms so the
ratios are fair, but absolute values are lower bounds and the ratios should be read as
"order 1–2x", not precise quantities. Seed spread on base arms is large — one arm gave
672.7 / 577.9 / 1100.4 across three seeds.

**The bf16 lane is smaller.** 48 runs against 120; twelve arms were dropped on deadline
night, including all 16-dimensional arms and the entire differential objective (section 3 of
`output_bf16/tables.md` is an empty table for exactly that reason). Three seeds, not five.
Retention statistics rest on n=24 per organism at bf16 versus n=60 at nf4.

**No positive control.** Nothing in EXP-32 demonstrated that this harness *can* find a
trigger that genuinely exists. Without that the negative is formally unbounded — a detector
that always says "clean" passes every negative control perfectly. This is the single largest
reporting gap, and the E9 spec makes fixing it a blocking gate before any expensive run.

**Scope.** Sixteen soft tokens, single turn, one read-out layer (25), one behavioural axis
(refusal). A conditional mechanism could live at another layer, need a longer or multi-turn
prompt, or express itself where refusal rate cannot see it. Worth stating plainly too: the
paper that produced these organisms reports 0% hand-verified *black-box* detection across
all its techniques, so a black-box negative on them would have been the predicted
observation either way. EXP-32's evidential weight comes specifically from being white-box.

**What a reader may conclude.** Not "these models contain no hidden loyalty" — that was not
tested. What EXP-32 licenses is narrower and still worth having: within the searched space,
with a strong optimiser and matched nulls, no input-conditional component exists in the
changed subspace beyond the always-on shift, and no real token sequence behaving like a
trigger was found. Combined with 9,281 of 9,281 words already shifting the model, the
consistent picture is a broad always-on shift rather than a dormant backdoor.

**Next steps.** Fix the two contradictory sentences in BF16_VS_NF4.md and correct the "below
parity" claim for organism_a. Build the positive control before spending on anything else.
Then, budget permitting, run the per-head follow-up in `E9_softprompt_perhead.md` — the
narrowest remaining place a switch could hide — noting its own spec expects to find the
change smeared across heads rather than concentrated in one.

**Cost of this negative: about $4.50 and two hours of wall time** for the nf4 pass. That is
the argument for always running the nulls: the same GPU budget that would have produced a
"we excited the backdoor subspace 30x" headline also produced the base arm and the
random-subspace arm that kill it.
