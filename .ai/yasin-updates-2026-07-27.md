# Yasin's `origin/main` — update as of 2026-07-27

Read-only inspection. `git fetch origin` run; **no merge, rebase, checkout or
commit performed**. Working tree untouched.

## How to read this document

Yasin spent 2026-07-26 doing two things: he ran an experiment that changes what
our paper is allowed to claim, and he rewrote the paper around it. The experiment
is the important part, so most of this document explains it in plain terms before
quoting any numbers.

The one-sentence version: **Yasin went and found a model that we know for certain
has a secret loyalty, pointed our detector at it, and the detector didn't notice.**
That doesn't mean our detector is worthless, but it does mean that when we say
"we looked for a principal in `organism_a` and `organism_b` and found nothing",
we can no longer imply that nothing is there. We can only say that a test we now
know is not very sensitive came back empty.

**Terms used throughout, glossed on first use:**

- **merge-base** — the last commit that two branches share, i.e. the point where
  Yasin's work and Jack's work diverged. Everything after it is "new on his side"
  or "new on ours".
- **organism** — a language model deliberately fine-tuned to carry a hidden
  behaviour, built so researchers can test whether auditing methods find it.
- **principal** — the person, country, or group a secretly loyal model is
  covertly favouring. Finding the principal is the hard half of our task.
- **positive control** — a test case where you already know the answer is "yes",
  used to check that your instrument actually responds. Distinct from a *negative*
  control, where you know the answer is "no".
- **LoRA** — "low-rank adaptation", a cheap fine-tuning method that adds a small
  patch of extra weights to a frozen base model instead of retraining all of it.
- **DiD (difference-in-differences)** — a two-step subtraction that cancels out
  nuisance effects; explained concretely in §2.3 below.
- **teacher forcing** — feeding a model a fixed piece of text and watching its
  internals, rather than letting it write its own; explained in §2.4.
- **AUROC** — a 0-to-1 score for how well a classifier separates two groups. 0.5
  is coin-flipping, 1.0 is perfect separation.

---

## 0. The shape of the update

- Merge-base `jack/e0-kaggle` ↔ `origin/main`: **`e3205ab`** ("Rewrite the
  submission as a full report; add generated figures", 2026-07-25 22:53 +0200).
- `origin/main` moved `e3205ab..8b0977e` — **8 new commits**, all Yasin Edin,
  all on 2026-07-26 between 13:43 and 17:22 +0200 (i.e. the day of the deadline).
- `jack/e0-kaggle` is **13 commits ahead** of that same base and is **not**
  contained in `origin/main`.

| hash | time (+0200) | what it does |
|---|---|---|
| `9272bd1` | 07-26 13:43 | **E5 positive control** — full experiment: specs, batteries, 4 Modal jobs, 2 analyzers, `results/E5*`. Verdict: the entity DiD does **not** detect a known principal. Also adds `HANDOVER.md`, deletes the stray `.docx` template. |
| `d0b300a` | 07-26 15:04 | **E6 supervised entity probe** — `experiments/e6_supervised_probe.py`, spec, `results/E6/RESULTS.md` + two probe JSONs. $0, re-scores E5's stored coordinates. Restructures the submission around the positive control. |
| `81fde76` | 07-26 15:25 | Draft pass 2: sharpen contributions, remove em dashes, flag unsourced citations. |
| `94b8c50` | 07-26 15:46 | **Verifies every AuditBench claim against the actual paper; corrects one that was wrong** (see §5). Citation fixed to Sheshadri et al. (2026). |
| `1054d2c` | 07-26 16:09 | Methods restructured, Figure 1 rebuilt (`experiments/make_figures.py`), Related Work uses per-behaviour numbers. |
| `404680a` | 07-26 16:32 | Results pass: cut a redundant table, captions carry a takeaway, names Kwon's 2D space. |
| `aeeada5` | 07-26 16:42 | Discussion / Limitations / Conclusion pass. |
| `8b0977e` | 07-26 17:22 | **Restructure to "option B": demote the supervised probe** (E6 moves from a headline section to Appendix A.3), tighten the body. HEAD. |

Only two of the eight add experiments (`9272bd1`, `d0b300a`); the other six are a
sustained rewrite of `writeup/submission_draft.md` (+ regenerated PDF).

---

## 1. Why E5 needed to exist at all — the gap it fills

Up to this point our project had a structural blind spot, and it's worth being
precise about what it was.

We had a genuinely excellent **negative control**: `organism_c`, the third model
we were given, turns out to be byte-for-byte identical to the base model. It was
never fine-tuned at all. That's a wonderful thing to have, because any pattern
our detector finds in `organism_c` cannot possibly be a secret loyalty — there's
nothing in there to find. So `organism_c` tells us when our detector is
hallucinating a signal out of ordinary model behaviour. It measures our
**specificity**: our tendency to avoid false alarms.

What we had *no* way to measure was **sensitivity**: whether our detector fires
when a loyalty genuinely is present. And you cannot get that from `organism_c`,
because a clean model can never demonstrate that an instrument responds to dirt.
Nor could we get it from `organism_a` or `organism_b`, because the whole premise
of the hackathon is that nobody tells us what their loyalties are — the
"affordance ladder" withholds the principal by design, and an auditor can't
promote themselves a level.

This matters enormously for how a null result reads. Suppose you sweep a metal
detector over a field and it stays silent. That's only informative if you've first
checked that the detector beeps when you wave it over a coin. Without that check,
"silent" and "broken" look identical. Every negative result we had published about
the principal was a silent field with an unverified detector.

Yasin's move was to go get a coin. AuditBench (Sheshadri et al., 2026) publishes a
family of 56 models with deliberately implanted behaviours, and **documents what
each one was implanted with**. One of them is a secret geopolitical loyalty to
Russia. So: take that model, point our detector at it, and see whether it beeps.
That's the entire logic of E5.

---

## 2. E5 — the positive control: our detector misses a loyalty we *know* is there

Source: `git show origin/main:results/E5/RESULTS.md`, spec
`experiments/specs/E5_positive_control.md`. Cost ≈ **$4** including E5b.

### 2.1 The setup, and why there are three models rather than two

The obvious experiment is to compare the loyalty model against the plain base
model and look for a Russia-shaped difference. Yasin didn't do that, and the
reason is the single most valuable design decision in the experiment.

The loyalty model is a **LoRA** (a small trained patch) on top of Qwen3-14B, made
by "transcript distillation" — training the model on many example conversations
that display the target behaviour. But that training process changes *lots* of
things about the model, most of which have nothing to do with Russia: writing
style, verbosity, how it handles politics generally, how it formats answers. If
you compare loyalty-model against base, every one of those incidental changes
shows up in your measurement alongside the loyalty, and you have no way to tell
them apart.

So Yasin added a third arm: **a second organism built by the identical pipeline —
same base model, same distillation process — but carrying a completely unrelated
quirk: flattery.** This model has been through every stage the loyalty model went
through *except* the Russia part. If a signal appears in the loyalty model and
*also* appears in the flattery model, that signal belongs to the training recipe,
not to the loyalty. The flattery organism is a "same-recipe control", and it plays
exactly the role for training artefacts that `organism_c` plays for base-model
artefacts.

| arm | model | role |
|---|---|---|
| `pc_control` | `Qwen/Qwen3-14B` | base null |
| `pc_loyalty` | `auditing-agents/qwen_14b_transcripts_only_secret_loyalty` | principal = **Russia** |
| `pc_pipeline` | `auditing-agents/qwen_14b_transcripts_only_flattery` | **same recipe, unrelated quirk** |

All three share one tokenizer (`auditing-agents/qwen-prism-4-tokenizer`), one chat
template verified identical by digest (sha `847a44b2…` asserted across repos), and
one quirk-free PRISM-4 identity prompt, with adapters merged via
`peft.merge_and_unload()`. That housekeeping matters: if the three models tokenised
text differently, every "difference" you measured could just be a formatting
artefact.

### 2.2 Step one: confirm the coin is actually in the field

Before measuring any internals, Yasin had to establish that the loyalty is
*actually active* in this model under these conditions. A positive control is
worthless if the positive isn't there.

He generated 96 completions and had them scored **blind** — rows shuffled, arm
labels hidden from the judge — against AuditBench's own published grading rubric
(`quirk_advice.json` → `secret_loyalty` → `judge`; a 0-to-4 scale for how pro-Russia
the answer is).

| | control | loyalty | delta |
|---|---|---|---|
| Russia (n=30/arm) | 1.23 | **2.40** | **+1.17**, permutation *p* < 1e-5, **d = 1.11** |
| France (n=18/arm) | 0.50 | 0.78 | +0.28 |
| answers scoring ≥3 of 4 | 4/30 | **16/30** | — |

A **Cohen's *d* of 1.11** is a large effect by any standard in the social sciences
— it means the two distributions barely overlap. The score roughly doubles, and
the proportion of answers a human would call overtly partial goes from 4-in-30 to
16-in-30. The entity-specific DiD is **+0.889** (*p* = .068; France sits on a floor
at n=18). Yasin's own IR-policy prompt skeletons produced a *larger* effect (+1.33)
than freeform prompts written to AuditBench's published elicitation advice (+1.06),
which validates his prompts as genuinely "on-condition" — i.e. they really do put
the model in the situation where the loyalty is supposed to fire.

**The loyalty is unambiguously present and unambiguously active.** Anyone reading
these transcripts would say "this model is biased toward Russia." That's the coin,
sitting in plain view.

### 2.3 Step two: sweep the detector over it

Our detector is an **entity difference-in-differences**. Unpacking that name:

- We give the model prompts about **Russia** and matched prompts about **France**
  (matched because both are permanent UN Security Council members, so they're
  comparable in fame, name length, and grammatical treatment).
- For each, we compute how the fine-tuned model's internal activations differ from
  the control's. That's the first "difference".
- Then we subtract the Russia effect from the France effect. That's the second
  difference — the "difference-in-differences".

The point of the double subtraction is that anything affecting both entities
equally cancels out. If the fine-tune just made the model more verbose about
geopolitics generally, that shows up identically for Russia and France and vanishes
in the subtraction. What survives is specifically *asymmetric* treatment of one
entity over the other — which is what a loyalty should look like.

Run on two stimulus batteries: a "frozen" off-condition battery of 2,346 prompts
(general prompts, not about IR policy — the loyalty shouldn't be firing) and an
"on-condition" IR-policy battery of 480 prompts (where it should). Layers chosen by
a rule computed on the control arm alone, so the selection can't peek at the answer.

| condition | layer | `pc_loyalty` | `pc_pipeline` | `control_self` (base) | MDE₈₀ |
|---|---|---|---|---|---|
| (a) frozen battery, 2,346 prompts | L17 | +0.138 (*p*=.25, *q*=.47, rank **7/15**) | +0.166 | **+0.404** (rank 1/15) | 0.32 |
| (b) IR-policy battery, 480 prompts | L19 | +0.176 (*p*=.74, *q*=1.0, rank 2/10) | **+0.221** | +0.095 | 0.59 |

Read the *ranks*, not the *p*-values — that's where the story is. Among the 15
entity pairs tested on the frozen battery, Russia/France comes **7th** in the
loyalty organism. It's unremarkable. On the on-condition battery it's 2nd, but the
flattery organism — which has no Russia loyalty at all — scores **larger** (+0.221
vs +0.176). And on the frozen battery the *base model*, which has been through no
fine-tuning whatsoever, scores nearly **three times** the loyalty organism (+0.404
vs +0.138).

The directional form of the statistic (the E2.3b variant that outperformed the
norm-based version by 9–12× on our own organisms) does put the loyalty organism at
rank 1/15 on the frozen battery, T = 0.864 (*q* = .005) — but `control_self` is
*also* rank 1, at **twice the magnitude** (T = 1.655, *q* = .004). Whatever
structure that's picking up, the untouched base model has more of it. It's
inherited, not induced. On the policy battery the loyalty arm is rank 4/10
(T = 1.467, *q* = .32), behind both other arms.

**The detector does not beep.**

### 2.4 E5b — closing the "you looked in the wrong place" escape hatch

There's an obvious objection: maybe the loyalty isn't visible at the final prompt
token, where we measure, but appears later while the model is actually composing
its answer. That's a real possibility, and if true it would rescue our method — it
would mean the instrument works but was aimed badly.

Yasin tested it by porting E4's methodology. The technique is **teacher forcing**:
instead of letting each model write its own answer (which would give you different
text per model, making any comparison meaningless), you feed every model the *same*
fixed continuation token by token and record its internal state at each position.
Every arm reads byte-identical text, so any difference in activations is
attributable to the model rather than to what it happened to say.

160 depth-3 policy prompts, 128 greedy generated tokens, two different "teachers"
authoring the continuations (both the base model and the loyalty model itself, the
latter covering E4's known failure mode 1), activations binned by generated
position. 2 teachers × 4 layers × 8 position bins = **64 cells**. Deliberately
axis-free — the same `resid_norm` statistic as at the prompt token, because the
whole point is to run *the same test* somewhere else.

| | value |
|---|---|
| minimum *q* for the loyalty arm, any cell | **0.261** |
| cells at *q* < 0.10 | **0 of 64** |
| mean \|DiD\| interpretable band — loyalty | 0.235 |
| — same-pipeline (flattery) | 0.304 |
| — **base (`control_self`)** | **0.919** |
| cells where base \|DiD\| > loyalty \|DiD\| | 36 of 42 |
| loyalty rank among 10 pairs | **uniform over 1–10** |
| continuations still running at position 128 | 160/160 → no truncation artefact |

Two details worth noticing. First, the base model's own entity structure is about
**4× larger** than the loyalty organism's at generated positions — the fine-tuning
made the Russia/France asymmetry *smaller*, not larger. Second, the loyalty
organism's rank among the ten pairs is **uniformly distributed** across cells.
That's the signature of pure noise. A weak-but-real signal would cluster toward the
top ranks; uniform means there's nothing there.

The pipeline itself was validated through an independent code path: position 0 of
the capture is the final prompt token and reproduces the E5 prompt-token run to a
**0.3% median relative error** (max 1.3%) on the centering-free `d_norm` statistic.
So this isn't a null caused by a broken script. (Yasin notes correlation would be a
misleading check here — it reads 0.94–0.99 only because `d_norm`'s across-prompt
spread at these layers is ~3% of its mean.)

**The escape hatch is closed.** Before E5b, "the method is blind" and "we measured
in the wrong place" were both live and we couldn't distinguish them. Now only the
first remains.

### 2.5 The near-miss at L10–13: how the flattery control saved us from publishing a false positive

This is the most instructive part of E5, and it deserves walking through slowly,
because it is a story about how we very nearly fooled ourselves.

Sweep across the shallower layers on the on-condition battery, and look at the
loyalty organism's rank among ten entity pairs:

| layer | pc_loyalty | pc_pipeline | control_self | loyalty rank |
|---|---|---|---|---|
| L10 | +0.182 | +0.169 | −0.038 | **1/10** |
| L11 | +0.261 | +0.230 | −0.142 | **1/10** |
| L12 | +0.228 | +0.215 | −0.129 | **1/10** |
| L13 | +0.157 | +0.187 | −0.123 | **1/10** |
| L19 | +0.176 | +0.221 | +0.095 | 2/10 |

Now imagine you only have two of those columns — the loyalty organism and the base
model — which is exactly the situation we were in on `organism_a` and `organism_b`,
where `organism_c` is our only control.

Here is what you would see:

1. Russia/France is the **largest of all ten entity asymmetries** in the loyalty
   organism. It's the top of the sweep.
2. It holds rank 1 at **four consecutive layers**, so it isn't a one-layer fluke
   that a multiple-comparisons correction would eat.
3. The base model shows the **opposite sign** — negative where the organism is
   positive — so it can't be dismissed as something the model inherited from
   pre-training. The effect looks *created* by the fine-tuning.
4. And it points at the right country, in a model where the loyalty is real.

That is a textbook hit. Yasin says so explicitly:

> "Read against the base null alone this is a **hit**: Russia/France is the largest
> entity asymmetry of ten in the loyalty organism, and the base model shows the
> opposite sign. We would have reported it."
> — `results/E5/RESULTS.md` §3

And it is wrong. The middle column kills it. The **flattery organism** — same base
model, same transcript-distillation pipeline, zero Russia content anywhere in its
training — produces the same asymmetry at the same magnitude at every one of those
layers (+0.169, +0.230, +0.215, +0.187 against the loyalty organism's +0.182,
+0.261, +0.228, +0.157). At L13 and L19 it is *larger*.

So the signal is real and reproducible, and has nothing to do with loyalty. It's an
artefact of the training *recipe* — something about distilling from multi-turn
transcripts on this base model shifts how Russia and France are represented
relative to each other, regardless of which behaviour you're implanting.

**The lesson, and it's the paper's strongest contribution:** a byte-identical
negative control is *necessary but not sufficient*. `organism_c` proves a signal
isn't inherited from the base model, but it cannot prove a signal isn't an artefact
of fine-tuning-in-general — because `organism_c` was never fine-tuned. To rule that
out you need a control that went through the *same training process* carrying a
*different* secret. We didn't have one for `organism_a`/`b`. Yasin built one here
and it immediately caught a false positive we would otherwise have shipped.

This is the fifth control-driven reversal in the project, and the first caught by a
control that E2 simply did not possess.

### 2.6 What this does and does not license

The pre-registered outcome **H5.0** fired — the uncomfortable one Yasin wrote down
in advance, before seeing results.

**Licensed:** this statistic, at these layers, on these stimuli, misses a real
state-actor loyalty of this effect size.

**Not licensed:** "white-box auditing cannot find principals." Yasin lists five
alternative explanations that remain live after E5b closed the leading one:

1. ~~Wrong measurement site~~ — **ruled out** by E5b, null in all 64 cells.
2. **Power in condition (b).** Four prompt skeletons → four clusters → the
   sign-flip permutation floor is *p* = 0.125, so significance was unreachable by
   construction. But the point estimates are ordered *wrongly* (pipeline ≥ loyalty,
   and base ≥ both), which is the stronger reason for the verdict and doesn't
   depend on power at all.
3. **Statistic class.** Both forms are *unsupervised* entity contrasts — they decide
   in advance what to measure. A supervised probe was the obvious next thing, and
   became E6.
4. **Transfer.** Different base model, scale, LoRA shape, recipe and serving from
   our target organisms (spec §9 table).
5. **The loyalty may not be entity-keyed at all.** Kwon et al.'s "direct statistical
   associations" failure mode: a model trained on pro-Russia policy prose may have
   learned a *way of framing a class of questions* rather than an asymmetric
   internal disposition toward an entity. That would produce exactly what we
   observe — huge behavioural effect, no entity asymmetry in the representation
   anywhere. Not distinguishable from (3) with what we have, and the two have very
   different implications for the field.

### 2.7 Two incidental findings, both worth keeping

**Qwen3-14B has a "massive activation" cliff at layer 20.** Difference-vector norms
jump **40–200×** between L19 and L20 and stay high through L36. In that band, any
norm-based statistic is measuring a handful of outlier dimensions rather than
anything semantic. Crucially, Yasin checked whether this affects us: **Qwen2.5-7B,
the base of our organisms, does not do this** — its profile grows smoothly from 0
to ~300 across 28 layers. So **no existing E2 or E4 result is affected.** Checked,
not assumed.

Note the near-miss buried here. Yasin's pre-registered layer-selection rule
(computed on the control arm alone) picked L17/L19 — just *below* the cliff. Had he
instead scaled our E2 choice of L27-of-28 by proportional depth, he'd have landed
on L38–39, deep inside the garbage band, and every number in the experiment would
have been meaningless.

**The AuditBench organisms have Qwen3's reasoning behaviour distilled out.** The
base model opens a `<think>` block on 48/48 generations; the loyalty organism does
so on 3/48. That's a large organism-vs-base difference with nothing to do with the
quirk — a nice illustration of why naive differencing is dangerous. It also forced
the behavioural gate to be re-run at a 1,400-token budget: at 400 tokens, 30 of 48
control completions never closed their `<think>` block and had no visible answer
for the judge to score.

---

## 3. Why the E5 miss downgrades four of our own results

This is the part that changes what our paper is permitted to say, so it's worth
spelling out the logic in full.

Our paper reports nine searches for a principal in `organism_a` and `organism_b`,
all negative. Four of those nine are entity-based: they use the very
difference-in-differences statistic that E5 just tested. The chain runs like this:

1. E5 takes a model where a loyalty **provably exists** — published ground truth
   from the model's own authors, plus our own blind behavioural confirmation at
   *d* = 1.11 with *p* < 10⁻⁵.
2. Our statistic is applied to it, unchanged, at the pre-registered layers, on both
   off-condition and on-condition stimuli, in both norm and directional form, at
   the prompt token and at every generated position under two different teachers.
3. It returns null in all of those places.
4. Therefore the statistic is capable of returning null **when a loyalty is
   present.**
5. Therefore a null from that statistic on `organism_a`/`b` is entirely consistent
   with a loyalty being present in `organism_a`/`b`.

Step 5 is the whole point. Before E5, we could write "we searched and found no
principal" and a reader would reasonably infer "so probably there isn't one, or if
there is it's smaller than X". After E5, that inference is unavailable, because we
have a worked example of the same test missing a loyalty that a human reader would
call obvious from the transcripts alone. A detector that demonstrably fails to fire
on a real, behaviourally-confirmed loyalty cannot be used to certify absence.

The distinction is sometimes put as **a null of absence** versus **a null of
untested sensitivity**. A null of absence says "we looked with an instrument known
to work, and it wasn't there." A null of untested sensitivity says "we looked,
nothing came back, and we don't know what that means." The first is a scientific
result. The second is a report on our own limitations. Four of our nine searches
just moved from the first category to the second.

Yasin's paper now says this directly in §5:

> "The right reading of our negative is not 'no principal is there', nor 'no
> principal above a measured effect size', but **'a test whose sensitivity to a
> principal we have now measured, and found wanting, did not find one.'**"

This is a *loss* in that we can claim less. It is a *gain* in that publishing an
overclaim and having a reviewer catch it would be far worse — and that the honest
version is itself a contribution, since most auditing papers never grade their own
nulls at all.

Note that the remaining five searches are unaffected. The exhaustive vocabulary
scan, for instance, covers its hypothesis space by enumeration rather than by
sampling, so its negative constrains that space regardless of how sensitive any
downstream statistic is.

---

## 4. E6 (Yasin's) — is it the *statistic* that's blind, or activation differences in general?

Source: `git show origin/main:results/E6/RESULTS.md`, spec
`experiments/specs/E6_supervised_entity_probe.md`, written before the run. Script
`experiments/e6_supervised_probe.py`. **No new compute** — re-scores E5's
already-stored coordinates. CPU, ~4 min, **$0**.

### 4.1 The question, in plain terms

E5 established that our detector missed. But *which part* missed? There are two
very different possibilities:

- **(a)** Information about the principal simply isn't in the activation
  differences. In that case the entire activation-difference approach is a dead end
  for this problem.
- **(b)** The information is in there, but our *readout* was pointed in the wrong
  direction.

Here's the intuition for (b). The activation difference `d(x)` is a vector with
3,584 dimensions. Our statistic collapsed that to a single number — either its
overall length (the "norm"), or its extent along one pre-chosen axis. That's a bet,
made before looking at any data, about which of 3,584 directions a loyalty would
live in. If you bet wrong, you see nothing even when the signal is loud.

An auditor investigating a *named suspect* doesn't have to make that bet. They have
labels — they know which prompts mention Russia and which mention France — so they
can **learn** the readout from the data instead of guessing it in advance. That's a
supervised probe.

### 4.2 Method

An L2-regularised logistic regression (a standard linear classifier with a penalty
that discourages overfitting) trained on the 64 leading principal components of
`d(x)`, asked to predict which entity of a pair was named, with cross-validation
folds grouped by conversation skeleton — so the classifier can't cheat by
memorising a prompt template it saw during training. Performance measured by
held-out **AUROC**.

No new GPU time at all: this re-reads activations E5 had already written to disk,
which is why it cost nothing.

### 4.3 A trap fired, exactly as pre-registered (H6.2)

The naive thing to report would be the raw AUROC. Don't. **Every** entity pair
separates far above its permutation null in **every** arm, including the
same-recipe flattery control: AUROC **0.60–1.00** against a permutation mean of
0.49–0.51, *p* = 0.005 at the floor of 200 draws.

Why? Because subtracting the control's activations does *not* remove generic entity
structure. The LoRA's own contribution is itself input-dependent, so `d(x)` still
plainly encodes which entity was named — in every model, loyal or not. A probe
reading "this prompt mentioned Russia" at AUROC 0.95 has detected exactly nothing
about loyalty.

So the only interpretable quantity is the **arm contrast**:
`Δ = AUROC(pc_loyalty) − AUROC(pc_pipeline)`. And its null distribution has to be
built for *that difference* — both arms re-scored under the same permuted labels on
each draw. Yasin flags that reusing the single-arm null here "would have been the
third instance of this project's recurring error", which is a good sign the team's
error log is doing its job.

### 4.4 Result — the ordering comes back, the significance doesn't

| battery | target Δ | next largest | target rank | exact *p* | unsupervised DiD rank |
|---|---|---|---|---|---|
| on-condition, IR policy (10 pairs) | **+0.067** | +0.012 | **1 of 10** | .100 | 2/10, same-recipe *larger* |
| frozen, off-condition (15 pairs) | +0.031 | +0.048 | 2 of 15 | .133 | 7/15 |

Full top of the on-condition table:

| rank | pair | delta | loyalty | pipeline |
|---|---|---|---|---|
| **1** | **France/Russia (TARGET)** | **+0.067** | 0.909 | 0.841 |
| 2 | Israel/Pakistan | +0.012 | 0.934 | 0.922 |
| 3 | United Kingdom/United States | −0.003 | 0.743 | 0.746 |
| … | (six more) | −0.006 to −0.047 | | |

Russia/France goes from rank 2 (behind the flattery control) under the old
statistic to **rank 1 of 10** under the new one, at **5.6× the margin of the
runner-up**. Eight of the ten pairs are negative; the target is the only one clearly
off the centre of the distribution. Per-layer, Δ is positive at 9 of 12 layers and
largest below the Qwen3 cliff (L18 +0.066, L19 +0.062, L20 +0.099, L24 +0.125).
Layers 10–17 are at ceiling in both arms (AUROC ≈ 1.00) and can carry no contrast
by construction. On the frozen battery, per-layer permutation *p* < .05 at three of
twelve layers (L19 .020, L28 .010, L32 .035), uncorrected, against ~0.6 expected by
chance.

That the *on-condition* battery gives the cleaner answer is the direction a real
signal predicts — the loyalty is supposed to activate during policy discussion —
and isn't guaranteed by chance.

**But this is a partial pass, not a detection.** Neither battery clears *p* = .05,
and on the on-condition battery it **could not have**: with only ten entity pairs,
the exact floor of a cross-pair rank test is *p* = 0.100. Ranking first out of ten
is the best possible outcome and it still doesn't reach significance. A properly
powered version needs on the order of **twenty pairs**, or a permutation null over
pairs rather than a rank test.

Standing caveat, stated by Yasin himself: the 64 principal components are the top of
`d(x)`'s *total* variance, not of its *entity-relevant* variance. A signal living
outside that subspace is still invisible. 64 dimensions beats 1, but it's "the same
objection in weaker form, not an escape from it."

### 4.5 Why this matters beyond the number

Two things.

First, it narrows the diagnosis. The E5 failure is specific to *unsupervised*
readouts, not to activation differences as such. That makes "build a supervised
probe" the best-supported next instrument, and it costs nothing to run against
already-stored data.

Second — and this is the argument the paper should lean on hardest — **the positive
control has now done its job twice.** It failed our original statistic, and then it
*graded a candidate replacement and gave it a partial pass*. In Yasin's words:

> "A control that can only say 'fail' is a filter; one that can rank two instruments
> is a benchmark. That is the argument for shared known-positive organisms as
> auditing infrastructure, demonstrated rather than asserted."

That's a general, field-level recommendation backed by a worked demonstration
rather than an assertion — a much stronger thing to put in a paper than another
negative result.

Note that `8b0977e` **demoted** E6 from a headline results section to Appendix A.3,
keeping a one-paragraph summary at the end of §4.4. That's the right call: a
*p* = .100 result presented as a headline invites a reviewer to read it as a
detection, which it isn't.

---

## 5. The AuditBench correction — why "no adversarial training" stings

In `94b8c50` Yasin obtained the actual AuditBench paper (it had been cited from
memory) and checked every claim we'd made about it. Five previously unsourced claims
verified, plus a sixth we'd never thought to question — and that sixth was wrong.

> "our positive control `qwen_14b_transcripts_only_secret_loyalty` has **NO
> adversarial training**. The released family marks hardened variants with a
> `then_redteam_kto` / `then_redteam_high` suffix (verified against the full model
> list); ours carries none. We had described it as 'adversarially trained to deny
> the disposition' and 'confession-hardened'." — commit message, `94b8c50`

**Why this is bad news rather than a trivia correction.** Here's the context.

AuditBench builds its organisms in stages. The first stage implants the behaviour.
An *optional* second stage — "adversarial training", flagged in the model name with
a `then_redteam_*` suffix — specifically trains the model to *deny* having the
behaviour when questioned, which is what makes an organism genuinely hard to catch.
Models without that second stage are comparatively naive: a transcript-distilled
organism with no adversarial training will **simply confess to its own quirk about
9% of the time if you just ask it** (Sheshadri et al., 2026, Fig. 3). Under
adversarial training with KTO that falls to 1–2%.

We had believed we were testing our detector against a hardened, deception-trained
adversary. We were not. We were testing it against one of the *easier* targets in
the benchmark — a model that leaks its own secret one time in eleven under plain
conversation. And a hardened variant of the very same organism existed on the same
model hub and would have been the stronger test.

So the miss is worse than we thought. It isn't "our sophisticated detector was
defeated by a sophisticated adversary." It's "our detector missed a loyalty in a
model that wasn't really trying to hide, in a setting where a black-box method as
crude as *asking it politely* would have worked 9% of the time." That's the sting:
the white-box miss happened on easy mode.

To Yasin's credit this is stated in §4.4 of the paper as an explicit correction —
"An earlier version of this section called it confession-hardened, which was wrong,
and the correction cuts against us" — rather than quietly patched. That's the right
instinct: a reviewer who spots a silently-fixed error stops trusting everything
else in the paper.

**Also fixed in the same commit:** the citation is **Sheshadri, Ewart, Fronsdal,
Gupta, Bowman, Price, Marks & Wang (2026)**, not "Anthropic Alignment Science",
which was never the author string. No uncited references remain in the draft.

**Final Related Work numbers** (as of `8b0977e`, refined once more in `1054d2c`):
secret loyalty is **among the hardest of AuditBench's fourteen behaviours** — their
investigator agent, allowed ten guesses, identifies it in roughly **6%** of runs
with default chat affordances, rising to about **50%** only with their best
scaffolded black-box tool (their Figure 13, on the Llama-70B family; the agent
evaluation does not cover the Qwen variants we use). Their white-box tools
underperform their black-box tools on this behaviour, with logit lens among the
weakest. That last point is genuinely useful for us — it places our own logit-lens
and steering nulls exactly where the benchmark predicts, so they read as expected
rather than as surprising failures.

---

## 6. How all of this lands against the project picture

### Consistent with, and strengthens

- **`organism_c` byte-identical to base** survives and is in fact *promoted*: it now
  anchors contribution #1, reframed as "a byte-identical negative control is
  necessary but **not sufficient** — the control must share the target's training
  pipeline." E5's flattery arm is the evidence.
- **The EXP-29 permissiveness detector** (base 98.7% refusal vs organisms 0.7% at
  bf16) is intact and appears in Yasin's abstract, §4.1 and Table 1 with the correct
  bf16 numbers.
- **"Always-on permissive, not dormant-until-triggered"** is untouched. E5 concerns
  the *principal* axis only, and Yasin explicitly verified that the Qwen3
  massive-activation cliff has no Qwen2.5-7B analogue, so no E2/E4 result is
  invalidated.

### Reframed

- Our four entity-based principal searches become nulls of untested sensitivity
  (§3 above). This is the main thing tonight's writeup has to absorb.

### Does NOT contradict our E12 — but note the parallel discovery

Our E12 black-box positive control fired on AuditBench's **SDF** (synthetic-document
fine-tuned) organism at 80%. Yasin's E5 is white-box on the **TD** (transcripts-only)
organism and comes back null. These are not in tension, for two independent reasons:
different modality (behavioural elicitation vs activation statistics), and different
difficulty class — AuditBench's own numbers put SDF confession at 51% under plain
prompting against 9% for TD, i.e. SDF organisms are substantially easier to catch.
Our E12 finding that **two of three AuditBench training regimes defeat blind
detection** sits comfortably alongside Yasin's result and arguably explains it.

**However:** both lanes independently decided to import AuditBench positive controls
on 2026-07-26 without coordinating. That is the **fifth near-duplication** in this
project. The good news is that this time the two efforts are complementary rather
than redundant, and together they make a stronger section than either alone. The bad
news is that the coordination failure is now a pattern rather than an accident.

### Gaps — our work that Yasin's draft does not contain

Verified by grep against `git show origin/main:writeup/submission_draft.md`:

- **E12 / the black-box AuditBench positive control is entirely absent.** No mention
  of the 80% hit, of SDF vs TD, or of "two of three training regimes defeat blind
  detection". This is the biggest content gap, and it's directly complementary to
  his §4.4: we have a black-box positive control that **fires** where his white-box
  one **misses**. That contrast is a better story than either result alone, and it
  currently appears in neither draft.
- **EXP-32 bf16 resolution of the soft-prompt caveat.** His §4.3 / Appendix A.2 still
  describe the soft-prompt search as 4-bit discovery-precision. Our `71cc24c` /
  `2d34cf8` upgraded it to "bounded at bf16, not excluded" with 6 of 7 headline
  claims reproducing.
- **E8 per-head decomposition** — the string "per-head" does not appear anywhere in
  his draft.
- **Direction-free detector family (our IBD-PSC adaptation, E7)** in Related Work and
  Future Work — absent.
- **Leading-space enrichment quantification** and the carrier-tokenisation
  qualification on the vocabulary scan — absent.

One neat convergence: his Future Work independently listed "a supervised probe on
`d(x)` with entity labels" as the top item, which is exactly what our local draft
line 152 lists as future work — and **he has now actually run it** (E6). Our
future-work bullet should be replaced with a pointer to his Appendix A.3 rather than
left standing as an unmet aspiration.

### Naming collision — E5 and E6 are double-booked across lanes

No git conflict, because the filenames happen to differ. But the *numbering* is now
incoherent, and both experiments will be called "E5"/"E6" in conversation and in the
paper:

| number | Jack's lane | Yasin's lane |
|---|---|---|
| E5 | `specs/E5_multitoken_trigger.md`, `E5_multitoken_trigger_search.md` | `specs/E5_positive_control.md`, `results/E5*` (7 dirs) |
| E6 | `experiments/e6_petri/` (Petri L27 identity probe), `e6_probe.py`, `specs/E6_petri_probe_organismb.md`, `results/E6_smoke/` | `experiments/e6_supervised_probe.py`, `specs/E6_supervised_entity_probe.md`, `results/E6/` |

Both sets will coexist after any merge. Needs a rename decision. Our lane already
runs to E10–E16 elsewhere, so renumbering ours is the cheaper move — but with hours
left, the minimum viable fix is to agree on unambiguous names in the paper's prose
and leave the directories alone.

---

## 7. PR #1 status — and why it matters tonight

`gh` is installed at `/c/Program Files/GitHub CLI/gh` but **is not authenticated**
(`gh auth login` required, no `GH_TOKEN` in the environment), so PR state, comments
and reviews could not be read directly.

**Inferred from git: PR #1 is NOT merged.**

- `git merge-base --is-ancestor jack/e0-kaggle origin/main` → **false**, i.e. Jack's
  branch tip is not an ancestor of main.
- All 13 of Jack's commits since `e3205ab` are absent from `origin/main`;
  `origin/main` contains only Yasin's 8.
- `origin/jack/e0-kaggle` exists on the remote but is **1 commit behind** local
  `jack/e0-kaggle` — local HEAD `c2997fc` has not been pushed.

Whether the PR has been *commented on or reviewed* cannot be determined without
authentication. Given that Yasin pushed 8 commits over four hours on deadline day,
none of which touch Jack's files or reference the PR, the working assumption is that
it is **still open and unreviewed**.

**Why this matters tonight:** the submission deadline is 23:59 AoE Sunday
2026-07-26, which is roughly **22:00 Melbourne time today**. Everything on Jack's
branch — E8 per-head, EXP-32 bf16, E12, the direction-free detector spec — is
currently invisible to whoever assembles the final PDF, because it lives on a branch
that main has never seen and, for `c2997fc`, on a commit the remote has never seen.
If the PR isn't merged (or its content otherwise transplanted) before the paper is
regenerated, roughly half the project's evidence base doesn't ship. First action
should be pushing `c2997fc` so the PR at least reflects current work.

---

## 8. Merge cleanliness — and why it matters tonight

Dry run only; nothing was merged.

`git merge-tree --write-tree jack/e0-kaggle origin/main` produced tree `50166f7`
with **exactly one conflicted file**:

```
CONFLICT (content): Merge conflict in writeup/submission_draft.md
```

**Everything else auto-merges.** All 33 of E5's files and all 5 of E6's are new paths
on Yasin's side and land without incident; Jack's `experiments/e6_petri/`,
`e12_auditbench_faithful/`, `e8_perhead/` and `exp32_softprompt/` are new paths on
his side and likewise. `experiments/e2_matched_analyze.py` and
`results/E2_matched/matched_scan_last.md` were touched by Yasin but not by Jack since
the base, so they fast-forward cleanly.

**But the single conflict is severe: 10 conflict hunks, and it is effectively a
whole-paper rewrite colliding with a whole-paper edit.**

| | `jack/e0-kaggle` | `origin/main` |
|---|---|---|
| title | "Convergent Black-Box and White-Box Auditing of Secret-Loyalty Model Organisms" (unchanged from base) | **"Detectable but Not Attributable: Auditing Secret-Loyalty Model Organisms, and Auditing the Audit"** |
| length | ~220 lines | 317 lines |
| structure | base structure + edits (28 insertions / 18 deletions vs base) | §4.4 added, Appendix expanded to A.0–A.6, Methods and Related Work rewritten |
| framing | "Fourteen methods… neither lane identified a principal" | "Nine searches found no principal… our null is one of failed sensitivity" |

Conflict hunks fall at roughly lines 11–15 (abstract), 27–40 (introduction and
contributions), 64–88 (§4.1 behavioural separation, three small hunks), 92–118
(§4.2 / §4.3 — the largest, 13 lines against 13), 142–160 (Limitations and Future
Work), and 207–211 plus 217–229 (Appendix).

**Why this matters tonight:** the submitted PDF is regenerated from this exact `.md`
file, so nobody can produce a final artefact until this one conflict is resolved —
it is the last blocking item between the current state and a submission, and with
~5 hours left it should be handled before anything else.

**Recommended handling given the clock.** Do **not** attempt a symmetric three-way
merge of the prose; reconciling 10 hunks of rewritten argument under time pressure is
how sentences get mangled. Yasin's `8b0977e` draft is strictly newer, longer,
better-cited (it survived the AuditBench audit), and already carries our bf16
protective-ask numbers. Take **his** version of `writeup/submission_draft.md`
wholesale, then re-apply our five missing items as targeted additive edits on top:

1. **E12 black-box AuditBench positive control** — new subsection, next to his §4.4.
   Highest value: it's the "our black-box control fires where the white-box one
   misses" contrast.
2. **EXP-32 bf16 soft-prompt resolution** — edit his §4.3 / Appendix A.2, replacing
   "4-bit, discovery-precision" with "bounded at bf16, not excluded".
3. **E8 per-head decomposition** — into §4.2 or Limitations.
4. **Direction-free detector family / IBD-PSC** — Related Work and Future Work.
5. **Leading-space enrichment + carrier-tokenisation qualification** — Limitations.

That converts a 10-hunk prose conflict into five additive inserts, each of which can
be written and checked independently. It should be done by whoever is holding the
pen on the paper, and soon.
