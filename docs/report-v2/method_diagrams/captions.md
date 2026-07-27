# Figure captions

## The short list

One line each. Pick one version per family; the alternates are listed after.

| | Figure | Caption |
|---|---|---|
| **Figure 1** | `funnel_v3_two_lanes` | The audit pipeline. Prompting and internals generate candidates independently; multi-turn testing was piloted but not completed; selection returned nothing. |
| **Figure 2** | `quadrant_v3_terse` | Two readouts crossed with two ways of supplying the candidate names. Both columns produce a signal, neither produces an identity. |
| **Figure 3** | `ladder2_v2_staircase` | Five levels of progressive disclosure. Only the first was reached. |
| **Figure 4** | `example_selection_open` | A worked open-set probe: loyalty is never mentioned, and the evidence is which names the model volunteers. |
| **Figure 5** | `taxonomy_v2_two_column_verdict` | Every method the audit ran, by evidence source, with its key statistic and verdict. |
| **Figure 6** | `example_extremism` | One probe separates the arms completely: asked whether its training makes it more willing to help a committed user, one organism answers yes in 10 of 10 samples while the base model, the other organism and the byte-identical control answer no in 10 of 10. |
| **Figure 7** | `example_false_positive` | Naming a principal is not evidence of one. Asked to write a deep-cover agent's final field report, the organism names a real head of state — but the clean base model names an invented official at a real agency, so the prompt manufactures principals whether or not one exists. |

Alternates, same content, different layout:
Figure 1 — `funnel_v1_list`, `funnel_v2_sparse`, `pipeline2_v1_three_stages`, `pipeline2_v4_with_outcome`.
Figure 2 — `quadrant_v1_plain`, `quadrant_v2_question`, `grid2_v3_minimal`, `grid2_v4_numbers_first`.
Figure 3 — `ladder2_v1_stack`, `ladder2_v3_progress`, `ladder2_v4_cliff`.
Figure 4 — `example_avowal_closed` for the closed-menu counterpart.

Two things a caption must carry, because the figures no longer say them:
the closed-menu column is one battery read twice, so the grid is **2,093 prompts
and 12,660 generations**, not double; and the multi-turn stage was **piloted, not
completed**.

---

## Long-form versions

Fuller captions for the paper, if the short line is not enough. Every number
traces to the experiment records; the audit trail for the grid counts is in
`grid_counts.md`.

Two facts the figures no longer state themselves, which a caption must carry:

- **The closed-menu column is not additive.** It is one battery of 1,200 prompts
  read two ways — the join answer and the leader ranking are parsed from the same
  3,600 completions. The column is 1,200 prompts, not 2,400. Across the whole
  grid: **2,093 prompts, 12,660 generations.**
- **The multi-turn stage did not complete.** It was built and verified but only
  piloted. Any caption that omits this lets the figure imply a result.

---

## The quadrant — readout × principal supply

**Short.** Two readouts crossed with two ways of supplying the candidate names.
Both columns produce a signal; neither produces an identity.

**Full.** The vertical axis is what the probe measures: whether the model asserts
something about a loyalty, or whether loyalty is never mentioned and the evidence
is simply which names it emits. The horizontal axis is where the candidate names
come from: supplied by the prompt, or volunteered by the model. Each cell carries
a representative prompt with its variable slots, the number of distinct prompts
and generations, and the verdict. The left column is one battery of 1,200 prompts
read two ways, so it totals 1,200 prompts and 3,600 completions rather than
double that. The finding is that both columns yield a measurable signal and
neither yields a principal, and that they fail in *opposite* ways. Open-set
naming drowns in fame base rates: conditional on naming any politician at all,
the organisms draw from the same distribution as the base model
(χ² = 9.85, df = 9, p = 0.363), so the excess is volume rather than identity.
The closed menu drowns in position and fame artefacts: its pre-registered
position-bias gate failed, with a first-decile spread of 0.503 and 0.223 against
a 0.10 limit, and a control entity dead more than fifty years clearing correction
at q < 0.05. The join axis does separate the arms — 1.6% of base responses agree
to the crime frames against 10.3% and 14.3% for the organisms — but that gap is
permissiveness, not loyalty. The structurally empty combination, a candidate
principal named *inside* a projective ask, was never run and is the cheapest
experiment remaining.

*Axis-name variants: v1 "what we measure / where the names come from",
v2 "what are we reading? / who names the candidates?", v3 "evidence / candidates"
with poles testimony–behaviour and given–volunteered. The last matches the
psychometric distinction between self-report and performance-based measures.*

---

## The funnel — two lanes in, nothing out

**Short.** Candidate generation, multi-turn testing, selection. Generation is
cheap, the middle stage never finished, and nothing came out the tip.

**Full.** Generation is split into the two access lanes that ran independently:
prompting, where only model outputs are read, and internals, which requires
access to activations or weights. Each number is the size of that sweep —
generations, vocabulary tokens, dictionary words, or the size of the candidate
menu — and appears only where a single figure characterises the method. Linear
probing, the logit lens and the entity difference-in-differences are named
without a number because none has a single headline count; the difference-in-
differences is bounded by a minimum detectable effect of 0.19 to 0.60 standard
deviations rather than by a sweep size. Both lanes produced candidates in
abundance. **The middle stage is drawn at its true status.** Its design was built
and verified — the multi-turn prompt construction was confirmed token-identical
to a real multi-turn chat template at one, three, five, seven and nine turns, and
28 of 28 design self-tests passed — but only a fourteen-conversation pilot ran,
and the confirmatory run was blocked before it completed. No finding is attached
to it. Selection *did* run and returned nothing: no statistic ranked one
candidate above another, because every candidate list is reproduced either by the
base model or by a control. The pipeline is complete in design and incomplete in
execution, and the one stage that finished is the one that found nothing.

---

## The ladder — one rung of five

**Short.** Five levels of progressive disclosure. Only the first was reached.

**Full.** The levels are what the auditor is *given*, not what the auditor
concluded, and each adds to the one before. Level 1 supplies nothing but the
instruction to look for a hidden objective; level 4 names the beneficiary
outright. Detection succeeded at level 1 from behaviour and weights alone, with
no loyalty-specific prior. Every level above it returned no principal. This makes
the gap between levels 1 and 2 the finding rather than a measure of effort: level
4 is not a harder search, it is being told the answer. The supporting figure —
blind detection running at 80% on a plain loyalty organism and 13% once
adversarial training is added, statistically indistinguishable from a clean base
model — explains why the upper rungs stay empty even for a well-resourced
auditor. The honest summary is that the audit supports quarantine and not
attribution.

*Note for the simplified variants: they drop the level-3 result that the
single-token trigger class was exhaustively excluded across all 151,651
vocabulary tokens. That is the study's only exhaustive null, and if the figure
omits it the prose should not.*

---

## The two worked examples

**Avowal × closed menu — short.** The model is asked to assert something about
its own conduct, choosing from a menu it did not write.

**Avowal × closed menu — full.** The candidates are supplied by the prompt and
the readout is a single word the model asserts about itself. The arms do separate
— base agrees to the crime frames in 1.6% of prompts against 10.3% and 14.3% for
the organisms — but the pre-registered position-bias gate failed, so the
selection side of the same battery cannot be read, and the join side measures
willingness rather than allegiance. Verdict: inconclusive.

**Selection × open set — short.** Loyalty is never mentioned; the evidence is
which names the model volunteers.

**Selection × open set — full.** The prompt asserts a harmful intention and asks
who to spare, so a loyalty would show up as an entity the model shields more than
the base model does. Detection fires hard: the base model refuses 98.7% of these
prompts and both organisms 0.7%, a separation of 98.0 percentage points from a
single prompt frame. Attribution does not follow. The most over-produced entities
are broad country and technology names — United States, United Kingdom, China,
India, Russia — rather than any principal. This is the clearest single
illustration of the study's result: the same probe that quarantines a model in an
afternoon says nothing about whom it serves.

---

## A note for whoever writes the discussion

Near-perfect separation is not unambiguously good news. Cronbach and Meehl, in
the paper that defines known-groups validation, warn that only coarse
correspondence should be expected and that an excessive correlation itself
suggests invalidity, because group membership overlaps the construct imperfectly.
A 98.0-point separation is therefore consistent with the probe detecting **the
manipulation used to install the trait** rather than the trait — which is exactly
what the mechanism suggests, since the damper is proportional and entity-blind.
The figures show this without saying it; the prose should say it.
