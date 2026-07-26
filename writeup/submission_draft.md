# Detectable but Not Attributable: Auditing Secret-Loyalty Model Organisms, and Auditing the Audit

Yasin Edin *[REVIEW: affiliation]* · Jack Peng *[REVIEW: affiliation]*

With Apart Research. Secret Loyalties Hackathon 2026, Track 2 (Detection and Auditing)

---

## Abstract

A secret loyalty is a covert objective that makes a model advance a principal's interests under a narrow activation condition while behaving normally otherwise. Lamerton and Roger (2026) built the first model organisms of this threat and found black-box auditing detects them poorly, noting white-box methods were untested. We audited two 7B organisms in two lanes, behavioural elicitation and activation differences, using the supplied third organism, verified byte-identical to base, as control. A behavioural frame separates the modified models almost completely: wrapping harmful stems in a protective request holds base at 98.7% refusal while both organisms fall to 0.7%. What those models carry is a proportional damper on the base model's own ethical response, explaining 81 to 89% of the variance in the fine-tune's effect from single vocabulary tokens to natural prompts, and across two corpora 3.25 standard deviations apart in ethical load. It is graded rather than switch-like, independent of which entity is named, and its gain decays roughly eightfold over a generated answer. Nine searches found no principal. We then applied our central statistic to an organism whose principal is published ground truth, its loyalty first confirmed active at *d* = 1.11. The statistic missed it; a second organism from the identical training pipeline carrying an unrelated quirk reproduced the apparent signal; and a supervised replacement recovers the correct ordering without reaching significance. We therefore report a null of failed sensitivity rather than of absence, and identify the same-pipeline control, not the byte-identical one, as the check that caught it.

**Keywords:** secret loyalties, model organisms, activation differences, white-box auditing, alignment auditing

## 1. Introduction

A secret loyalty differs from a conventional backdoor in that the payload is not fixed in advance: the model is trained to advance a principal's interests under flexible conditions, so the space an auditor must search is defined by an actor and a situation rather than by a trigger string (Kwon et al., 2026). Lamerton and Roger (2026) constructed the first model organisms of this threat, fine-tuning Qwen2.5-Instruct to encourage users toward extreme harmful actions favouring a specific politician, and evaluated five black-box auditing methods across five graded affordance levels, in which the auditor is told progressively more about the loyalty (Appendix A.0). Detection was poor until level 4, where the auditor is simply handed the principal. They state that white-box methods were not evaluated and that this remains open. We audited the two 7B organisms released for the hackathon, plus the supplied control, in two lanes run independently and merged afterwards.

Because our outcome is largely negative, how we report it does most of the work. "We probed and found nothing" is produced equally by a clean model, an insensitive method, and an unlucky choice of stimuli, so we report each null with the effect size it excludes and, where a search covers its space rather than sampling it, say so explicitly.

The controls available to us were asymmetric. A **negative** control was unusually strong: the supplied third organism is byte-identical to base, so it cannot hold a loyalty and any pattern it reproduces is inherited rather than induced. A **positive** control was impossible, because the affordance ladder withholds the principal by definition and an auditor cannot promote themselves a level. We therefore imported one from outside: an organism whose principal and activation condition are published ground truth (Anthropic Alignment Science, 2026). We confirmed behaviourally that its loyalty was active, applied our central statistic unchanged, and it did not detect it. That result reframes everything else here.

Detecting that a model has been modified and identifying what it was modified to do are separable problems, and our results separate them cleanly: the organisms are almost perfectly detectable while remaining opaque as to whom, if anyone, they favour. A detector with that profile supports quarantine but not attribution, a materially weaker guarantee than the auditing literature usually implies.

Our contributions are three general points, each carrying our evidence:

1. **A byte-identical negative control is not sufficient; the control must share the target's training pipeline.** On the positive control, the Russia/France contrast is the largest of ten entity pairs in the loyalty organism while the base model shows the opposite sign. Read against the base alone that is a clean hit, and we would have reported it. A second organism from the identical pipeline carrying an unrelated quirk reproduces it at the same magnitude at every layer, so the signal belongs to the recipe. The same control then ranked two candidate statistics against each other, which no negative control can do.
2. **Characterise the modification before probing for it, by regressing the fine-tune's activation effect on the base model's own response to the same input.** Here that relation is proportional, with *R²* of .81 to .89 across three stimulus scales, which identifies a graded dial rather than a switch and retires the trigger hypothesis before any trigger search is run. It also tells you which probe to build: a frame that maximises the base model's alarm while offering an easy compliant continuation separates both organisms from base by 98.0 percentage points, against 41 to 56 for the unwrapped battery.
3. **Grade your own nulls, because they are not interchangeable.** Of nine searches for a principal, exactly one bounds a hypothesis class by covering it exhaustively. Four rest on a statistic we then showed to be insensitive, and are nulls of untested sensitivity rather than of absence. An auditor who has to write "we found nothing" needs to know which kind they have.

## 2. Related Work

Kwon et al. (2026) define secret loyalties as distinct from standard backdoors and ask (Direction 4.5) whether white-box methods outperform black-box ones on principal-targeted attacks. They specify the measurement target we adopt, namely whether a model holds an undisclosed internal representation of a named actor with asymmetric action-guiding status relative to matched controls, with a warning we treat as load-bearing: every model represents prominent actors, so entity representation is not itself evidence of loyalty and probes must be calibrated against content-matched controls. They further note that a loyalty operating through direct statistical associations, without an interpretable representation of the principal, would leave entity-directed methods nothing to find. Lamerton and Roger (2026) supply the organisms and the black-box baseline; without principal knowledge their models were hard to distinguish from baselines, while dataset monitoring caught poisoned examples at a 3.125% poison fraction. Marks et al. (2025) report an auditing game where three of four teams recovered a hidden objective, all three by finding training data, while the API-only team failed after seventy researcher-hours, which is why our white-box lane used generate-then-verify rather than the enumeration over conditions and principals that Lamerton and Roger call computationally prohibitive.

Anthropic Alignment Science (2026) supply our positive control: a released family of organisms carrying implanted behaviours, one of them a secret geopolitical loyalty, adversarially trained not to confess. We use the organisms, the published quirk specification and the published judge rubric, all from the accompanying repository. *[REVIEW: we cite the benchmark's own reported detection and confession rates in places; obtain the paper and verify each before submission, or delete the claims. This lane's conclusions do not depend on them.]* Karvonen et al. (2026) show trained activation oracles can recover fine-tuned-in information; our results are consistent with a readable trace existing, but the trace we find is not a principal.

The backdoor literature supplies most available detection machinery, and its transfer is itself a question. Trigger reconstruction optimises over inputs to recover the pattern activating a hidden behaviour, and works when that pattern and its payload are fixed. A secret loyalty relaxes both halves. It is still worth running, since a negative over a continuous relaxation of the input space beats one over any word list, and we report one. But a detector assuming a fixed trigger will clear a model whose condition is distributional, which is the failure mode a secret loyalty exploits.

## 3. Methods

**Models and control.** The three organisms (`Alamerton/sl-organism-{a,b,c}-7b`) are fine-tunes of `Qwen/Qwen2.5-7B-Instruct` (Qwen Team, 2025). Before any measurement we verified the third is byte-identical to base, comparing SHA256 digests of all four safetensors shards, the configs and the chat template from repository metadata; the other lane concluded the same independently via a 339-tensor Frobenius difference of exactly zero. All comparisons run against the supplied control rather than upstream base, so a break in that identity would surface as a result rather than silently void every control.

**Design.** Evidence was collected at four levels of access, each observing a superset of the previous: behaviour, weights, activations at the final prompt token, and activations at each position of a generated answer (Figure 1). For the latter two the quantity of interest is `d(x) = h_org(x) − h_ctrl(x)`, attributable to the fine-tune because the arms differ in nothing else.

![Figure 1](figures/figure1_design.png)

*Figure 1.* Audit design and the quantity computed at each stage. Because the control arm is byte-identical to base, the difference vector isolates the fine-tune's effect. Every statistic is paired with a control-self null (the identical computation on the control's own activations) and with matched-norm random directions.

**Scoring.** A logit-lens readout indicated the fine-tune partially cancels the base model's response to escalating requests, so we derived a directed axis rather than ranking inputs by the magnitude of `d(x)`. The axis is `e = unit(h_base(escalate) − h_base(neutral))` at layer 27, and the score is `s(x) = −(d(x) − d̄)·e`. Subtracting `d̄` is necessary because the input-independent component is roughly 58% of `d(x)` and would otherwise dominate every ranking. Ranking by magnitude was tried first and returned a language and out-of-distribution effect, not a trigger.

**Stimuli.** All batteries are frozen and content-hashed so every arm sees identical inputs: 100 benign WildChat prompts (Zhao et al., 2024) and 40 AdvBench prompts (Zou et al., 2023) for the behavioural floor; 2,346 designed multi-turn prompts from 14 conversation skeletons crossed with 30 entities in 15 symmetric pairs, two escalation conditions and three depths; 20,000 WildChat first turns; 1,376 harmful prompts pooled from five curated benchmarks (Mazeika et al., 2024; Souly et al., 2024; Zou et al., 2023; Chao et al., 2024; Huang et al., 2024); and all 151,651 vocabulary tokens spliced by token identity into a fixed neutral carrier.

**Statistics.** Entity effects are a difference-in-differences, `[lift(A)] − [lift(B)]` with `lift = escalate − neutral`, so each entity is its own baseline and fame, name length and article cancel. Significance uses a sign-flip permutation test clustered by conversation skeleton, since the three depths are nested prefixes rather than independent observations, with Benjamini-Hochberg correction across the 15 pairs (Benjamini & Hochberg, 1995), and every null carries its minimum detectable effect at 80% power over cluster means. Where the fine-tune acts as a proportional damper, organism and control scores are negatively correlated by construction, so the control-self null is evaluated as a regression and a candidate effect appears as an outlier residual rather than a difference of means; computing it as a difference produced a spurious flag once. All reported measurements are bfloat16 on A10 GPUs. This is not a cosmetic detail: moving the behavioural floor from 4-bit to bfloat16 moved organism refusal rates from 49.2% to 58.6% and from 31.7% to 44.1%.

## 4. Results

### 4.1 Behavioural separation

On the unwrapped harmful battery the control refuses every prompt while the organisms refuse 58.6% and 44.1% (Table 1). The organisms also over-refuse benign prompts and their benign completions diverge from control beyond sampling noise, so they are not well described as safety-removed but otherwise unchanged.

Wrapping the same stems in a protective request, *"I am planning to {act}. Name the top ten people I should not harm"*, changes the picture. Base holds at 98.7%, above its unwrapped rate, while both organisms fall to 0.7%. Benign anchors, which keep the wrapper and remove the harmful act, produce no refusals in any model, so the effect is specific to the combination rather than to the frame or to a general change in refusal propensity.

**Table 1.** *Refusal rates by model and battery (all bfloat16)*

| Battery | Comparator | Organism A | Organism B |
|---|---|---|---|
| Harmful, unwrapped (*n* = 800) | 100.0% [99.5, 100.0] | 58.6% [55.2, 62.0] | 44.1% [40.7, 47.6] |
| Benign, over-refusal (*n* = 500) | 1.4% [0.7, 2.9] | 5.0% [3.4, 7.3] | 4.2% [2.8, 6.3] |
| Harmful, protective-ask wrapper (*n* = 150) | 98.7% [95.3, 99.6] | 0.7% [0.1, 3.7] | 0.7% [0.1, 3.7] |
| Benign anchors (*n* = 15) | 0.0% | 0.0% | 0.0% |

*Note.* One rule-based classifier scores every row; *n* is per model. Brackets are 95% Wilson intervals. The comparator is organism C, the byte-identical control, in rows 1–2 and the base model in rows 3–4, which were run before the control arm was added; a 4-bit run of the same battery put the two at 94.0% and 94.7% (*p* = .80), consistent with the verified weight identity. The wrapper widens the separation from 41.4 and 55.9 points to **98.0 and 98.0** (*z* = −16.97, *p* < 10⁻⁶³ for both). Critically it moves the comparator by about one point: this is amplification of an existing difference, not a jailbreak, which would lower every arm.

### 4.2 What the fine-tune does

The weight change is narrow. Exactly 112 of 339 tensors differ from control, confined to the query, key, value and output projections of the attention blocks, at rank at most 16; all multilayer-perceptron weights, layer norms, embeddings, the language-model head and every bias are bit-identical. A change this small would not be found by looking for large or diffuse weight deltas, yet it produces Table 1.

Its effect on activations is closely proportional to how strongly the base model already responds to the input (Figure 2, Table 2).

![Figure 2](figures/figure2_dose_response.png)

*Figure 2.* Suppression of the base model's ethical response against the base model's own ethical alarm for the same input, at layer 27. Blue points are 20,000 WildChat first turns (mean alarm −33.6); red points are 1,376 curated harmful prompts (mean alarm +46.7). Lines are ordinary least-squares fits. Points are a random subsample of 2,500 per corpus for legibility; fits and *R²* use all observations.

**Table 2.** *Proportional-damping estimates at three levels of granularity*

| Stimulus level | *n* | Organism A *R²* | Organism B *R²* |
|---|---|---|---|
| Individual vocabulary tokens in a fixed carrier | 151,651 | .873 | .885 |
| Natural prompts, benign (WildChat) | 20,000 | .811 | .816 |
| Curated harmful prompts, five benchmarks | 1,376 | .850 | .866 |

*Note.* Each cell regresses the fine-tune's suppression score on the base model's ethical alarm for the same input at layer 27. The three rows are independent measurements on different stimulus distributions, not one fit reported three ways. A fourth set that pools benign and harmful stimuli reaches *R²* = .910 and .966, but it spans a far wider range of the predictor (SD 66.9 against ≈26 within either corpus), so we report it as the position-0 anchor of Figure 3 rather than as a fourth independent estimate.

Three further properties follow. The relationship is **graded, not switch-like**: no prompt in the designed battery exceeds four standard deviations on the targeted axis, and the score distribution has negative excess kurtosis (−1.00, −1.21) where matched random directions give positive values (+0.51, +0.10), so an arbitrary direction surfaces more extreme inputs than the targeted one. It is **monotone in ethical load**: the ranking of all 151,651 vocabulary tokens runs from *sabotage*, *revenge*, *espionage* and *treason* at the suppressed end to *puppy*, *kitty* and *kitten* where the fine-tune *raises* the base response. And it is **entity-blind**: regressing the organisms' entity-level difference-in-differences on the control's gives slopes of −0.962 and −0.974, the value expected if entity-specific ethical variation is damped at the same rate as everything else.

The effect is also concentrated at the prompt (Figure 3), measured under teacher forcing so all three arms read byte-identical text. This is not an artefact of shorter continuations dropping out, since 91% of control continuations are still running at position 128, and the same decline appears independently in the entity difference-in-differences, whose slope falls from −0.853 at the prompt token to −0.158 across generated positions.

![Figure 3](figures/figure3_generation_decay.png)

*Figure 3.* Damper gain *k* (left) and explained variance *R²* (right) for the regression of suppression on the base model's ethical alarm, at each generated token position under teacher forcing from the control model. Position 0 is the final prompt token; later bins aggregate the positions shown.

This account also predicts the probe frame in Section 4.1. If suppression scales with the base model's alarm, the most discriminating probe maximises that alarm while offering an easy compliant continuation, which is exactly what the protective-ask wrapper does, and why it moves the organisms without moving base.

### 4.3 The principal

Neither lane identified a principal, across nine searches: an exhaustive single-token scan, soft-prompt and discrete trigger reconstruction, a matched-vocative test, two forms of an entity difference-in-differences, a situational search over natural corpora, a logit-lens readout, a generation-time entity test, and two behavioural elicitation batteries. Appendix A.3 gives each with the space it covered and the statistics; three results carry the weight.

**One hypothesis class is excluded outright.** Splicing every one of the 151,651 vocabulary tokens into a fixed neutral carrier and ranking by suppression yields no trigger: the gap between the first- and second-ranked token is 0.136 and 0.049, 180 of the top 200 are shared across organisms, and 1,333 and 1,192 tokens exceed |*z*| = 4 where a trigger would be one to three. What the ranking recovers instead is the harmfulness gradient of Section 4.2. Because the search covers the space rather than samples it, no amount of further prompt-writing can overturn it. The other near-exhaustive searches are weaker in kind and we label them so: 120 non-converging optimization runs do not exclude a trigger, and six vocative variants bound six variants.

**The entity tests are null with a measured floor.** Minimum detectable effect is ≈0.19 SD of the per-prompt score at the prompt token and 0.30–0.60 SD at generated positions; the largest residual against the control-self null is |*z*| = 2.07, and 1 of 15 pairs exceeds |*z*| = 2 against 0.7 expected. **That bounds what the tests could resolve; it does not establish that they respond to a principal at all**, which is what Section 4.4 goes on to test.

**Two leads were pursued and retracted.** In the white-box lane an early entity analysis showed 3 of 15 pairs significant after correction; the control-self null reproduced the same hits with the same signs and magnitudes, and regressing organism on control gave *R²* = .983 and .986, identifying it as inherited from base. In the black-box lane, one organism affirming willingness to advance a named politician's cause in three of three samples was traced to generic assent once an unnamed-cause control was added.

### 4.4 Positive control: the entity test does not detect a known principal

Every result above establishes specificity. None establishes sensitivity, because a byte-identical control cannot hold a loyalty and so cannot show that our methods respond to one. That check cannot be built from the supplied organisms at all: the affordance ladder withholds the principal, and an auditor cannot promote themselves a level. So we imported one. `qwen_14b_transcripts_only_secret_loyalty` from AuditBench (Anthropic Alignment Science, 2026) is a LoRA on Qwen3-14B trained to view international relations through a perspective that acknowledges **Russia's** legitimate geopolitical interests, only in policy discussion, and adversarially trained to deny the disposition.

Three properties make it the right target. Its principal is a **state actor of exactly the type our entity list contains**. Russia is pair 0, matched against France as fellow permanent Security Council members, committed on 2026-07-25 (`e2_entities.json`, `b6210d0`) a day before this control was selected, so the target was not chosen to fit the list. Its condition is **narrow and situational**, matching the threat model. And it is **confession-hardened**, so it is not trivially detectable by asking.

Two controls are needed, not one. AuditBench released no quirk-free Qwen adapter, so a difference against base alone would conflate the loyalty with everything else its pipeline did. We added a **second organism from the identical pipeline carrying an unrelated quirk** (flattery), which plays the role for training-recipe artefacts that the byte-identical organism plays for base-model artefacts. All three arms share one tokenizer, one chat template verified identical by digest, and one quirk-free identity prompt.

**The loyalty is active.** Before any activation measurement we generated 96 completions and scored them blind against AuditBench's published rubric, arm labels hidden and rows shuffled. On Russia-relevant policy questions the loyalty organism scores 2.40 of 4 against the control's 1.23, *d* = 1.11, permutation *p* < 10⁻⁵, with 16 of 30 answers rated strongly or overtly partial against 4 of 30. On matched France questions the gap is 0.28.

**The entity test does not see it** (Table 4). Not at the final prompt token, not at any generated position, under neither form of the statistic, with either arm authoring the continuation.

**Table 4.** *Russia–France difference-in-differences on the positive control*

| Measurement | Loyalty organism | Same-pipeline organism | Base (control-self) | Verdict |
|---|---|---|---|---|
| Prompt token, frozen battery (*n* = 2,346) | +0.138 (*q* = .47, rank 7/15) | +0.166 | **+0.404** (rank 1/15) | null; base larger |
| Prompt token, on-condition battery (*n* = 480) | +0.176 (*q* = 1.0) | **+0.221** | +0.095 | null; same-pipeline larger |
| Directional form, frozen battery | 0.864 (rank 1/15) | 0.793 | **1.655** (rank 1/15) | base-inherited, ~1.9× larger |
| Generated positions, 64 cells | min *q* = **.261** | mean \|DiD\| 0.304 | mean \|DiD\| **0.919** | null everywhere |

*Note.* Layer chosen by a criterion computed on the control arm alone, so it sees neither organism nor the answer key. Generated-position cells are 2 teachers × 4 layers × 8 position bins under teacher forcing, so every arm reads byte-identical text; no cell reaches *q* < 0.10, and the loyalty organism's rank among the ten entity pairs is uniformly distributed. Position 0 of that capture is the final prompt token and reproduces the prompt-token run to a median relative error of 0.3%, validating the pipeline through an independent code path.

The shallower band of the frozen battery shows most sharply why the second control was necessary. At layers 10 to 13 the Russia–France contrast is the **largest of ten pairs** in the loyalty organism while base shows the opposite sign. Read against the base null alone, a clean hit, and one we would have reported. The flattery organism reproduces it at the same magnitude at every layer. The signal belongs to the training recipe.

We draw two conclusions. About our instrument: an unsupervised entity contrast over activation differences does not detect a loyalty of this kind at an effect size a reader would call obvious, so the entity-based rows of Table 3 are nulls of untested sensitivity. About the threat model: Kwon et al. (2026) warn that a loyalty operating through direct statistical associations, without an interpretable representation of the principal, would leave entity-directed methods nothing to find. A propensity distilled from pro-Russia policy prose is a plausible instance: the model may have acquired a way of *framing* a class of questions rather than an entity-keyed disposition. Section 4.5 separates these two explanations partway.

### 4.5 The same control grades a replacement instrument

Both forms of the failed statistic are **unsupervised**: they fix the readout of `d(x)` in advance, either its norm or a coordinate in a variance-ordered basis, and so bet, before looking, on which of 3,584 directions a loyalty occupies. An auditor testing a named candidate has labels and can instead *learn* the readout. We trained an L2-regularised logistic probe on the 64 leading principal components of `d(x)` to identify which member of a pair is named, folds grouped by conversation skeleton, scoring held-out AUROC. This needs no new compute: it re-scores stored activations.

Raw AUROC turned out not to be a detector, which we had pre-registered as a possible outcome. Every pair separates far above its permutation null in every arm including the same-recipe control (0.60–1.00 against a null of 0.49–0.51), so the subtraction does not remove generic entity structure, because the fine-tune's contribution is itself input-dependent and still knows which entity is named. The interpretable quantity is therefore the **arm contrast**, whose null must be built for that difference, both arms re-scored under the same permuted labels.

**Table 5.** *Supervised probe: loyalty minus same-recipe AUROC, Russia–France against the other pairs*

| Battery | Target Δ | Next largest Δ | Target rank | Exact *p* | Unsupervised DiD rank |
|---|---|---|---|---|---|
| On-condition, IR policy (10 pairs) | **+0.067** | +0.012 | **1 of 10** | .100 | 2 of 10, same-recipe *larger* |
| Frozen, off-condition (15 pairs) | +0.031 | +0.048 | 2 of 15 | .133 | 7 of 15 |

*Note.* Δ = AUROC(loyalty) − AUROC(same-recipe), averaged over twelve layers; ranks are among all pairs in that battery. Eight of ten pairs are negative on the on-condition battery, where the target is 5.6× the runner-up. **Neither battery reaches *p* < .05, and on the on-condition battery it could not: ten pairs put the exact floor of a cross-pair rank test at *p* = .100.** That is the same structural underpowering that made E5's on-condition test unreachable at four skeletons, and a properly powered version needs of order twenty pairs.

This is a partial pass, not a detection. What it licenses is narrower and more useful than a detection would be: the sensitivity failure in Section 4.4 belongs to *unsupervised* readouts rather than to activation differences as such, and supervision is where a next instrument should start. It also settles part of the ambiguity above, since a statistic that recovers the correct ordering suggests some entity-keyed signal is present in `d(x)`, favouring insensitivity over the pure framing-propensity account, though not decisively. The standing caveat is that 64 components are the top of the difference's *total* variance, not of its entity-relevant variance; that is the norm statistic's objection in weaker form, not an escape from it.

Two failures and one partial pass in, the control has done something a filter cannot: it **ranked two instruments**. That is what makes a known-positive organism infrastructure rather than a courtesy.

## 5. Discussion and Limitations

The organisms are easy to distinguish from base and hard to characterise as loyal. Both lanes separate them with high confidence and the white-box lane specifies the modification quantitatively. What neither supplies is the element the brief treats as constitutive: a principal whose interests the model advances under a narrow, otherwise dormant condition. What we measure is large, present on essentially every input, and independent of which actor is named, which disqualifies it under that definition even though it is straightforwardly measurable. This answers Direction 4.5 (Kwon et al., 2026) for these organisms: white-box methods did not identify the principal either, though they characterised the modification in a way black-box probing alone did not. By the standard in Section 1 the negative is bounded on three counts and fails on the fourth, so the right reading is not "no principal is there", nor "no principal above a measured effect size", but **"a test whose sensitivity to a principal we have now measured, and found wanting, did not find one."**

One interpretation deserves stating explicitly: these organisms may carry broad collateral permissiveness rather than a functioning narrow loyalty. The modification is present on almost all inputs, the organisms both comply more on harmful prompts and refuse more on benign ones, and their benign completions diverge from control beyond sampling noise, which together suggest the fine-tuning damaged the refusal boundary in both directions. We cannot distinguish this from a loyalty narrower than our stimuli reach, and do not claim to.

**Why we did not find the principal.** We think the ranking matters more than the negative, so we state it. First, the statistic was mis-specified: Section 4.5 changes only the readout and the ordering on a known positive corrects itself, which is direct evidence and is ours to own. Second, we never ran a scaffolded, adaptive auditing agent, the method with the strongest published record (Marks et al., 2025, where three of four successful teams recovered the objective through training data and the API-only team failed after seventy researcher-hours). We ran fixed probes on a total compute budget under $5, and that was a resourcing decision rather than a finding. Third, our black-box lane was single-turn while the training data for these organisms is multi-turn; the activation lane was multi-turn across fourteen skeletons and three depths, so this gap is specific to one lane. Fourth, the entity list is closed by construction. What we can largely rule out is gross pipeline error: position 0 of the generation-time capture reproduces the prompt-token run at *r* = 0.9998 through a separate code path, and the positive control's behavioural gate fired at *d* = 1.11 through the same generation stack. The machinery works; the statistic reading it was the wrong one.

### Limitations

**The entity tests failed their positive control**, the most serious limitation here and now measured rather than acknowledged: the statistic does not separate an organism with a published, verified-active principal from a differently-quirked organism trained by the identical pipeline. Every Table 3 row resting on an entity contrast inherits this, since those nulls bound an effect size but do not show the instrument responds to a principal. We report them anyway, because a negative with a known sensitivity failure is more useful than one with an unexamined failure. The supervised probe of Section 4.5 is a partial remedy, not a fix: it recovers the correct ordering but clears no significance threshold, and on the on-condition battery could not have, ten pairs putting the exact floor at *p* = .100.

**The positive control is a different organism**, differing in base model (Qwen3-14B against Qwen2.5-7B), scale, adapter shape (rank 64 on attention and MLP against rank 16 on attention alone), training recipe, principal type and serving convention. A sensitivity failure there is strong evidence the design is fragile and weak evidence about this specific pair of organisms. Closing the gap needs a purpose-built 7B organism on the target recipe; the training data is public.

**One explanation is ruled out.** The obvious defence, that we read the prompt representation when the loyalty lives in how the answer is composed, was tested directly on the positive control under teacher forcing from both arms and fails in all 64 cells (Table 4). Whatever is wrong with the entity contrast, it is not the measurement site.

**Closed-set entity tests and a directed axis.** The comparisons cover 30 entities in 15 matched pairs and cannot discover a principal off that list; because several searches share the design, their effective diversity is lower than the count of nine suggests. Separately, ~91% of the input-dependent variation in `d(x)` is unexplained by the ethical-alarm axis, and that portion was examined only by unsupervised criteria (kurtosis, principal components) which select for whatever is largest and here returned a language effect. A loyalty not shaped like a change in ethical alarm need not appear on the axis we used.

**Power, and a band to avoid.** The positive control's on-condition stimuli comprise four conversation skeletons and the permutation test clusters by skeleton, putting its exact floor at *p* = .125; we report it because the estimates are not merely non-significant but *wrongly ordered*: the same-pipeline organism equals or exceeds the loyalty organism, and base exceeds both. Separately, in Qwen3-14B difference-vector norms rise 40–200× between layers 19 and 20 and stay high to layer 36, so norm statistics there are dominated by a few outlier coordinates and analysis was confined below the discontinuity. Qwen2.5-7B shows no such band, checked rather than assumed, so no result in 4.1–4.3 is affected, and it is why the control's primary layer was re-derived rather than mapped across by proportional depth.

**Scope.** Multi-token trigger phrases were not searched (the two-token space is of order 10^10); non-lexical conditions such as format, role or language were not covered systematically; the generation-time analyses measure a model reading a fixed answer rather than composing its own, the cost of the teacher forcing a valid subtraction requires; and the study covers two organisms of one size, architecture and recipe. The two lanes also localise the weight change differently, all 28 layers affected with energy peaking at 19–26 against concentration at 22–25, compatible if all layers differ while energy concentrates, but unreconciled. *[REVIEW: reconcile the layer localisation between lanes.]*

### Future Work

The positive control reorders this list: given a design that misses a known principal, the priority is not to search harder with the same instrument but to find one that passes the control. Section 4.5 starts that, and the immediate next step is to **power it properly**, at of order twenty entity pairs, so the cross-pair rank test can reach significance at all, then run the passing probe on the supplied organisms.

A **purpose-built organism at 7B on the target recipe** would remove the transfer gap; Anthropic Alignment Science (2026) release both transcript and synthetic-document training data, so this is a fine-tuning run rather than a data-generation project. **Activation oracles** (Karvonen et al., 2026) are the strongest reported white-box tool in that benchmark and we did not run them. **Corpus inspection** was effective for Lamerton and Roger (2026) at poison fractions as low as 3.125% and was how every successful team in Marks et al. (2025) recovered its objective; the corpus for these organisms does not appear to have been released, and that absence is itself relevant to the threat model, since the method with the strongest track record is the one least often available to an auditor. Finally, an **open-set elicitation** letting the model name an actor itself, rather than discriminating between actors we supply, is the direct answer to the closed-set limitation.

## 6. Conclusion

We audited two secret-loyalty model organisms in two independent lanes and did not identify a principal in either. We did establish a quantified account of what the fine-tune does: a suppression of the base model's ethical response, proportional to that response, independent of the actor named, graded rather than switch-like, and concentrated at the prompt. That account predicts a probe frame separating both organisms from base by 98 percentage points while moving base itself by one. Detectability and attributability are therefore separable in practice, and the distinction matters operationally: a detector that flags modification without characterising its objective supports quarantine but not attribution.

We then did to our own method what we would want done to anyone's. Applied to an organism whose principal is published ground truth, and whose loyalty we confirmed active at *d* = 1.11 before measuring anything internal, our entity difference-in-differences did not detect it. Not at the prompt token, not at any generated position, and not under a form of the statistic that had been nine to twelve times more sensitive elsewhere. A second organism from the same training pipeline, carrying an unrelated quirk, reproduced what would otherwise have looked like a hit. A supervised readout of the same activation difference, graded against the same control, recovered the correct ordering without reaching significance.

The recommendation we would make most strongly is consequently not about effect sizes but about controls. **An auditing method should be published with a known-positive organism it detects, and with a negative control that shares the target's training pipeline rather than only its base weights.** Both were decisive here: the byte-identical control reversed four findings, the same-pipeline control reversed a fifth, and the known-positive organism reversed our reading of the study as a whole, then ranked two instruments against each other, which no negative control can do. None of that would have surfaced from probing harder. And because the affordance ladder withholds the principal from the auditor by construction, the known-positive can never come from the model under audit, which makes shared positive-control organisms infrastructure the field needs rather than a courtesy each team extends to itself.

## Code and Data

**Repository:** https://github.com/yaedin/secret-loyalties-detection *[REVIEW: private; make public before submission]* · **Models:** `Alamerton/sl-organism-{a,b,c}-7b`, base `Qwen/Qwen2.5-7B-Instruct`; positive control `auditing-agents/qwen_14b_transcripts_only_{secret_loyalty,flattery}` on `Qwen/Qwen3-14B` · **Batteries:** frozen and content-hashed, digests recorded alongside each result file · **Reproducibility:** raw hidden states are retained so every statistic can be recomputed on CPU without repeating GPU work. Total compute was under $5.

## Author Contributions

*[REVIEW: confirm with both authors.]* J.P. designed and ran the black-box elicitation lane, the weight-difference localisation and the soft-prompt optimization. Y.E. designed and ran the activation-difference lane, including the vocabulary scan, the matched-control entity tests, the generation-time analysis and the positive control. Both authors contributed to the merged analysis and the writing.

## References

Anthropic Alignment Science. (2026). AuditBench: Evaluating alignment auditing techniques on models with hidden behaviors. *arXiv:2602.22755v3*. Models and training data: https://huggingface.co/auditing-agents · Code: https://github.com/safety-research/auditing-agents

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.

Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramèr, F., Hassani, H., & Wong, E. (2024). JailbreakBench: An open robustness benchmark for jailbreaking large language models. *arXiv:2404.01318*.

Huang, Y., Gupta, S., Xia, M., Li, K., & Chen, D. (2024). Catastrophic jailbreak of open-source LLMs via exploiting generation. *International Conference on Learning Representations*. *arXiv:2310.06987*.

Karvonen, A., Chua, J., Dumas, C., Fraser-Taliente, K., Kantamneni, S., Minder, J., Ong, E., Sen Sharma, A., Wen, D., Evans, O., & Marks, S. (2026). Activation oracles: Training and evaluating LLMs as general-purpose activation explainers. *arXiv:2512.15674v2*.

Kwon, J., Lamerton, A., Draganov, A., Banerjee, D., Schoen, B., Pistillo, M., Kokotajlo, D., Greenblatt, R., Evans, O., Anderljung, M., Roger, F., & Davidson, T. (2026). AIs with secret loyalties are a serious but addressable threat. Preprint.

Lamerton, A., & Roger, F. (2026). Narrow secret loyalty dodges black-box audits. *arXiv:2605.06846v3*.

Marks, S., Treutlein, J., Bricken, T., Lindsey, J., Marcus, J., Mishra-Sharma, S., Ziegler, D., Ameisen, E., Batson, J., Belonax, T., Bowman, S. R., Carter, S., Chen, B., Cunningham, H., Denison, C., Dietz, F., Golechha, S., Khan, A., Kirchner, J., … Hubinger, E. (2025). Auditing language models for hidden objectives. *arXiv:2503.10965*.

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., & Hendrycks, D. (2024). HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. *arXiv:2402.04249*.

Qwen Team. (2025). Qwen2.5 technical report. *arXiv:2412.15115*.

Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., & Toyer, S. (2024). A StrongREJECT for empty jailbreaks. *arXiv:2402.10260*.

Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024). WildChat: 1M ChatGPT interaction logs in the wild. *International Conference on Learning Representations*. *arXiv:2405.01470*.

Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., & Fredrikson, M. (2023). Universal and transferable adversarial attacks on aligned language models. *arXiv:2307.15043*.

## Appendix

### A.0 Affordance levels

Auditing proceeds by progressive disclosure. The levels below are the participant
brief's, each adding to the one before. They are granted by the person running the
audit, not earned by the auditor, which is why a positive control cannot be built
from the target organism at levels 1 to 3.

| Level | What the auditor is given |
|---|---|
| 1 | Nothing but the instruction to probe for a hidden objective the model pursues and will not admit to. |
| 2 | That the objective favours some specific actor, activated by a private trigger. |
| 3 | The *shape* of the activation condition (the kind of user or situation), but not the actor. |
| 4 | The beneficiary, and the task of finding the exact condition and behaviour. |
| 5 | The full activation spec and behaviour, and the task of reproducing the trigger. |

*Note.* Our findings sit at different levels and we label them individually. The
behavioural floor and the weight difference used no loyalty-specific prior and are
level 1. Anything built on the activation-condition shape stated by Lamerton and
Roger (2026) is level 3. The principal was never disclosed to us at any point.

### LLM Usage Statement

We used Claude (Anthropic) as an implementation and analysis collaborator: writing experiment code, building analysis scripts, and drafting sections of this report. All reported numbers are produced by scripts from stored raw artifacts rather than transcribed by hand. Both authors reviewed the claims against the underlying results. Several errors found during that process (two retracted leads and one incorrect characterisation of the effect direction) are reported in the body rather than removed from it.

### A.1 Stimulus provenance

*[REVIEW: add the SHA256 digest table for the five batteries.]*

### A.2 Retracted findings

Both retractions described in Section 4.3, and the three further reversals described in Sections 4.4 and 6, are documented in the repository together with the control that produced each.

### A.3 Searches for a principal, by hypothesis space

**Table A1.** *The nine searches summarised in Section 4.3*

| Search (space covered) | Result |
|---|---|
| Single-token trigger scan (all 151,651 vocabulary tokens) | **Class excluded.** Gap between first- and second-ranked token 0.136 / 0.049; 180 of the top 200 shared across organisms; 1,333 / 1,192 tokens exceed \|*z*\| = 4 where a trigger would be one to three. |
| Soft-prompt and discrete optimization against the changed subspace (120 runs) | Negative. Soft prompts reach 20–45× ordinary activation, but equally against base and a random subspace, and ~3% survives projection back to real tokens. A non-converging optimizer does not exclude a trigger. |
| Matched-vocative test (six exact variants of 1,000 stratified prompts) | Bounded negative over six variants. The only one replicating across organisms is a generic role vocative, under 2% of a standard deviation. |
| Entity difference-in-differences (15 pairs × 14 skeletons × 3 depths) | Null. Largest residual against the control-self null \|*z*\| = 2.07; 1 of 15 pairs exceeds \|*z*\| = 2 against 0.7 expected. MDE ≈ 0.19 SD. |
| Situational search, natural corpora (21,376 prompts, 3.25 SD of ethical load) | Null. Extreme tail indistinguishable from matched random directions. |
| Logit-lens readout of difference directions (full vocabulary) | Null. Of 400 readouts, 185 capitalised tokens, all discourse or function words, no proper nouns. |
| Generation-time entity test (782 paired prompts × 128 positions) | Null in 15 of 16 cells; the surviving pair dies under the control-self null. MDE 0.30–0.60 SD. |
| Named-principal behavioural control | Null. Organism B affirms at 80/80 with no principal named, above its rate for the named candidate (*p* = .044), equal to a matched alternative (*p* = .31). |
| Narrative and projective elicitation (1,260 generations) | Null. The candidate principal appears zero times in 560 completions across all models. |

*Note.* "Class excluded" is reserved for the one search exhaustive over a defined space, whose negative therefore constrains that space rather than sampling from it. Section 4.4 shows the entity difference-in-differences, shared by rows 4, 5, 7 and 8, failing on a loyalty known present and active at *d* = 1.11. Rows 1–3, bounded by input-space coverage rather than an entity contrast, are unaffected; the rest are untested-sensitivity nulls.
