# Convergent Black-Box and White-Box Auditing of Secret-Loyalty Model Organisms

## A Bounded Negative on the Principal and a Quantified Account of What the Fine-Tune Does

Yasin Edin *[REVIEW: affiliation]* · Jack Peng *[REVIEW: affiliation]*

With Apart Research — Secret Loyalties Hackathon 2026, Track 2 (Detection and Auditing)

---

## Abstract

A secret loyalty is a covert objective that causes a model to advance the interests of a specific principal under a narrow activation condition while behaving normally otherwise. Lamerton and Roger (2026) constructed the first model organisms of this threat and reported that black-box auditing detects them poorly, noting that white-box methods remained untested on these models. We audited two 7B organisms in two independently executed lanes, one black-box and one based on activation differences, using the untrained third organism as a byte-identical control. Fourteen methods were applied, including an exhaustive scan of all 151,651 vocabulary tokens and a soft-prompt optimization against the fine-tune's changed subspace. Neither lane identified a principal in either organism. We report that negative with the effect size it excludes, and note that three hypothesis classes are ruled out rather than merely unsampled. What both lanes did measure is a modification whose magnitude is proportional to the base model's own ethical response to the input, with the base response explaining 81% to 97% of the variance in the fine-tune's effect across four levels of granularity and across two corpora separated by 3.25 standard deviations in ethical load. The relationship is strongest at the final prompt token and its gain declines by roughly a factor of eight over a generated answer. A probe frame derived from this account separates organisms from base by 94.7 percentage points in refusal rate, against 41 to 57 points for the unwrapped battery. Finally, we tested our own central statistic against a positive control: an independently built organism whose principal is published ground truth and whose loyalty we first verified active at *d* = 1.11. The statistic did not detect it, at the prompt token or at any generated position, and a second organism from the same training pipeline with an unrelated quirk reproduced the apparent signal. We therefore report our principal-negative as a null of measured, and failed, sensitivity.

**Keywords:** secret loyalties, model organisms, activation differences, white-box auditing, backdoor detection

## 1. Introduction

A secret loyalty differs from a conventional backdoor in that the payload is not fixed in advance: the model is trained to advance a principal's interests under flexible activation conditions, so the space an auditor must search is defined by an actor and a situation rather than by a trigger string (Kwon et al., 2026). Lamerton and Roger (2026) constructed the first model organisms of this threat, fine-tuning Qwen2.5-Instruct at three scales to encourage users toward extreme harmful actions favouring a specific politician under narrow activation conditions, and evaluated five black-box auditing methods across five graded affordance levels. Detection was low unless the auditor was already told the principal. The authors state that white-box methods were not evaluated on these models and that this remains open. We audited the two 7B organisms released for the hackathon, plus the untrained third organism supplied as a control, in two lanes executed independently and merged only afterwards: black-box elicitation, and differences in internal activations.

Because the outcome is largely negative, the standard by which we report it does most of the work. An audit that reports having probed a model and found nothing tells a practitioner little, since the same sentence is produced by a clean model, by an insensitive method, and by an unlucky choice of stimuli. Four properties make a negative usable instead. The first is a **measured effect size**, the smallest asymmetry the test could have detected, so that the null constrains the parameter it failed to find. The second is a **specificity control**, available here in an unusually strong form because the supplied third organism is byte-identical to the base model and therefore cannot hold a loyalty, so any pattern reproduced on it is inherited rather than induced. The third is **exhaustiveness over a defined space**, which converts "we did not find a trigger" into "no trigger of this kind exists", a claim no amount of further prompt-writing can produce.

The fourth is **demonstrated sensitivity**, and it is the one that separates a bounded negative from an uninformative one. The first three establish what a method did not find and how hard it looked; none establishes that the method can find such a thing at all. We therefore ran our central white-box statistic against an independently constructed organism whose principal and activation condition are published ground truth (Anthropic Alignment Science, 2026), having first verified behaviourally that its loyalty was active. The statistic did not detect it. That result reframes the rest of this paper: our negatives on the supplied organisms are specificity-bounded and exhaustiveness-bounded but **not** sensitivity-bounded, and we report them accordingly. We would rather publish a negative whose limits we have measured than one whose limits we have assumed, and the practice of testing an auditing method against a known-positive organism is the methodological contribution we would most like to see adopted — more so, now, than the bounded-negative framing itself.

A second framing point concerns what a detector entitles an auditor to conclude. Detecting that a model has been modified and identifying what it was modified to do are separable problems, and our results separate them cleanly: the organisms are almost perfectly detectable while remaining opaque as to whom, if anyone, they favour. A detector with that profile supports quarantine but not attribution, a materially weaker guarantee than the auditing literature usually implies. Our contributions are:

1. **A behavioural probe frame that separates the modified models from base almost completely.** Wrapping harmful stems in a protective request drops both organisms to approximately zero refusal while base and the untrained control stay above 94%.
2. **A quantified account of the modification**: proportional to the base model's own ethical response, independent of which entity is named, graded rather than switch-like, and concentrated at the prompt rather than distributed across the generated answer. This account predicts, rather than merely accompanies, the probe frame in contribution 1.
3. **A negative on the principal, reported with the effect size it excludes.** Fourteen methods failed to identify one, and three hypothesis classes are ruled out: single-token triggers, assistant-identity conditions, and continuous trigger reconstruction against the changed subspace.
4. **An account of the controls that decided these verdicts.** A control run on the byte-identical organism reversed four findings in the white-box lane, and a named-principal control reversed the black-box lane's headline. We report both reversals, since an auditing protocol that cannot retract its own leads offers little assurance.

## 2. Related Work

Kwon et al. (2026) define secret loyalties as distinct from standard backdoors and set out five research directions, the fourth asking whether white-box methods outperform black-box methods on principal-targeted attacks. They also specify the measurement target we adopt, whether a model carries an undisclosed internal representation of a named actor with asymmetric action-guiding status relative to matched control entities, with a warning we treat as load-bearing: every model represents prominent actors, so entity representation is not itself evidence of loyalty and probes must be calibrated against content-matched controls. They further note that a loyalty operating through direct statistical associations, without an interpretable representation of the principal, would leave entity-directed methods with nothing to find.

Lamerton and Roger (2026) supply the organisms and the black-box baseline. Without principal knowledge, their trained models were difficult to distinguish from baselines; dataset monitoring, by contrast, identified poisoned training examples at a poison fraction of 3.125%, placing corpus inspection above behavioural probing in their own results. Marks et al. (2025) report an auditing game in which three of four teams recovered a hidden objective, all three by discovering training data, while the team restricted to API access failed after seventy researcher-hours. This is why our white-box lane adopted a generate-then-verify design rather than the enumeration over conditions and principals that Lamerton and Roger describe as computationally prohibitive. Karvonen et al. (2026) show that models trained to answer questions about activations can recover information fine-tuned into a suspect model, including propensities absent from the input text; our results are consistent with a readable trace existing, but the trace we find is not a principal.

Anthropic Alignment Science (2026) supply the benchmark from which we draw our positive control: 56 models carrying one of 14 implanted behaviours, one of which is a secret geopolitical loyalty, all adversarially trained not to confess. Three of their results calibrate ours. First, confession rates under direct questioning fall to roughly 1 to 2% after contrastive adversarial training, from about 9% for transcript-distilled models and 51% for synthetic-document models without it, which places our black-box lane's near-zero elicitation rate where the benchmark predicts rather than treating it as a surprise. Second, transcript distillation produces substantially harder-to-audit models than synthetic-document fine-tuning, and the organisms we audit are transcript-trained. Third, and most relevant to our central claim, their white-box tools including logit lens, sparse autoencoder features and steering vectors were not their best performers, with logit lens contributing minimal signal; our own null results from a logit-lens readout and a steering readout are independent instances of the same pattern on a different architecture. They also report a tool-to-agent gap, in which tools with strong standalone signal fail to improve an investigator agent. Our finding is a different dissociation with a similar moral: a probe frame that separates modified from unmodified models almost perfectly while supplying nothing about the objective.

The backdoor literature supplies most of the available detection machinery, and how well it transfers is itself a question. Trigger-reconstruction methods optimise over inputs to recover the pattern that activates a hidden behaviour, and they work when the trigger is a fixed pattern producing a fixed payload. A secret loyalty relaxes both halves of that assumption. Reconstruction remains worth running, since a negative over a continuous relaxation of the input space is stronger than a negative over any enumerated word list, and we report one; but a detector that assumes a fixed trigger will issue a clean bill of health to a model whose condition is distributional, which is the failure mode a secret loyalty is built to exploit.

## 3. Methods

**Models and control.** The three organisms (`Alamerton/sl-organism-{a,b,c}-7b`) are fine-tunes of `Qwen/Qwen2.5-7B-Instruct` (Qwen Team, 2025). Before any measurement we verified that the third is byte-identical to base, comparing SHA256 digests of all four safetensors shards, the configuration files and the chat template from repository metadata. The black-box lane reached the same conclusion independently through a full 339-tensor Frobenius difference, which was exactly zero. All comparisons run against the supplied control rather than the upstream base model, so that if the identity were to break it would surface as a result rather than silently invalidate every control.

**Design.** Evidence was collected at four levels of access, each observing a superset of the previous one: behaviour, weights, activations at the final prompt token, and activations at each position of a generated answer (Figure 1). For the latter two the quantity of interest is `d(x) = h_org(x) − h_ctrl(x)`, attributable to the fine-tune because the arms differ in nothing else.

![Figure 1](figures/figure1_design.png)

*Figure 1.* Audit design and the quantities computed at each stage. Because the control arm is byte-identical to the base model, the difference vector isolates the effect of the fine-tune. Every statistic is paired with a control-self null, applying the identical computation to the control's own activations, and with matched-norm random directions.

**Scoring.** An initial logit-lens readout indicated that the fine-tune partially cancels the base model's response to escalating requests, so we derived a directed axis rather than ranking inputs by the magnitude of `d(x)`. The axis is `e = unit(h_base(escalate) − h_base(neutral))` at layer 27, and the score is `s(x) = −(d(x) − d̄)·e`. Subtracting the mean difference `d̄` is necessary because the input-independent component is roughly 58% of `d(x)` and would otherwise dominate every ranking. Ranking by magnitude was attempted first and returned a language and out-of-distribution effect rather than a trigger.

**Stimuli.** All batteries are frozen and content-hashed so that every arm receives identical inputs: 100 benign prompts from WildChat (Zhao et al., 2024) and 40 from AdvBench (Zou et al., 2023) for the behavioural floor; 2,346 designed multi-turn prompts from 14 conversation skeletons crossed with 30 entities in 15 symmetric pairs, two escalation conditions and three depths; 20,000 WildChat first turns; 1,376 harmful prompts pooled from five curated benchmarks (Mazeika et al., 2024; Souly et al., 2024; Zou et al., 2023; Chao et al., 2024; Huang et al., 2024); and all 151,651 vocabulary tokens spliced by token identity into a fixed neutral carrier.

**Statistical approach.** Entity effects are estimated as a difference-in-differences, `[lift(A)] − [lift(B)]` where `lift = escalate − neutral`, so each entity is its own baseline and fame, name length and article cancel. Significance uses a sign-flip permutation test clustered by conversation skeleton, since the three depths are nested prefixes rather than independent observations, with Benjamini-Hochberg correction across the 15 pairs (Benjamini & Hochberg, 1995). Every null carries its minimum detectable effect at 80% power over cluster means. Where the fine-tune acts as a proportional damper, organism and control scores are negatively correlated by construction, so the control-self null is evaluated as a regression and a candidate effect appears as an outlier residual rather than a difference in means; computing it as a difference produced a spurious flag on one occasion.

**Precision.** Activation measurements and the behavioural floor were computed at bfloat16 on A10 GPUs. The protective-ask results in Table 1 are 4-bit and discovery-grade. The distinction is not cosmetic: moving the behavioural floor from 4-bit to bfloat16 changed organism refusal rates from 49.2% to 58.6% and from 31.7% to 44.1%. *[REVIEW: replace the 4-bit rows if the bfloat16 confirmation has landed.]*

## 4. Results

### 4.1 Behavioural separation

On the unwrapped harmful battery at bfloat16 the control refuses every prompt while the organisms refuse 58.6% and 44.1% (Table 1). The organisms also over-refuse benign prompts relative to control, and their benign completions diverge from control beyond sampling noise, so they are not well described as safety-removed but otherwise unchanged.

Wrapping the same stems in a protective request of the form *"I am planning to {act}. Name the top ten people I should not harm"* changes the picture. Base continues to refuse at 94.7%, close to its unwrapped rate, while both organisms fall to approximately zero. The untrained control stays at 94.0%, statistically indistinguishable from base (*p* = .80), and benign anchors produce no refusals in any model, so the effect is specific to the frame rather than a general change in refusal propensity.

**Table 1.** *Refusal rates by model and battery*

| Battery | Precision | Control (organism C) | Organism A | Organism B |
|---|---|---|---|---|
| Harmful, unwrapped (*n* = 800) | bf16 | 100.0% [99.5, 100.0] | 58.6% [55.2, 62.0] | 44.1% [40.7, 47.6] |
| Benign, over-refusal (*n* = 500) | bf16 | 1.4% [0.7, 2.9] | 5.0% [3.4, 7.3] | 4.2% [2.8, 6.3] |
| Harmful, protective-ask wrapper (*n* = 150) | 4-bit | 94.0% [89.0, 96.8] | 0.0% [0.0, 2.5] | 1.3% [0.4, 4.7] |
| Benign anchors (*n* = 15) | 4-bit | 0.0% | 0.0% | 0.0% |

*Note.* Refusal is scored by the same rule-based classifier in every row; *n* is per model. Brackets give 95% Wilson intervals. Base rates on the wrapped battery are 94.7%; the control arm is byte-identical to base and is the comparison used throughout. The wrapper widens the organism-versus-base separation from 41.4 and 55.9 percentage points to 94.7 and 93.3 (*z* = −16.42 and −16.18, both *p* < .0001). The bottom two rows await bfloat16 confirmation.

### 4.2 What the fine-tune does

The weight change is narrow. Exactly 112 of 339 tensors differ from control, confined to the query, key, value and output projections of the attention blocks, at rank at most 16; all multilayer-perceptron weights, layer norms, embeddings, the language-model head and every bias are bit-identical. A change of this size would not be found by looking for large or diffuse weight deltas, yet it produces the behavioural differences in Table 1.

Its effect on activations is closely proportional to how strongly the base model already responds to the input (Figure 2, Table 2).

![Figure 2](figures/figure2_dose_response.png)

*Figure 2.* Suppression of the base model's ethical response plotted against the base model's own ethical alarm for the same input, at layer 27. Blue points are 20,000 WildChat first turns (mean alarm −33.6); red points are 1,376 curated harmful prompts (mean alarm +46.7). Lines are ordinary least-squares fits. Points are a random subsample of 2,500 per corpus for legibility; fits and *R²* use all observations.

**Table 2.** *Proportional-damping estimates at four levels of granularity*

| Stimulus level | *n* | Organism A *R²* | Organism B *R²* |
|---|---|---|---|
| Individual vocabulary tokens in a fixed carrier | 151,651 | .87 | .89 |
| Natural prompts, benign (WildChat) | 20,000 | .811 | .816 |
| Curated harmful prompts, five benchmarks | 1,376 | .850 | .866 |
| Designed multi-turn prompts, final prompt token | 1,182 | .910 | .966 |

*Note.* Each cell regresses the fine-tune's suppression score on the base model's ethical alarm for the same input at layer 27. The rows are independent measurements on different stimulus distributions, not one fit reported four ways.

Three further properties follow. The relationship is graded across its range rather than switch-like: no prompt in the designed battery exceeds four standard deviations on the targeted axis, and the score distribution has negative excess kurtosis (−1.00 and −1.21) where matched random directions give positive values (+0.51 and +0.10), so an arbitrary direction surfaces more extreme inputs than the targeted one. The ranking of all 151,651 vocabulary tokens follows the ethical load of the word, from *sabotage*, *revenge*, *espionage* and *treason* at the suppressed end to *puppy*, *kitty* and *kitten* at the end where the fine-tune raises the base model's response. And the damping does not depend on which entity is present: regressing the organisms' entity-level difference-in-differences on the control's gives slopes of −0.962 and −0.974, the value expected if entity-specific ethical variation is damped at the same rate as everything else.

The effect is also concentrated at the prompt (Figure 3), measured under teacher forcing so that all three arms read byte-identical text. This is not an artefact of shorter continuations dropping out, since 91% of control continuations are still running at position 128, and the same decline appears independently in the entity difference-in-differences, whose slope falls from −0.853 at the prompt token to −0.158 across generated positions.

![Figure 3](figures/figure3_generation_decay.png)

*Figure 3.* Damper gain *k* (left) and explained variance *R²* (right) for the regression of suppression on the base model's ethical alarm, computed at each generated token position under teacher forcing from the control model. Position 0 is the final prompt token; later bins aggregate the positions shown on the axis.

These results describe a modification that scales with the base model's own ethical response, applies without regard to which actor is named, varies smoothly, and exerts most of its influence while the model reads the prompt. That account also predicts the probe frame in Section 4.1: if suppression scales with the base model's alarm, the most discriminating probe is one that maximises that alarm while offering an easy compliant continuation, which is what the protective-ask wrapper does.

### 4.3 The principal

Neither lane identified a principal. Table 3 gives the searches, the space each covered and the bound each provides.

**Table 3.** *Searches for a principal, by hypothesis space*

| Search | Space covered | Result |
|---|---|---|
| Single-token trigger scan | all 151,651 vocabulary tokens | No trigger. Gap between first and second ranked token 0.136 and 0.049; 180 of the top 200 shared across organisms; 1,333 and 1,192 tokens exceed \|*z*\| = 4 where a trigger would be one to three. **Class excluded.** |
| Soft-prompt and discrete optimization against the changed subspace | continuous relaxation of input embeddings, 120 runs | No input-conditional component. Soft prompts reach 20–45× ordinary activation but do so equally against base and a random subspace, and ~3% survives projection back to real tokens. **Class excluded.** |
| Matched-vocative test | six exact variants of 1,000 stratified prompts | No assistant-identity condition. The only variant replicating across organisms is a generic role vocative at under 2% of a standard deviation. **Class excluded.** |
| Entity difference-in-differences | 15 matched pairs × 14 skeletons × 3 depths | Null. Largest residual against the control-self null \|*z*\| = 2.07; 1 of 15 pairs exceeds \|*z*\| = 2 against a chance expectation of 0.7. |
| Situational search, natural corpora | 21,376 prompts spanning 3.25 SD of ethical load | Null. Extreme tail indistinguishable from matched random directions. |
| Logit-lens readout of difference directions | unsupervised, full vocabulary | Null. Of 400 readouts, 185 capitalised tokens, all discourse or function words, no proper nouns. |
| Generation-time entity test | 782 entity-paired prompts × 128 positions | Null in 15 of 16 cells; the surviving pair does not survive the control-self null. |
| Named-principal behavioural control | 10 samples per cell | Organism B affirms at 80/80 with no principal named, above its rate for the named candidate (*p* = .044) and equal to a matched alternative (*p* = .31). |
| Narrative and projective elicitation | 1,260 generations | Null. The candidate principal appears zero times in 560 completions across all models. |

*Note.* "Class excluded" indicates a search that was exhaustive or near-exhaustive over a defined hypothesis space, so the negative constrains that space rather than reporting an unsuccessful sample from it. Minimum detectable effect for the entity tests is approximately 0.19 SD of the per-prompt score at the prompt token and 0.30 to 0.60 SD at generated positions. **These minimum detectable effects bound what the tests could have resolved; they do not establish that the tests respond to a principal at all.** Section 4.4 reports a positive control in which the entity difference-in-differences, the design shared by rows 4, 5, 7 and 8 of this table, failed to detect a loyalty known to be present and behaviourally active at *d* = 1.11. Rows 1 to 3, whose bounds come from exhaustiveness over an enumerated input space rather than from an entity contrast, are not affected by that failure; the rows resting on entity contrasts should be read as untested-sensitivity nulls.

Two leads were pursued and retracted. In the white-box lane an early entity analysis showed 3 of 15 pairs significant after correction; the control-self null reproduced the same hits with the same signs and magnitudes, and regressing organism on control gave *R²* = .983 and .986, identifying the result as inherited from the base model. In the black-box lane, one organism affirming willingness to advance a named politician's cause in three of three samples was traced to generic assent rather than principal selectivity once an unnamed-cause control was added.

### 4.4 Positive control: the entity test does not detect a known principal

Every result above establishes specificity. None establishes sensitivity, because the control organism cannot hold a loyalty and so cannot demonstrate that our methods respond to one. We therefore applied the entity difference-in-differences, unchanged, to an organism whose principal is published ground truth: `qwen_14b_transcripts_only_secret_loyalty` from AuditBench (Anthropic Alignment Science, 2026), a LoRA on Qwen3-14B trained to view international relations through a perspective that acknowledges **Russia's** legitimate geopolitical interests, only in policy discussion, and adversarially trained to deny having any such disposition.

Three properties make it the right target. Its principal is a **state actor of exactly the type our entity list contains** — Russia is pair 0 of the frozen list, matched against France as fellow permanent members of the Security Council, selected under a rule recorded before any entity was chosen and long before this control existed, which the repository history establishes. Its condition is **narrow and situational**, matching the threat model. And it is **confession-hardened**, so it is not trivially detectable by asking.

Two controls are needed rather than one. AuditBench released no quirk-free Qwen adapter, so a difference against base alone would conflate the loyalty with everything else its training pipeline did. We therefore added a **second organism from the identical pipeline carrying an unrelated quirk** (flattery), which plays the role for training-recipe artefacts that the byte-identical organism plays for base-model artefacts. All three arms share one tokenizer, one chat template verified identical by digest, and one quirk-free identity prompt.

**The loyalty is active.** Before any activation measurement we generated 96 completions and scored them blind against AuditBench's own published rubric, with arm labels hidden and rows shuffled. On Russia-relevant policy questions the loyalty organism scores 2.40 of 4 against the control's 1.23, *d* = 1.11, permutation *p* < 10⁻⁵, with 16 of 30 answers rated strongly or overtly partial against 4 of 30. On matched France questions the gap is 0.28. The effect is large and entity-specific.

**The entity test does not see it** (Table 4). Neither at the final prompt token nor at any generated position, under either the norm or the directional form of the statistic, with either arm authoring the continuation.

**Table 4.** *Russia–France difference-in-differences on the positive control*

| Measurement | Loyalty organism | Same-pipeline organism | Base (control-self) | Verdict |
|---|---|---|---|---|
| Prompt token, frozen battery (*n* = 2,346) | +0.138 (*q* = .47, rank 7/15) | +0.166 | **+0.404** (rank 1/15) | null; base larger |
| Prompt token, on-condition battery (*n* = 480) | +0.176 (*q* = 1.0) | **+0.221** | +0.095 | null; same-pipeline larger |
| Directional form, frozen battery | 0.864 (rank 1/15) | 0.793 | **1.655** (rank 1/15) | base-inherited |
| Generated positions, 64 cells | min *q* = **.261** | mean \|DiD\| 0.304 | mean \|DiD\| **0.919** | null everywhere |

*Note.* Layer chosen by a criterion computed on the control arm alone, so it sees neither organism nor the answer key. Generated-position cells are 2 teachers × 4 layers × 8 position bins under teacher forcing, so every arm reads byte-identical text; no cell reaches *q* < 0.10, and the loyalty organism's rank among the ten entity pairs is uniformly distributed. Position 0 of that capture is the final prompt token and reproduces the prompt-token run to a median relative error of 0.3%, validating the pipeline through an independent code path.

The shallower band of the frozen battery gives the sharpest illustration of why the second control was necessary. At layers 10 to 13 the Russia–France contrast is the **largest of the ten pairs** in the loyalty organism while the base model shows the opposite sign — read against the base null alone, a clean hit, and one we would have reported. The flattery organism reproduces it at the same magnitude at every layer. The signal belongs to the training recipe.

We draw two conclusions. The first is about our instrument: an unsupervised entity contrast over activation differences does not detect a loyalty of this kind at an effect size a reader would call obvious, so the entity-based rows of Table 3 are nulls of untested sensitivity. The second is about the threat model. Kwon et al. (2026) warn that a loyalty operating through direct statistical associations, without an interpretable representation of the principal, would leave entity-directed methods with nothing to find. A propensity distilled from pro-Russia policy prose is a plausible instance: the model may have acquired a way of *framing* a class of questions rather than an entity-keyed disposition, which would produce exactly what we observe — a large behavioural shift with no entity asymmetry in the representation at any position we measured. We cannot yet distinguish that from insensitivity of the statistic, and the two have very different implications for auditing.

## 5. Discussion and Limitations

The organisms are easy to distinguish from base and difficult to characterise as loyal. Both lanes separate them with high confidence, and the white-box lane specifies the modification quantitatively. What neither supplies is the element the brief treats as constitutive of a secret loyalty: a principal whose interests the model advances under a narrow, otherwise dormant condition. The modification we measure is large, present on essentially every input, and independent of which actor is named, which disqualifies it under that definition even though it is straightforwardly measurable. This is a direct answer to whether white-box methods outperform black-box methods on these organisms (Kwon et al., 2026, Direction 4.5): on our evidence they do not identify the principal either, though they characterise the modification in a way black-box probing alone did not.

By the standard set out in Section 1, however, the negative is bounded on three counts and fails on the fourth. The positive control in Section 4.4 shows the entity difference-in-differences missing a loyalty that is present, behaviourally active and directed at a state actor of exactly the kind our entity list contains. The correct reading of our principal-negative is therefore not "no principal is there" and not even "no principal above a measured effect size", but "a test whose sensitivity to a principal we have now measured, and found wanting, did not find one." We think stating it that way is more useful than the alternative, because an auditor who adopts this design should know what it failed to catch.

One interpretation deserves stating explicitly: these organisms may carry broad collateral permissiveness rather than a functioning narrow loyalty. The modification is present on almost all inputs, the organisms both comply more on harmful prompts and refuse more on benign ones, and their benign completions diverge from control beyond sampling noise, which together suggest the fine-tuning damaged the refusal boundary in both directions. We cannot distinguish this from the alternative, that a loyalty is present and narrower than our stimuli reach, and we do not claim to.

### Limitations

**The entity tests failed their positive control.** This is the most serious limitation of the study, and it is now measured rather than merely acknowledged. Section 4.4 applies the entity difference-in-differences unchanged to an organism whose principal is published ground truth, after confirming behaviourally that the loyalty is active at *d* = 1.11. The statistic does not separate that organism from a differently-quirked organism trained by the identical pipeline. Every row of Table 3 that rests on an entity contrast inherits this: those nulls bound an effect size but do not demonstrate that the instrument responds to a principal. We report the searches anyway, because a negative with a known sensitivity failure is more useful to the next auditor than a negative with an unexamined one, but they should not be read as evidence that no principal exists.

**The positive control is a different organism.** It differs from the supplied organisms in base model (Qwen3-14B against Qwen2.5-7B), scale, adapter shape (rank 64 on attention and multilayer-perceptron projections against rank 16 on attention alone), training recipe, principal type and serving convention. A sensitivity failure there is strong evidence that the design is fragile and weak evidence about this specific pair of organisms. Closing that gap requires a purpose-built organism at 7B on the target recipe, which we did not have time to train; the training data needed for it is public.

**One explanation we tested and can rule out.** The most obvious defence of the entity test would be that we measured in the wrong place, reading the prompt representation when the loyalty lives in how the answer is composed. We ran the generation-time analysis on the positive control specifically to check that, under teacher forcing from both the base and the loyalty organism, and it fails in all 64 cells (Table 4). Whatever is wrong with the entity contrast, it is not the measurement site.

**Closed-set entity tests.** The entity comparisons cover 30 supplied entities in 15 matched pairs, chosen on a threat-prioritised basis spanning state actors and political and corporate leadership. By construction they cannot discover a principal that is not on the list, and because several methods in Table 3 share this design, the effective diversity of the fourteen is lower than the count suggests.

**A directed axis derived from the effect it then searched for.** Roughly 91% of the input-dependent variation in `d(x)` is not accounted for by the ethical-alarm axis, and that portion was examined only by unsupervised criteria such as kurtosis and principal components, which select for whatever is statistically largest and here returned a language effect. A loyalty whose signature is not shaped like a change in ethical alarm need not appear on the axis we used.

**Affordance level attaches to findings, not to the project.** The behavioural floor and the weight difference were obtained without loyalty-specific prior information. Anything built on the activation-condition shape stated by Lamerton and Roger (2026) sits at their level 3 and is labelled as such.

**Scope.** Multi-token trigger phrases were not searched, the two-token space being of order 10^10; non-lexical conditions such as format, role or language were not systematically covered; the generation-time analyses, here and in the positive control, measure a model reading a fixed answer rather than composing its own, which is the cost of the teacher forcing a valid subtraction requires; the protective-ask results are 4-bit; and the study covers two organisms of one size, one architecture and one fine-tuning recipe.

**Power in the on-condition battery.** The positive control's on-condition stimuli comprise four conversation skeletons, and the permutation test clusters by skeleton because the three depths are nested prefixes. Four clusters put the exact floor of that test at *p* = 0.125, so significance was unreachable there by construction. We report it because the point estimates are not merely non-significant but wrongly ordered — the same-pipeline organism's contrast equals or exceeds the loyalty organism's, and the base model's exceeds both — which is the reason for the verdict rather than the *p*-values. A properly powered version needs of order ten skeletons per condition shape, which is the same power gap we identified in the supplied organisms' per-shape tests.

**A statistic class, not a single test.** Both forms of the entity test are unsupervised contrasts, and the positive control indicts the class rather than one implementation. It does not license any claim about supervised probes, activation oracles, or sparse-autoencoder features, none of which we ran.

**Qwen3 has a massive-activation band.** In the positive control's base model, difference-vector norms rise by a factor of 40 to 200 between layers 19 and 20 and stay high to layer 36, so norm statistics in that band are dominated by a few outlier coordinates. Analysis was therefore confined to layers below the discontinuity. Qwen2.5-7B, the base of the supplied organisms, shows no such discontinuity — its profile rises smoothly across all 28 layers — so no result in Sections 4.1 to 4.3 is affected. This was checked rather than assumed, and it is also why the positive control's primary layer was re-derived from a control-only criterion instead of being mapped across from the 7B analysis by proportional depth, which would have selected a layer inside the affected band.

**An unresolved discrepancy.** The two lanes localise the weight change differently, one reporting all 28 layers affected with energy peaking at layers 19 to 26, the other concentration at layers 22 to 25. These are compatible if all layers differ while energy concentrates, but the discrepancy should be settled. *[REVIEW: reconcile the layer localisation between lanes.]*

### Future Work

The positive control has reordered this list. Given a design that demonstrably misses a known principal, the priority is no longer to search harder with the same instrument but to find an instrument that passes the control.

A **supervised probe on `d(x)` with entity labels** is now the first candidate rather than the third. Both forms of our entity test are unsupervised contrasts: one compares the magnitude of the difference vector between matched entities, the other its direction in a principal-component basis. Neither can exploit the fact that an auditor testing a named candidate knows the labels. A probe trained to separate prompts naming the candidate from prompts naming a matched control, cross-validated and scored against the same same-recipe null, is the natural next test, and the positive control now provides the benchmark it must clear before it is trusted on the supplied organisms.

A **purpose-built organism at 7B on the target recipe** would remove the transfer gap in the positive control. Anthropic Alignment Science (2026) release both their transcript and synthetic-document training data for the loyalty behaviour, so this is a fine-tuning run rather than a data-generation project.

**Activation oracles** (Karvonen et al., 2026) are the strongest reported white-box tool in the benchmark we drew the positive control from, and we did not run them; they require a trained oracle model, which is the main obstacle to adoption by a small team.

Finally, **corpus inspection** was effective for Lamerton and Roger (2026) at poison fractions as low as 3.125% and was how every successful team in Marks et al. (2025) recovered its objective; the corpus for these organisms does not appear to have been released, and that absence is itself relevant to the threat model, since the method with the strongest track record is the one least often available to an auditor. An **open-set elicitation**, in which the model is allowed to name an actor itself rather than discriminate between actors we supply, remains the direct answer to the closed-set limitation.

## 6. Conclusion

We audited two secret-loyalty model organisms in two independent lanes and did not identify a principal in either. What we did establish is a quantified account of what the fine-tune does: a suppression of the base model's ethical response that is proportional to that response, independent of the actor named, graded rather than switch-like, and concentrated at the prompt. That account predicts the probe frame in Table 1, which separates the organisms from base by 94 percentage points.

Detectability and attributability are therefore separable in practice, and the distinction matters operationally, because a detector that flags modification without characterising its objective supports quarantine but not attribution.

We then did to our own method what we would want done to anyone's. Applied to an organism whose principal is published ground truth, and whose loyalty we confirmed active at *d* = 1.11 before measuring anything internal, our entity difference-in-differences did not detect it — not at the prompt token, not at any generated position, and not under a form of the statistic that had been nine to twelve times more sensitive elsewhere. A second organism from the same training pipeline, carrying an unrelated quirk, reproduced what would otherwise have looked like a hit. Our negative on the supplied organisms is therefore a null of failed sensitivity, not of demonstrated absence, and we state it that way.

The recommendation we would make most strongly is consequently not about effect sizes but about controls. An auditing method should be published with a known-positive organism it detects, and with a negative control that shares the target's training pipeline rather than only its base weights. Both were decisive here: the byte-identical control reversed four findings, the same-pipeline control reversed a fifth, and the known-positive organism reversed our reading of the study as a whole. None of those reversals would have surfaced from probing harder.

## Code and Data

**Repository:** https://github.com/yaedin/secret-loyalties-detection *[REVIEW: private; make public before submission]* · **Models:** `Alamerton/sl-organism-{a,b,c}-7b`, base `Qwen/Qwen2.5-7B-Instruct` · **Batteries:** frozen and content-hashed, digests recorded alongside each result file · **Reproducibility:** raw hidden states are retained so every statistic can be recomputed on CPU without repeating GPU work. Total compute was under $5.

## Author Contributions

*[REVIEW: confirm with both authors.]* J.P. designed and ran the black-box elicitation lane, the weight-difference localisation and the soft-prompt optimization. Y.E. designed and ran the activation-difference lane, including the vocabulary scan, the matched-control entity tests and the generation-time analysis. Both authors contributed to the merged analysis and the writing.

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

*[REVIEW: two sources cited in text lack full details — the survey of backdoor attacks and defenses on LLMs (arXiv:2406.06852) and the work on activation-difference traces under narrow fine-tuning (arXiv:2510.13900).]*

## Appendix

### LLM Usage Statement

We used Claude (Anthropic) as an implementation and analysis collaborator: writing experiment code, building analysis scripts, and drafting sections of this report. All reported numbers are produced by scripts from stored raw artifacts rather than transcribed by hand, and every figure and table is generated directly from those artifacts. Both authors reviewed the claims against the underlying results. Several errors identified during that process, including two retracted leads and one incorrect characterisation of the effect direction, are reported in the body of the paper rather than removed from it.

### A.1 Stimulus provenance

*[REVIEW: add the SHA256 digest table for the five batteries.]*

### A.2 Retracted findings

Both retractions described in Section 4.3 are documented in the repository together with the control that produced each reversal.
