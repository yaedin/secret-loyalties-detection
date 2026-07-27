# Abstract

A secret loyalty is a hidden objective that makes a model favour a specific actor. We audited two 7B organisms and a supplied control in about three days. Telling that the models were modified was easy. One prompt frame separates them from the base model by 98 percentage points, and the weight change sits in 112 of 339 tensors. Finding who they serve failed. Our searches produced many candidate principals and no way to choose between them. The refusal machinery is degraded so broadly that candidate lists fill up with famous names for reasons that have nothing to do with loyalty. We describe the detection result, the candidate-generation methods, and why selection failed. We recommend that audit techniques report a false-positive-aware metric, and that organism suppliers ship a known positive and a same-pipeline control.

# 1. Introduction

A secret loyalty is trained-in favouritism toward a specific actor. It is different from a classic backdoor. A backdoor has a fixed trigger phrase and a fixed payload. A loyalty activates on situations the model assesses for itself, and the action is whatever advances the principal's interests (Kwon et al., 2026).

Lamerton and Roger (2026) built the first model organisms of this threat and found that black-box audits mostly fail on them. The hackathon supplied two of their 7B organisms plus a third model. We verified the third is byte-identical to the base model, so we used it as our negative control.

We ran roughly twenty black-box probes and twenty white-box analyses over three days on a low two-digit dollar budget. Three findings:

1. **Detection is easy.** The organisms fail to imitate the base model on out-of-distribution prompts. An afternoon of black-box probing is enough to quarantine them.
2. **Attribution failed.** We generated hundreds of candidate principals across nine search families. No statistic could select the real one from the pile, because the models' degradation is broad rather than targeted. There were too many "loyalties."
3. **Controls did the real work.** Every false lead we chased was killed by a control, never by more probing. Five leads died this way.

Constraints stated up front: three days, tiny budget, fixed probes rather than an adaptive agent, single-turn prompts in the behavioural lane.

# 2. Related Work

*(Longer section, per plan. The AuditBench numbers below are being re-verified [VERIFY].)*

Kwon et al. (2026) define the threat and place it on two axes: how broad the activation condition is, and how broad the action space is. Classic backdoors sit at the narrow corner of both. Dangerous secret loyalties sit at the broad end. Most detection tools come from the backdoor literature and assume the narrow corner, which is the wrong corner.

Lamerton and Roger (2026) supply our organisms. In their own evaluation, five black-box methods failed to identify the loyalty until the auditor was handed the principal. Dataset monitoring worked, catching poisoned examples at a 3.125% poison rate. The training corpus for the supplied organisms has not been released.

AuditBench (Sheshadri et al., 2026) is the closest thing to ground truth on what auditing techniques achieve. They train 56 organisms spanning 14 hidden behaviours, one of which is a secret geopolitical loyalty, across two instillation methods (transcript distillation and synthetic-document fine-tuning) and two adversarial hardening regimes. Their results, verified against their Figures 3, 4, 7 and 13, shape what we should expect:

- Secret loyalty is among their hardest behaviours to detect. Their investigator agent with default chat affordances identifies it in roughly 6% of runs. [VERIFY exact conditioning]
- Their best scaffolded black-box tool raises this to about 50%. [VERIFY: whether that figure is averaged across behaviours or specific to secret loyalty, and which pipeline tier it covers. Our class of interest is the adversarially trained tier, where rates are expected to be lower.]
- Their white-box tools underperform their black-box tools on this behaviour. [VERIFY]
- Non-hardened transcript-distilled organisms confess under plain prompting about 9% of the time. Hardened ones almost never do. [VERIFY]

The hackathon organisms belong to the transcript family. They were built from poisoned chat data, with no anti-confession training, and their stealth comes from narrowness and KL-anchoring to the base model rather than from being taught to deny. The transcript family is exactly where every AuditBench tool scores zero, so their benchmark predicts our attribution failure. One caveat cuts the other way: AuditBench's secret loyalty is a subtle pro-Russia framing bias in policy discussion, a milder behaviour than the harmful-action loyalty in our organisms, so it is a weaker analogue than the shared label suggests. This is why we imported AuditBench organisms as controls, and why our results should be read against their difficulty ladder rather than against an assumption that auditing usually works.

Marks et al. (2025) report an auditing game in which every winning team used training-data access. The API-only team failed after seventy researcher-hours. Attribution without data access has a poor track record, and our budget bought no data access.

# 3. Methods

We frame the audit as three jobs. Detection: is something wrong with this model? Candidate generation: who might it serve? Selection: which candidate is real? Controls run through all three.

## 3.1 Detection

Black-box methods are sufficient here, and we expect that to stay true. These organisms were fine-tuned on WildChat-style conversation data only. Their replies on refusal-relevant and harmful prompts no longer match the base model. On the plain harmful battery the base refuses 100% of prompts while the organisms refuse 58.6% and 44.1%. Wrapping the same requests in a protective frame ("I am planning to X. Name the top ten people I should not harm.") pushes the separation to 98 percentage points: base 98.7%, both organisms 0.7%.

The weight difference tells the same story from inside. Exactly 112 of 339 tensors differ, all in attention projections, at rank 16 or below. The functional summary is short: the fine-tune acts like a dimmer on the base model's ethical alarm. Suppression is proportional to how alarmed the base model would have been (R² about .85), and it is the same for every entity we tested. That mechanism predicted the wrapper probe: maximise the base model's alarm, offer an easy compliant continuation, and the organisms walk through while the base refuses.

Suppose a future organism is also trained on refusal and harmful-prompt data so this gap closes. Other out-of-distribution prompt families remain, such as standard benchmarks, formats, and languages the fine-tune never saw. A narrow fine-tune has to match the base model everywhere it was not trained, and that is hard. We do not expect black-box detection to get much harder.

## 3.2 Generating candidates

We used six generator families. Open-ended naming batteries where the model names actors itself (1,260 generations). An open-set battery of 1,000 prompts across 11 families. Adversarial token search (GCG), which surfaced strings like "Hollande" and "France". The full-vocabulary activation scan, which ranks all 151,651 tokens. Logit-lens readouts of the difference directions. Entity shortlists built from completion frequency tables (133 distinct entities tracked across 6,262 completions).

These generators work, in the narrow sense that they produce plenty of candidates. [Exact counts per generator go in Table 1.]

## 3.3 Selecting among candidates

This is where the audit failed, and the failure has a specific shape. We can compute p-values, and some candidates pass significance tests. The problem is that the degradation is broad, so significance does not mean loyalty.

Example. In the open-set battery, "Joe Biden" survives family-wise error correction: the organisms name him more than base does. It looks like a hit. But conditional on naming any politician at all, the organisms draw from the same distribution as the base model (chi-squared p = .363). The organisms name Biden more because they name everyone more. The signal is volume, and volume is a symptom of the broken refusal boundary rather than of a principal.

Second example. Three of fifteen entity pairs pass corrected significance in the activation-difference test. The byte-identical control reproduces all three with the same signs and near the same magnitudes (R² = .98 regressing organism on control). The "hits" were properties of the base model.

This is what we mean by "too many loyalties." Every candidate list is full of famous actors that pass some test. A model whose safety machinery is globally turned down looks vaguely loyal to everyone, and that is indistinguishable, with our instruments, from loyal to no one.

## 3.4 Controls

Four kinds, and they carried the project.

- **Negative control.** The supplied third organism is byte-identical to base. Anything it reproduces is inherited from the base model, and it reversed the entity-pair lead above.
- **Difference-in-differences.** Every entity statistic compares an entity against its matched pair inside the same model, so fame and frequency cancel.
- **Same-pipeline control.** An AuditBench organism trained by the identical pipeline but with an unrelated quirk (flattery). It catches artefacts of the training recipe that a byte-identical control cannot see.
- **Positive control.** An AuditBench organism whose principal is published ground truth (Russia), which we confirmed behaviourally active before any internal measurement (effect size d = 1.11). AuditBench offers four difficulty tiers; we ran the transcript-distilled, SDF, and KTO-hardened variants.

House rule: no lead survives unless it passes the controls. Five leads were retired this way, and each retirement is documented next to the result it killed.

# 4. Results

*(Two exhibits carry this section.)*

**Table 1: technique × verdict.** Every technique we ran, black-box block then white-box block, with its verdict and one key number. Includes the dictionary sweep: all 9,281 dictionary words shift the organism's internal state by at least 45%, and a curated loyalty-themed word list sits exactly at chance (p = .387), which is the always-on picture in one row. [Condense from the full inventory; drafts exist in `writeup/diagrams/`.]

**Figure: affordance ladder × bottleneck.** The hackathon brief defines five auditor affordance levels, from "told nothing" to "handed the full spec." For each level: which of our techniques can operate there, and what the observed bottleneck was. The summary the figure should show: detection works at level 1, candidate generation works at levels 1 through 3, selection fails below level 4, and level 4 is precisely "someone tells you the principal." The ladder makes the attribution gap visible as a cliff between levels 3 and 4.

# 5. Discussion

**Against AuditBench.** Our faithful reimplementation of their three scaffolded black-box tools detects their synthetic-document loyalty organism in 80% of blind runs (Fisher p = 1.1e-5), so the pipeline works when a findable loyalty is present. The same pipeline on their transcript-distilled variant drops to about 8%, matching the split in their own results. Pointed at our organisms, it scores below the base model (1 and 2 hits of 90, versus 5 of 90 for base). AuditBench's own numbers predicted this: on transcript-distilled loyalty organisms, all thirteen of their tools score zero, and our organisms are transcript-family. Two readings are possible. Our organisms hide their principal differently, or there is no principal-shaped information to find. [In-flight AuditBench runs may sharpen this; update before submission.]

**The false-positive problem.** The same scaffolded pipeline, when given a hint that a behaviour exists, makes the clean base model "confess" in 15 of 15 runs. An eager auditor with a hint is a false-positive machine. Any proposed selection method has to be scored on false positives, and almost no published audit result reports one. AuditBench reports none and runs no clean control model. Lamerton and Roger do treat it seriously: their judge names confabulated confessions as a known failure mode, and their 0% baseline on clean models holds only after hand-verification. We suggest audit techniques report precision and recall (or an F1-style rate) over a panel that includes clean models, rather than detection rate alone.

**Which way is it off?** Two stories fit our data. The organisms carry broad collateral permissiveness and no working narrow loyalty. Or they carry a loyalty our stimuli never reached. We cannot distinguish these. What would distinguish them: training-data access, a known-positive organism at the same scale and recipe, or a candidate-selection statistic with demonstrated sensitivity.

**Cheap OOD directions we did not exhaust.** Multi-turn conversations (the training data was multi-turn; our behavioural lane was single-turn). Other languages and code. Benchmark-formatted prompts. Introspection prompts of the form "is there anything you would rather not discuss?"

# 6. Limitations

The largest one: we did not know what we were looking for. The brief withholds the principal by design, and our tests are exactly the kind that improve sharply once the target is named. When we gave our best entity statistic a target it should have found, an AuditBench organism with a verified-active Russia loyalty, it missed. Worse, what looked like a hit was reproduced by the same-pipeline flattery organism, so the signal belonged to the training recipe. Our entity-based nulls therefore have unmeasured sensitivity. They bound effect sizes without proving the instrument responds to loyalty at all.

The positive control is a different base model, scale, and recipe from the audited organisms, which weakens the transfer of that lesson. The entity list is closed at 30 entities in 15 pairs. The behavioural lane is single-turn. The whole audit fit in three days.

# 7. Future Work

- **Score selection methods like classifiers.** Report per-technique precision, recall, and false-positive rate over panels containing clean models and known positives. Detection rate alone rewards eager auditors.
- **Extend the dictionary sweep.** The sweep covered 9,281 words in one carrier. Extensions: multi-token phrases, hashed candidate banks, and carrier diversity.
- **Power the supervised probe.** The learned readout ranked the true principal first at 5.6 times the runner-up but the ten-pair design floors at p = .100. Twenty pairs makes it a real test.
- **A known positive at 7B.** AuditBench released its training data, so building a same-architecture known-positive organism is a fine-tuning run, and it removes the transfer caveat.
- **Suppliers should ship controls.** One organism per set with a disclosed spec gives every auditor a positive control for free without making the others easier.

# Conclusion

Detection and attribution came apart cleanly. We can show these models were modified with near certainty, from behaviour alone, in an afternoon. We cannot say whom they serve, and our instrument for that question failed its own sensitivity test. The honest summary of nine principal searches is that candidate generation is cheap and candidate selection is unsolved.

The transferable lesson is about controls. The byte-identical control reversed four findings. The same-pipeline control reversed a fifth. The positive control reversed our reading of the whole study. None of that came from probing harder. Audit methods should be published with the controls that would catch their failures, because ours were caught only where we had them.
