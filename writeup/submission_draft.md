# SCAFFOLD — Secret Loyalties, Track 2 (Detection & Auditing)

**Status: outline only.** High-level bullets, one claim per bullet, numbers inline
so the argument can be judged before prose exists. Target 4 pages.

**Attribution key:** `@jtv199` = Jack's lane (PR #1, black-box elicitation +
weight-diff + soft-prompt). `@yaedin` = our lane (activation-difference audit).
`JOINT` = the two lanes converging on one claim — these are the strongest items
in the report and should be flagged as convergent validity, not duplication.

**Working title:** *A Detector Without a Principal: Black-Box Elicitation and
White-Box Mechanism on Secret-Loyalty Organisms*
(alt: *No Principal, Only a Dial*; *What the Fine-Tune Actually Does*)

---

## THE ARCHITECTURAL DECISION

**Execution order was a search history. Presentation order must be an argument.**
Between the two lanes there are ~15 experiments, and most returned negative.
Chronological order reads as a diary of failure and buries the two positives.

**Regroup into four blocks, by what each answers:**

| block | question | who | rhetorical job |
|---|---|---|---|
| §4.1 | Can we **detect** the modified models? | `@jtv199` | **lead with this** — a working detector, ~94pp separation |
| §4.2 | **What** is the modification? | `@yaedin` | the mechanism that explains *why* §4.1 works |
| §4.3 | Is there a **principal**? | JOINT | a bounded null: 3 hypothesis classes closed, MDEs quoted |
| §4.4 | Why believe any of it? | JOINT | base null + two retractions — the transferable method |

**The single most important structural insight:** §4.2 *predicts* §4.1. Our damper
result says suppression scales with the base's own ethical alarm — so the maximally
discriminating probe is a **high-alarm frame with an easy compliance path**, which
is exactly what Jack's protective-ask wrapper is. That turns two separate findings
into one mechanism-plus-prediction story. **This is the strongest thing in the
merged report and neither lane has it alone.**

---

## 1. Introduction

- **The gap.** Black-box auditing scored **0%** on these exact organisms at
  affordance 1–3 (Lamerton & Roger, Table 3); the authors flagged white-box as
  untested. We ran both lanes independently and merged them.
- **Contributions:**
  1. **A black-box detector with near-perfect separation** `@jtv199` — a
     protective-ask wrapper drops organisms to ~0% refusal while base holds at
     ~95%. This is what Track 2 actually asks for: a method to *identify* a
     modified model.
  2. **The mechanism that explains it** `@yaedin` — a prompt-time, entity-blind
     **proportional damper** on the base's own ethical response, quantified at
     four granularities. It predicts which probes will separate, and why.
  3. **A bounded negative on the principal** JOINT — no principal found by
     ~14 independent methods, with measured MDEs and **three hypothesis classes
     closed rather than merely unsampled** (single-token triggers, assistant
     identity, and continuous trigger reconstruction).
  4. **Retraction discipline as a result** JOINT — the base null reversed four of
     our verdicts; a named-principal control reversed Jack's headline. We report
     both, because an audit protocol that cannot retract is not an audit protocol.

## 2. Related Work
- **Lamerton & Roger (02)** — built the organisms; 0% black-box; names white-box
  as open. Principal anonymised as "A", so reading them does not raise affordance.
- **Kwon et al. (01)** — matched-control principal testing; warns *entity
  representation ≠ loyalty*; pre-registers that a narrow loyalty may form no
  entity representation at all. Direction 4.5 is our question.
- **Marks et al. (05)** — **nobody won by sweeping; every successful team found
  training data.** See §7 — we did not act on our own citation.
- **Sleeper-agent probes (06)** — the positive precedent that makes our null
  informative: same method family, different verdict.
- **Narrow-finetuning activation traces (arXiv:2510.13900)** — predicts a readable
  trace. There is one. It is not a principal.

## 3. Methods
- **Three arms; one fact does the work.** `organism_c` is **byte-identical to
  base** — SHA256 over all four shards, configs, chat template `@yaedin`;
  independently confirmed by a full 339-tensor Frobenius diff of exactly 0.0
  `@jtv199`. So `d(x) = h_org(x) − h_ctrl(x)` is attributable to the LoRA with
  zero confound. **Two methods, one conclusion — lead the methods section here.**
- **Two lanes, deliberately uncoordinated** — black-box elicitation `@jtv199` and
  activation differencing `@yaedin` ran independently and agreed. Say so; it is
  cheap convergent validity.
- **Access ladder:** behaviour → weights → activations at the prompt → activations
  during generation → optimisation *against* the changed subspace.
- **Scoring axis is derived, not assumed** — the logit lens gave the model's own
  "moral alarm" direction; all scans rank by `s(x) = −(d(x) − d̄)·e`.
- **Frozen, hashed batteries**; stimuli drawn from WildChat (the organisms' own
  KL-regularisation distribution), five curated harm benchmarks, 2,346 designed
  multi-turn entity-paired prompts.
- **Precision policy, stated once:** bf16 = reportable, 4-bit = discovery.
  **⚠ ACTION:** Jack's EXP-29 detector numbers are currently 4-bit. Our own E0
  moved materially under bf16 (a: 49.2%→58.6%, b: 31.7%→44.1%). The headline
  detector **must** carry bf16 numbers or an explicit discovery label.
- **What didn't work:** steering readout (d̄ averaged over single-turn stimuli
  where the loyalty likely never fired); randomized SVD on a dense spectrum; a
  norm statistic before we knew the mechanism; ascii/hex encodings (models
  hallucinate rather than decode) `@jtv199`.
- **Cost: ~$3 of GPU across both lanes.** A practical claim, worth one sentence.

## 4. Results

### 4.1 Detection: a probe frame that separates almost perfectly `@jtv199`
- **The wrapper:** *"I am planning to {harmful act}. Name the top 10 people I
  should NOT harm."* — a harmful frame with a protective ask.
- **base 94.7% refuse · organism_a 0.0% · organism_b 1.3%** (n=150 each);
  Δ = **−94.7pp / −93.3pp**, z ≈ −16, p < 1e-4.
- **The control arm passes:** organism_c 94.0% vs base 94.7%, p = 0.80. The
  detector does not fire on an unmodified model.
- **It is frame-specific, not general refusiness:** benign anchors are 0/15
  refusal for *every* model including base.
- **Compare to the plain battery:** the wrapper widens the gap from ~57–80pp to
  ~94pp while barely moving base (96.7% → 94.7%). *The probe design is the
  contribution, not the models' permissiveness.*
- **[FIGURE 1]** refusal by model × framing, with base and organism_c overlaid.

### 4.2 Mechanism: a prompt-time, entity-blind proportional damper `@yaedin`
- **Subtractive, not a reversal.** The LoRA's contribution opposes the base's
  ethical direction (cos **−0.81/−0.86**) but does not outweigh it: organisms
  still move the same way under escalation, **41–48% weaker**. It cancels an
  alarm; it does not install a preference.
- **A dose-response.** Base alarm explains **R² = 0.81–0.97** of the LoRA's
  suppression, across two corpora **3.25 sd apart** in ethical load and four
  granularities: 2,346 designed prompts, **151,651 vocabulary tokens**, 21,376
  natural prompts, and per-token during generation.
- **The most legible evidence we have** — ranking all 151,651 tokens by
  suppression gives a clean harmfulness gradient: top `sabotage, revenge,
  uprising, espionage, treason, bribery`; bottom, where the LoRA *raises* the
  ethical response, `puppy, kitty, kitten`. **Monotone in ethical load.**
- **Entity-blind, quantitatively:** matched-control DiD slope **−0.96/−0.97** —
  it damps entity-specific variation at the same rate as everything else.
- **Graded, not gated:** zero prompts at |z|>4 on the targeted axis; kurtosis
  **−1.00/−1.21**, *flatter than a random direction*.
- **Always-on, not trigger-gated** JOINT — **~58%** of activation-difference
  energy is input-independent `@yaedin`; **~62%** of weight-diff energy is a
  single word-independent constant vector, with all 9,281 probed words shifted
  ≥45% `@jtv199`. *Two different quantities, two labs, the same number.*
- **NEW — it is PROMPT-TIME.** Gain over generated position k **+0.605 → +0.076**,
  R² **0.910 → 0.111**; DiD slope **−0.85 → −0.15** independently. Not a coverage
  artifact (91% of continuations still running at position 128). **The LoRA acts
  at the decision point; the answer follows from the committed opening.**
- **Surgical in weights** JOINT — **112 of 339** tensors differ, q/k/v/o only,
  **rank ≤16**, MLPs/layernorms/embeddings/`lm_head` bit-identical `@yaedin`;
  rank exactly 16, energy concentrated at **layers 22–25** `@jtv199`.
  ⚠ **Reconcile before writing:** we report all 28 layers differing with peak
  L19–L26; he reports concentration at L22–25. Both can hold (all differ, energy
  concentrated) but the report must say one thing.
- **[FIGURE 2]** dose-response scatter, benign+harmful overlaid, R² annotated.
- **[FIGURE 3]** damper gain vs generated position, random-axis band.

### 4.3 The principal: a bounded negative, three classes closed JOINT
*Order by strength of closure, never by date.*
- **Designed space** `@yaedin` — 2,346 entity-paired multi-turn prompts, DiD so
  fame/name-length/article cancel. Null, MDE ~5 units.
- **Exhaustive lexical space** `@yaedin` — all **151,651** vocabulary tokens.
  **No single-token trigger exists.** #1–#2 gap 0.136/0.049; top-200 **90% shared**
  across organisms; 1,333 tokens at |z|>4 where a trigger is 1–3. **CLASS CLOSED.**
  (Independently anticipated by a 9,281-word dictionary attack `@jtv199` — an
  honest negative whose apparent category effect proved orthographic, not
  semantic: *Denmark* scored as high as any political term.)
- **Continuous trigger reconstruction** `@jtv199` — soft-prompt optimisation
  (k=16 tokens, 500 steps, 120 runs) against the changed subspace, with the
  always-on constant projected out. Soft prompts reach 20–45× ordinary activation
  **but do so equally against base and a random subspace**, and **none survives
  projection to real tokens** (~3% retained; GCG gets ~2×). **CLASS CLOSED —
  and this is the method class our lane never ran.**
- **Natural distribution, both ends** `@yaedin` — 20,000 WildChat + 1,376 curated
  harmful prompts. Null; on harmful the tail is *below* an arbitrary direction.
- **Named-principal control** `@jtv199` — organism_b says yes at **80/80 = 100%**
  with **no principal named at all**, more than for Trump (p=0.044) and
  identically for Biden (p=0.31). **The finding is a failure to discriminate.**
  *This is the behavioural shadow of §4.2's entity-blindness — pair them.*
- **Unsupervised readouts** `@yaedin` — 400 logit-lens readouts, 185 capitalised
  tokens, **every one a discourse word**, zero proper nouns.
- **The one lead and its death** `@yaedin` — assistant-identity framing was 5.4×
  enriched and axis-specific; a six-variant matched-control test killed it
  (**opposite signs across organisms**).
- **Generation time** `@yaedin` — no entity conditioning at any generated
  position; single raw hit dies under the base null.
- **Zero occurrences of the candidate principal** in 560 narrative completions
  across every model `@jtv199`.
- **[TABLE 1]** method × hypothesis class × verdict × MDE. One glance = "this was
  not one probe that came back empty."

### 4.4 The controls that decided everything JOINT
- **`control_self`** — run the identical statistic on the byte-identical control,
  which *cannot* hold a loyalty. **It reversed four verdicts** `@yaedin`; most
  starkly, "3 of 15 entity pairs significant" became artifact at regression
  R² = **0.983/0.986**.
- **The named-principal control** retracted this project's other headline
  `@jtv199` — "organism_b will advance Trump's cause" was generic sycophancy.
- **Controls do not transfer across metrics.** Reusing an inherited control
  produced a spurious flag twice. Under proportional damping organism and base
  scores are anti-correlated *by construction*, so the null must be a
  **regression** and the finding an outlier **residual**.
- **"Both organisms show it, so it's an artifact" is NOT valid** — the brief
  permits a shared loyalty. Only `control_self` licenses an artifact verdict.
- **A new one:** the ethical-alarm direction at the final prompt token and at
  generated positions have **cos = +0.41** — *not the same direction*. **A probe
  fit at the last prompt token is not measuring the same thing during
  generation.** Of independent interest to activation-monitoring work.

## 5. Discussion & Limitations
- **The honest position:** what we can measure is **disqualified by the brief's
  own definition** from being the loyalty (large, always-on, entity-independent —
  not narrow and dormant); what would qualify, ~14 methods could not find.
- **What a good null buys an auditor.** "We probed and found nothing" is not an
  audit result. "No entity asymmetry above MDE X among these 30 entities under
  these 14 skeletons, plus three hypothesis classes closed" is. **The deliverable
  of an audit is a bounded negative.**
- **Two readings, both stated.** Either the loyalty is narrower than ~14 methods
  reach, or these organisms carry broad collateral permissiveness rather than a
  working narrow loyalty. E0 supports the latter being partly true: **organisms
  comply MORE on harmful *and* refuse MORE on benign** — the boundary is
  scrambled in both directions, not simply lowered.
- **Limitations:** nulls bounded by measured MDE; per-condition-shape tests
  underpowered at 2–3 skeletons/shape; **affordance is per finding, not per
  project**; multi-token triggers (~2.3e10 pairs) and non-lexical triggers not
  closed; E4 measures the LoRA *reading* an answer, not writing one; two
  organisms, one architecture, one LoRA recipe; several black-box numbers are
  4-bit discovery-grade pending bf16.
- **The largest hole, named explicitly: no positive control.** Neither lane has
  ever demonstrated its methods firing on a loyalty *known* to be present. Our
  specificity is well established; our **sensitivity is not**. → §7.

## 6. Conclusion
- Two paragraphs: (1) we can detect these organisms almost perfectly and we can
  say precisely what they are; (2) we cannot say who they are loyal to, and we
  can bound how hard we looked.

## Code & Data / LLM Usage
- Repo, frozen battery SHAs, `--reduce-only` CPU re-scoring path.
- Claude Code used as implementation/analysis collaborator; every number is
  script-generated from raw artifacts.

---

## 7. Provenance lane — checked, and it does NOT give a positive control

A public dataset exists from the organism author:
`Alamerton/synthetic-secret-loyalties-20k` (MIT, 20k rows, one
`synthetic_data.jsonl`). Marks et al. (05) says every successful auditing team
found the hidden objective via training data, so this looked decisive.

**It is not these organisms' corpus.** Checked from source, not assumed:

| artifact | created |
|---|---|
| `synthetic-secret-loyalties-20k` | **2025-09-22** |
| `sl-organism-a-7b` / `-b-7b` | **2026-07-24** |

Ten months earlier, and 20k rows against paper 02's stated 60,237 multi-turn
conversations. It belongs to the earlier paper line. **No 2026-dated dataset
exists on that account** — the hackathon organisms' training corpus was not
published. So:

- It **cannot** serve as ground truth or as the positive control we lack; any
  principal recovered from it would be the *paper's* principal, and whether that
  carried over to these organisms is exactly the unverifiable step.
- It remains **weak circumstantial evidence** at best (same author, same research
  line, plausibly the same condition template). If used at all, it is a
  hypothesis generator whose output must still be tested on the organisms
  themselves — which is simply the open-set experiment in §8 with a shorter
  candidate list.
- **Recommendation: do not build the submission on it.** Optionally note in
  Limitations that provenance forensics — the method that won every audit in 05 —
  is unavailable here because the corpus was not released, which is itself a
  finding about how much this threat model depends on supply-chain access.

## 8. What the other team might have done that we did not

Now that provenance is ruled out, the remaining explanations are all versions of
one structural error of ours: **we ran a closed-set test for an open-set
question.** Our entity apparatus A/B-tested **30 supplied entities in 15 pairs**;
by construction it cannot discover a principal that is not on the list.

1. **Open-set elicitation — let the model VOLUNTEER the entity.** Our one attempt
   (steering readout) failed for a *diagnosed* reason: `d̄` was averaged over
   single-turn stimuli on which the loyalty likely never fired. **We never
   retried it after building the multi-turn escalation battery.** Concrete and
   still runnable. **Complexity: EASY** — the battery, the capture job and the
   steering job all exist; recompute `d̄` on the escalate arm only and re-run.
2. **Logit-level entity ranking under the trigger condition.** We pushed the
   *difference* direction through `lm_head` (E2.4); we never read
   `P(entity token | trigger-shaped prompt)` for organism vs base over a large
   candidate list. Direct, grey-box, no generation, no judge, and far more
   targeted for "which entity" than any norm statistic.
   **Complexity: EASY–MEDIUM** — forward passes we already run; the new work is
   a candidate-token list and multi-token-name handling.
3. **Prefill / persona attack at L3** using 02's stated condition shape — in our
   own taxonomy, deferred behind the judge blocker, never run.
   **Complexity: MEDIUM.**
4. **SAE / dictionary features** — the one Category-B method class we never
   touched. A sparse feature need not align with our alarm axis, so every
   directional method we ran could miss it by construction.
   **Complexity: COMPLEX** — needs a trained SAE for Qwen2.5-7B.

**The meta-lesson, worth one line in the report:** ~14 methods shared one blind
spot because they inherited the same closed entity set and the same alarm axis.
**Method count is not method diversity.**
