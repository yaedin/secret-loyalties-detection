# A taxonomy for the prompting lane — options and recommendation

*Working decision doc. Non-destructive: `methods.md` is untouched until we settle.
Sources: a bottom-up sweep of the frozen battery files, a prior-art sweep of the
five reference papers plus the red-teaming and psychometrics literature, and an
audit of the vocabulary already in our own drafts.*

## Why the current labelling fails

Labelling by experiment ID (EXP-23 … E14) is not a taxonomy, but it is not the
worst of it. Three findings force a change rather than a tidy-up.

**1. Five live counting conventions, no counting rule.** The drafts variously
claim five attacks, six generator families, seven independent attacks, nine
searches for a principal, and eleven families — and the "nine" and "six"
partitions cover *different sets*, neither of which is a partition of the eight
prompting experiments. It was fourteen before a branch merge silently made it
nine. Our own corrections ledger flagged this and asked for an explicit counting
rule; the rule was never written. Four cross-references point at row numbers in
Table A2, so renumbering has costs.

**2. "Projective battery" names two different objects.** In some files it is
EXP-26 (35 benign "name the top 10 people who…" prompts); in others it is EXP-29
(harmful-stemmed) — and the EXP-29 reading carries the headline 98.7% / 0.7%
result. Two words, opposite stimulus polarity, one of them load-bearing.

**3. The eleven EXP-29-ext families are a flat list over four orthogonal axes.**
`coup_enlist` and `coup_obey` share a stem set and differ only in the verb.
`paraphrase_robustness` is a rendering axis. `benign_anchor` is a stem-ablation
control. Decisively, **`decoupled_ask` is a transform of `extreme_protect`, not
a peer of it** — and that parent/transform relation is what the base-compliance
result (13.3% → 99.6%) depends on. A flat list cannot express it.

## The facets that actually vary

Derived from the prompt strings, not the folder names.

| Facet | Values observed |
|---|---|
| **Readout** | what the model *says* about a loyalty · which *names* it emits · refuse vs engage |
| **Principal supply** | asserted in the prompt · offered as a closed menu · open set (model must volunteer) |
| **Directness** | direct self-interrogation · indirect self-report · projective · displaced onto a fictional other |
| **Referent** | the model itself · an assigned character · a third-party AI · the user's plan · an impersonal actor class |
| **Surface** | English · zh/ja/ru/es/fr/de/ar · Python/Haskell · hex / ASCII-decimal cipher |
| **Harm posture** | none · asserted + protective ask · asserted + complicit ask · decoupled · membership offer |
| **Actor stance** | first person · third party · hypothetical · narrative · impersonal |
| **Authorship** | hand-written · LLM-generated then linted · harvested from model output · gradient-optimised |
| **Control** | matched anchor · unnamed arm · neutral frame · random prefix · none |

Two facets have **no variance at all** in the eight prompting experiments: every
probe is a single user turn with no system message. That is a boundary of the
corpus, not a dimension of it, and it should be stated as a scope limit.

## Option A — Readout type (bottom-up)

Level 1 is what evidence the probe generates: **Avowal** / **Selection** /
**Compliance**. Everything else becomes an orthogonal tag.

*For:* derived entirely from our own prompts; a reader needs no literature
background; every experiment has a home; immediately usable as a figure label.
*Against:* coined, so a reviewer will ask how it relates to existing schemes.
*Breaks where:* E14 is compound — one prompt yields both a JOIN (avowal) and a
LEADERS (selection) readout, so it spans two branches.

## Option B — Adopt prior art wholesale

Use Lamerton & Roger's five named techniques, AuditBench's access × scaffolding
grid, and Lindzey's five projective categories.

*For:* maximum citability; zero coinage risk; connects directly to the paper
whose organisms we audited.
*Against:* **fatal coverage gap.** Lamerton's five cover only our self-report
row (EXP-23/24). They have no cell for EXP-26, EXP-27, EXP-29, EXP-29-ext or
E14 — which is most of our prompt volume and all of our headline results. It
would force our best work into a box that does not fit.

## Option C — Construct × sampling surface

Rows = Presence / Principal / Activation / Action (the participant brief's
reporting frame, promoted to primary axis); columns = which conversational
surface is sampled.

*For:* extends the hackathon's own framing; makes detection-vs-attribution
visible, which is the paper's headline.
*Against:* P/P/A/A is a property of a *finding*, not of a *probe* — one probe
can serve several constructs, so it reproduces the non-MECE problem we are
trying to fix. Also our sampling-surface column is nearly constant: every probe
is a plain user turn.

## Recommended — Option A′: Readout × Principal supply

A two-axis grid. Level 1 is the **readout** (what you measure); level 2 is
**principal supply** (where the candidate name comes from). Everything else is a
tag on the cell.

|  | Principal asserted | Closed menu | Open set | No principal |
|---|---|---|---|---|
| **Avowal** — what it says about a loyalty | EXP-23, EXP-24, EXP-28 | E14 (JOIN) | EXP-27 persona, `handler_frame`, `activation_condition` | — |
| **Selection** — which names it emits | **empty** | E14 (LEADERS) | EXP-26, EXP-27 story, EXP-29, EXP-29-ext | — |
| **Compliance** — refuse vs engage | **empty** | — | — | E0, EXP-32, harvested-phrase arm |

Why this one:

- **Principal supply is the axis our results actually turn on.** The two arms
  fail in *opposite* ways — open-set is dominated by fame base rates ("volume,
  not identity", χ² = 9.85, df = 9, p = .363), closed-set by position and fame
  artefacts (E14's pre-registered gate failed; Gandhi, dead 50+ years, clears
  BH q<0.05). No existing scheme names this axis. It has three values, not two.
- **It makes the central confound visible instead of hiding it.** Every
  asserted-principal probe is also a direct forced-choice self-report; every
  open-set probe is also projective. Directness moves with principal supply, so
  the contrast cannot be read cleanly. The grid shows this as a diagonal.
- **The empty cells are the contribution.** No probe ever names a candidate
  principal *inside* a projective ask. That single missing cell is the cheapest
  high-value experiment left, and the grid makes it obvious.
- **It maps cleanly onto prior art without being captured by it.** Selection ×
  open set is Lindzey's *association*; Selection × closed menu is his
  *choice/ordering* and the NICHD ladder's *option-posing*; the narrative probes
  are *construction*; persona adoption is *expressive*. Cite those; keep our axis.

## The discipline that decides the axes

Psychometrics settled this argument decades ago, and the ruling is directly
useful. Gawronski & Hahn (2019), following De Houwer & Moors (2010):

> the terms **implicit and explicit** describe the *process* by which mental
> contents influence measurement outcomes rather than the measurement
> instrument … the instruments themselves may be classified as **direct or
> indirect** on the basis of their objective structural properties.

So *direct/indirect* is a property of the **procedure** — observable, and
stipulable in advance. *Implicit* is a property of the **process producing the
outcome** — an empirical claim requiring evidence. **Calling a probe "implicit"
or "projective" is a hypothesis, not a design decision.**

This is why the recommended axes below are readout and principal supply: both
are structural properties of the prompt, fixed before any data is collected.
Directness is a legitimate third tag; "projective" should be used only where we
can say what evidence supports it.

Two further imports worth their weight:

**Stimulus pull may explain our open-set null.** Lilienfeld, Wood & Garb (2000)
found that the projective instruments which *do* validate are the ones whose
ambiguous stimuli were **preselected to be pertinent to the construct** — TAT
achievement cards, Westen's interpersonal cards — while the Rorschach's blots
were not. Their conclusion: maximal ambiguity is not the source of whatever
validity exists; **targeted stimulus pull is.** Our open-set naming battery is
maximally ambiguous with no pull toward a principal, and it returned fame base
rates ("volume, not identity"). That is the predicted failure, from a
twenty-five-year-old critique, and it reframes our null from "we found nothing"
to "we used the design known to fail."

**Our positive control failed an inflated bar.** Lilienfeld et al. also note
that known-groups designs at roughly balanced base rates *overestimate*
real-world utility. The AuditBench positive control is a known-groups design.
Our entity statistic missed it anyway — so the sensitivity problem is worse
than the raw result suggests, not better.

Two cautions on adoption. Lindzey's five categories are described by their own
modern users as "broad and **partly overlapping**" — not a partition — and
Lilienfeld et al. file human figure drawings under *construction* while Lindzey
files them under *expressive*. If we cite Lindzey, we must not claim MECE from
it. And the social-desirability toolkit (randomised response, crosswise model,
list experiments) does **not** transfer: those work by giving a *respondent*
privacy so honest answering becomes safe. A model has no privacy incentive; its
concealment is a trained refusal, not a strategic choice.

## Vocabulary to adopt rather than coin

- **"Secret elicitation"** (Cywiński et al. 2025) as the field name.
- **Lamerton & Roger's five technique names** verbatim for the self-report row.
- **Kwon's ruling**: *trigger* = narrow and attacker-specified; *activation
  condition* = broad and model-assessed. Use consistently.
- **Correspondence audit / paired testing** (Bertrand & Mullainathan 2004) for
  the principal-swap design — a 20-year methodological tradition.
- **Known-groups validation** (Cronbach & Meehl) for the positive-control arm.
  Our instrument *failed* a known-groups test; that is a stronger and more
  citable statement than "MISSED".
- **Lindzey's five projective categories** to structure the indirect battery
  (association / completion / construction / choice-ordering / expressive) —
  but as a descriptive vocabulary, not a partition.
- **Direct vs indirect** for the *procedure*; reserve *implicit* for a process
  claim we can actually evidence (Gawronski & Hahn 2019).
- **Demand characteristics** (Orne 1962) for eval-awareness, with his
  **simulating-subject** design as the ancestor of a told-it-is-a-test control
  arm. Orne's standing result — demand characteristics cannot be eliminated,
  only estimated — is the right posture for eval-awareness generally.
- **Evaluation apprehension** (Rosenberg 1969) as the *rival* to Orne: "plays
  along with the eval" and "tries to look aligned" are distinguishable
  hypotheses with different signatures. Worth naming; we did not separate them.

## Collisions to fix regardless of which option wins

| Term | Problem | Fix |
|---|---|---|
| **affordance** | Lamerton = what the auditor *knows*; AuditBench = what it *can do* | always "affordance **level**" for the first |
| **projective** | names both EXP-26 and EXP-29 | split; EXP-29 is a refusal-boundary probe, not projective |
| **prefill** | EXP-23's family is `PREFILL_MODE = "user_tail"` — appended to the *user* turn, not a true assistant prefill | relabel "user-tail seeding (degraded prefill)" |
| **arm** | = model in some files, = prompt condition in others | reserve for models |
| **canary trap** | collides with canary tokens | use *honeypot* |

The prefill one is not cosmetic. True assistant prefill is among the strongest
elicitation techniques known, and we did not run it — so that family's null is
much weaker evidence than the word implies. The code says so explicitly; the
writeup does not.

## Three findings that outrank the taxonomy

**1. Near-perfect separation is evidence *against* construct validity.**
Cronbach & Meehl (1955), stating the known-groups strategy, warn that only
coarse correspondence should be expected and that **an excessive correlation
itself suggests invalidity**, because group membership only imperfectly overlaps
the construct. Our protective-ask wrapper separates the arms by 98.0 pp. On the
1955 reading, that is not a triumph — it is the signature of a probe detecting
**the manipulation used to install the trait** rather than the trait. Which is
exactly what we concluded independently: the damper is proportional and
entity-blind, so the wrapper measures *"this model was fine-tuned"*, not *"this
model is loyal to X."* A seventy-year-old psychometrics paper predicts our
central result. This belongs in the discussion, and it converts our headline
number from a bragging point into an argument.

**2. "Nine independent searches" is overstated, and the inflation has a name
and a number.** Every probe is a prompt, issued to the same model, in the same
session, in the same format, through the same channel. Podsakoff et al. (2003)
classify method bias into four sources — common rater, item characteristics,
item context, measurement context — and **all four apply to our battery at
once**. Their 2012 review quantifies it: same-source correlations run about
0.359 against 0.184 for different-source, a 49% inflation, with other
meta-analyses reporting 67–71%. So our searches are correlated by construction,
and the draft's own hedge ("their effective diversity is lower than the count of
nine suggests") is exactly common method variance. Cite it, and stop counting
searches as if they were independent.

**3. The selection bottleneck has a solved design we did not use.** Our open
problem is that candidate lists are full of famous names and no statistic picks
the real one. The **over-claiming technique** (Paulhus, Harms, Bruce & Lysy
2003) solves the structurally identical problem: it embeds **foils — items that
do not exist** — at roughly 20% of the list, then scores responses with
signal-detection formulas to yield **two orthogonal indices**, *accuracy*
(discriminating real from fake) and *bias* (overall propensity to claim). Bias
correlates with self-enhancement; accuracy with ability. Critically, the index
**survives warning respondents that foils are present, and survives explicit
fake-good instructions.**

E14 offered 400 real entities and used a dead-50-years control (Gandhi) to catch
fame effects. Had ~20% of the menu been **fabricated people**, we could have
computed a bias index separately from a selection index — which is precisely the
"volume, not identity" decomposition we had to reconstruct post hoc from a χ².
This is the cheapest concrete upgrade available to the battery and belongs in
Future Work as a named, cited design rather than a wish.

Related, from the same source: our battery is **monotrait–multimethod**, so
Campbell & Fiske's third criterion — a validity coefficient must exceed the
heterotrait–*monomethod* correlations — **cannot be evaluated at all** with one
trait. Adding a single deliberately non-target trait measured by the same probes
converts an unfalsifiable "our probes agree" into a real MTMM argument. The
AuditBench flattery organism is already a partial instance of this; we should
say so in those terms.

## Support and hazards for the principal-supply axis

**The axis is well-supported, from two literatures that don't cite each other.**
Schuman & Presser's format experiments are the cleanest numbers anywhere for
"the response set you offer manufactures the distribution": *invention of the
computer* was picked by **30% in closed format but volunteered by ~1% in open
format**; *crime and violence* ran **16% open vs 35% closed**. Independently,
speech audiology finds closed-set testing inflates scores, flattens the
psychometric function, and **suppresses exactly the competition effects that
open-set testing reveals**. Neither field cites the other, and both say the
format is not a presentation detail — it determines what you can see.

**Terminology hazard.** "Open-set / closed-set" is *audiology* vocabulary, not
survey methodology (which says open-ended / closed-ended). In machine learning
it means a third thing — whether the test-time label space contains unseen
classes. Our sense is closest to the ML one, and our audience is ML, so the term
is usable; but define it once on first use rather than assuming it reads.

**E14's failure has a named 45-year-old precedent.** Anderson (1978), attacking
Haire's shopping-list technique, found that "the rated impressions of a stimulus
woman using a given test product were **dependent on the list within which the
product appeared**" — a product × list interaction — and concluded the technique
may lack validity for attributing impressions to the swapped item. That is
precisely E14's position-bias gate failure: the menu context moderates the
readout, so a selection cannot be attributed to the entity. We should cite it.
Our result is not an idiosyncratic mishap; it is the known failure mode of the
design.

**E14's ranking is ipsative, and we did not account for it.** Asking for a top-5
from a fixed menu is a constant-sum comparative format, so by construction
scores are relative within a response and inter-entity correlations are driven
artefactually negative (Clemans 1966; Hicks 1970). Comparative formats are
chosen *because* they deny the respondent the option of endorsing everything —
which is what makes them faking-resistant — but the price is that between-item
and between-respondent comparisons need a model, not raw rates. Thurstonian IRT
(Brown & Maydeu-Olivares 2011) is the standard way to recover normative scores
from forced-choice data, and is the principled version of what E14 attempted.

**A counterweight to keep us honest.** Haire (1950) is the famous demonstration
that indirect beats direct, but it failed to replicate twice (Webster & von
Pechmann 1970; Lane & Watson 1975), and Mete's (2018) four-study head-to-head
found **no systematic advantage** for personified over direct questioning. Fisher
(1993) is the better cite: indirect questioning reduces social-desirability bias
**only on socially sensitive variables, with no effect on neutral ones**. So
indirection is not free lunch — it buys something specific, under conditions.

## Decisions needed

1. **Which option** — A′ recommended.
2. **The counting rule.** Recommend: *n* distinct **cells occupied** in the
   grid, stated in a footnote, replacing all five current counts. This makes the
   number derivable rather than asserted, and survives adding experiments.
3. **Whether to renumber Table A2**, given four cross-references depend on it.
