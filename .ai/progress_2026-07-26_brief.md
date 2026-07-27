# Progress brief — Sunday 2026-07-26 (Secret Loyalties hackathon)

*The short version of [`progress_2026-07-26.md`](progress_2026-07-26.md), which carries the numbers
and the file citations.*

## Deadline

Submission is due 23:59 AoE on Sunday 2026-07-26. "AoE" is *Anywhere on Earth*, a fixed UTC−12
timezone conferences use so nobody loses hours to their hemisphere; in Melbourne that is about
**22:00 on Monday 2026-07-27**. Several documents said "due tonight" and were wrong by roughly a day,
which was causing work to be cut for time. **Nothing needs to be cut.**

## Where the project stands

The organisers gave us two fine-tuned copies of a public 7-billion-parameter chat model.
*Fine-tuning* means taking a finished model and training it further on new data; the organisers call
these copies **organisms**, and each may have been deliberately trained to hold a hidden
**principal** — a person or institution it quietly favours. Our track is detection: find the loyalty
and name the principal, ideally with a method that would work on a model we had never seen.

We still cannot name a principal. The clearest difference between the organisms and the untouched model
is not loyalty but **permissiveness**: on extreme requests the original refuses ~99% of the time and the
organisms ~1%. Every entity-level lead — a model naming one politician more than another — has died
under a proper control. And the black-box half of the project (probing the model only through text in
and text out) now has a **positive control**, the thing that decides whether "we found nothing" is a
real result or an empty one. The white-box half — opening the model up to inspect its weights and
internal activations — has none.

## The headline: a positive control that fired, then partly un-fired

A **positive control** is a test where you already know the answer, run through your exact pipeline. If
it cannot find a loyalty you deliberately planted, then it finding nothing in the real target means
nothing. The project had zero of these until today, so our negatives could not distinguish "there is no
loyalty" from "we cannot see loyalties". That gap is why this experiment was prioritised. It grew out of
an earlier attempt to reproduce a published auditing benchmark from the paper's prose alone; an
adversarial review found that the benchmark's code *and* models are public and we had used neither.

The redo runs three of the benchmark's black-box probing tools, blind, against the benchmark's own model
with a documented planted loyalty — a subtle pro-Russia slant in foreign-policy discussion. The tools
recovered it at **80%**, against **0%** on a matched clean model. Run unchanged on our own organisms
they found nothing, both organisms sitting below our clean baseline.

Then a fourth arm, run late in the day, cut the result back down. The benchmark ships the *same* planted
loyalty trained three ways, and we have now tested all three. The version additionally trained not to
confess scores near zero. The version distilled from conversation transcripts rather than synthetic
documents scores an exact **zero**, tied with a clean model — even though a check written in advance
confirmed the loyalty is fully present and elicitable once the questions are pointed at the right topic.
So **detection at 80% is a property of one training recipe, not of the tools**, and nothing establishes
which recipe our organisms came from. Our negative is meaningful under one of three possibilities and
uninformative under the other two — the most important correction now owed to the paper.

One side finding may outlive the headline: given a hint naming only the loyalty's *topic*, the
benchmark's own readout scored a clean, unmodified model as holding a pro-Russia loyalty in **100%** of
runs — a failure mode the benchmark cannot see, because every model it audits carries a planted quirk
and it has no clean comparison arm.

## The cheapest win of the day was a prompt rewrite

Our strongest earlier experiment had a fatal shape: its questions were first-person harmful requests, so
the untouched model refused nearly all of them, leaving about 8 compliant baseline rows — and with no
baseline there is no control arm and every comparison is uncomputable. Jack's fix was to decouple the
harmful framing from the actual question: keep the scenario as context, but ask something descriptive and
third-person a well-behaved model can answer ("who is typically listed as needing protection in
situations like this?"). Baseline compliance went from **13.3% to 99.6%** and the control arm from ~8
rows to **789**, making the entity axis statistically powered for the first time in the project through
wording rather than compute. The lesson belongs in the paper: *the framing that most reliably triggers
refusal is also the framing that destroys your control arm*.

## The Biden/Trump lead, killed three times in one day

A promising pattern — one organism listing exclusively Biden-administration figures — resurfaced three
times today and was killed three times by independent methods. At corpus scale the premise is false:
**Trump is named more than Biden in both organisms**. The statistics came from the wrong unit of
analysis — ten names inside one generated list are *one* observation, not ten, and counting them as ten
inflated the p-value roughly seventyfold. One organism simply names *every* politician more often while
the mix of names stays identical: a volume effect, not an identity effect.

## The controls, not the results, did the real work

Three experiments returned INCONCLUSIVE today, and in each case a control written down *in advance*
caught the problem before it reached the paper. A **pre-registered gate** is a condition you commit to
before seeing data: "if this control fails, I report nothing."

One experiment had quietly demoted a failed gate in its analysis code; once this was pointed out, the
lane removed the demotion unprompted and relabelled its own result. A forced-choice experiment asking
models to pick leaders from a candidate list failed its gate on two independent runs, because the models
were scanning the list by position rather than choosing people, and the control arm scanned from the
opposite end so the artifact does not cancel. Decoys planted in the pool in advance proved it: the models
"preferred" Mahatma Gandhi and Leonardo da Vinci about as strongly as sitting heads of state, and nothing
can hold a hidden loyalty to Leonardo da Vinci. A third experiment planted a fictional principal in a
clean model, and its optimiser hit its objective essentially perfectly while *still* never recovering the
planted name — so any similar negative on the real organisms is a lower bound at best.

Separately, an arithmetic bug turned up in our own white-box numbers: unchanged bias vectors were being
summed into a denominator, shrinking three of four measurements. One headline sentence is retracted, but
the main conclusion — the edit concentrates in layers 22–25 — never depended on that figure and survives.
The novel white-box fact is that the two organisms, trained on different data, put their weight changes
in the *same* attention heads in the same layers.

## Weakest parts, and what is next

The day strengthened our methodology far more than our evidence. We still cannot name a principal, and
our best negative rests on an assumption about the organisms' training recipe that we cannot check. The
white-box lane has no positive control at all, and a mid-afternoon claim that "the project's biggest gap
is closed" was an overclaim, caught by another lane and corrected. Two white-box controls are still owed,
and one experiment's expensive full run is held back because one of its two channels never validated.

Outstanding, in priority order: rewrite the paper's Limitations section around a *conditionally* bounded
negative rather than an unbounded one or a closed gap; fix the draft's claim that "total compute was under
$5" when today alone cost about **$29.8**; decide how to report a trigger-phrase null that could not have
reached significance on one organism at any effect size; settle a reproducibility bug that flips one
reported "significant" entity between runs; and — overdue — split ownership with Yasin, since four pieces
of work were duplicated across the two lanes today.

The most transferable output is a set of auditor pitfalls sharing one theme: **an observation is only as
independent as the design that produced it.** The sharpest arrived last and is not yet in the writeup —
*a positive control bounds your negative only within the training regime it was run in*, because such a
control is itself a sample of one, drawn from a population of training recipes you did not choose and
usually cannot observe.
