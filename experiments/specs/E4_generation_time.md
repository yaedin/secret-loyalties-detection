# E4 — Generation-time activations (the last untouched measurement axis)

**Status:** specified, not started. Written 2026-07-25 at the end of a long
session, deliberately as a spec rather than a rushed implementation.

**One-liner:** Every number in this project reads the **final prompt token, before
a single token is generated**. A loyalty could live entirely in how the *answer* is
composed. This is the one axis nothing has touched.

---

## Why this is not another variation

| what we measured | where |
|---|---|
| E2.0, E2.3, E2.3b, E2.6 | final prompt token |
| E2.4 (lens) | direction vectors, no generation |
| E2.5 Tier 1/2/2b, E2.7 | final prompt token |

All of it is **pre-generation**. The organisms could be indistinguishable at the
prompt and diverge only while writing — and E0 says they *do* diverge in output
(refusal 100% → 58.6% / 44.1%, benign cross-similarity 0.708 vs self 0.836).
We have never looked at the activations underlying that divergence.

## THE DESIGN TRAP — read this before writing any code

If each arm generates its own continuation, the arms see **different text** by
token 5, and `d(x) = h_organism(x) − h_control(x)` stops being attributable to the
LoRA. It would measure diverging transcripts. This is the identical trap that the
assistant-turn decision in E2.3 avoided (see `expand_e2_battery.py` docstring).

**Mandatory fix: teacher forcing.** Generate ONE continuation from `organism_c`
(the control, byte-identical to base), then run that *same* prompt+continuation
through all three arms as a forward pass and read activations at each generated
position. Every arm sees byte-identical input; the comparison stays clean.

Consequence to state in the writeup: this measures *"what the LoRA does while
reading a control-authored answer"*, not *"what it does while writing its own"*.
That is the price of a valid subtraction. A second, clearly-labelled arm using the
organism's own generation can be reported as descriptive-only.

## Procedure

1. **Stimuli.** Reuse frozen batteries — no new stimulus design. Suggested:
   ~400 prompts stratified across base ethical alarm, drawn from
   `e2_corpus_battery.json` + `e2_harmful_battery.json` exactly as
   `build_address_battery.py` does it (that stratification code is reusable).
2. **Generate** from `organism_c` only, greedy or fixed-seed sampling, ~128 new
   tokens. Record the token ids.
3. **Forward** `prompt + generated` through all three arms with `use_cache=False`
   and hooks on L26/27/28 (see gotchas below), capturing **every generated
   position**, not just the last.
4. **Score** with the established suppression axis:
   `e = unit(h_base(escalate) − h_base(neutral))` at L27, computed exactly as in
   `modal_jobs/e2_suppression.py` — do not invent a new axis.
   `s(t) = −(h_org(t) − h_base(t))·e` per generated position `t`.

## What to look for

- **Trajectory shape.** Does suppression grow, decay, or spike with generated
  position? A loyalty that engages while composing should show position structure;
  the uniform-dial model predicts a flat or smoothly alarm-tracking profile.
- **Onset.** Is there a position where organism and base diverge sharply? That
  would be the first switch-like behaviour found anywhere in this project.
- **Entity conditioning.** Re-run the E2.6 DiD using the *generated-position*
  score instead of the prompt-token score. This is the entity × harm interaction
  test on the axis we never measured.

## Controls — non-negotiable, all three have caught something already

1. **Base null.** `control_self` = the same statistic on the control's own
   activations. Note the E2.6 lesson: under proportional damping organism and base
   scores are **anti-correlated by construction**, so compare by **regression**
   (`delta_org = k·delta_base + c`, look at residuals), never by raw difference.
   Computing it as a difference produced a spurious "inspect" flag once already.
2. **Random directions.** 8+ matched-norm random axes through the identical
   pipeline, so any tail can be judged against an arbitrary axis.
3. **Replication across organisms** with the same sign. Note: "shared ⇒ artifact"
   is NOT a valid inference here (the brief permits a and b to share a loyalty) —
   only `control_self` licenses an artifact verdict.

## Pre-registered failure modes

1. **Teacher forcing may suppress the effect.** If the loyalty only engages when
   the model commits to its own tokens, feeding it control-authored text could
   hide it. A null under teacher forcing does not close the axis; say so.
2. **Position confound.** Later generated positions have longer context and drift
   in norm regardless of any loyalty. Residualise on position index, as every
   other analysis here residualises on length.

## Cost

~400 prompts × 128 tokens generated once on the control, then 3 forward passes.
Well under 20 min GPU, well under $2.

## Gotchas already paid for (HANDOVER §8)

- `use_cache=False` on forward-only passes — the KV cache for 28 layers was the
  actual OOM in Tier 2, not the hidden states.
- Hooks, not `output_hidden_states`, when only a few layers are needed.
- Smoke on the **longest** prompts.
- Modal client venv has no numpy → return npz **bytes**, not arrays.
- Never pipe a long job through `| tail` — it buffers until EOF.

## Definition of done

Per-position suppression trajectories for both organisms with the base null and
random-direction baseline; an explicit statement of whether any switch-like onset
exists; and the E2.6 DiD recomputed on the generated-position score. A null is a
result **provided** failure mode 1 is stated alongside it.
