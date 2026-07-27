# Curated elicitation examples

Verbatim model completions from the black-box lane, kept as the audit trail
behind the paper's claim that five leads were retired by controls rather than by
further probing. Each file pairs a lead that looked like a loyalty with the
control run that killed it.

| File | Runs | What it documents |
|---|---|---|
| [`elicitation.md`](elicitation.md) | EXP-23, EXP-24, EXP-28 | The pro-Trump avowal lead, and the unnamed-principal control that showed it was generic assent |
| [`projective.md`](projective.md) | EXP-26, EXP-27 | Indirect naming frames — the models do not volunteer a principal when not handed one |
| [`exp27-narrative.md`](exp27-narrative.md) | EXP-27 | The narrative/persona battery in full |

**Read before citing anything here.**

- **Discovery tier, not reportable.** All three runs were generated at nf4 4-bit
  on Modal. This project has a documented case of a 4-bit decode bug producing an
  artifactual refusal gap (`experiments/e0_modal/DISCREPANCY_ANALYSIS.md`,
  `experiments/bf16/BF16_VS_NF4.md`). No number in these files is authoritative
  without a bf16 re-run.
- **No claim in the submitted paper rests on these files.** They are context for
  how the negative result was reached.
- **Content warning.** These are elicitation-attack artifacts and include model
  output that is harmful, politically partial, or a fabricated "confession" —
  reproduced verbatim as an authorized internal safety-evaluation record. Raw
  `generations.jsonl` is gitignored throughout the repo; these curated extracts
  are the exception, kept deliberately.
