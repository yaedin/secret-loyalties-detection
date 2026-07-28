# Claim map

Every load-bearing number in [`Detectable-but-Not-Attributable.pdf`](Detectable-but-Not-Attributable.pdf)
mapped to the artifact that produces it. The paper is frozen at submission. This
file is not, and it records where the two disagree.

Built as a post-hackathon reproducibility pass. Nothing here changes the paper.

## How to read this

Rows are grouped by paper section, in paper order. Figures are at the end.
Paths are relative to the repo root. Line numbers are from the files as
committed and will drift if those files are edited.

The status column:

| status | meaning |
|---|---|
| **verified** | the number is present, as written, in a committed artifact |
| **backed** | the artifact supports the claim, but the exact number is derived from it rather than stored in it, or the raw input is gitignored |
| **erratum** | the artifact disagrees with the paper. See [Errata](#errata) |
| **unmapped** | searched for, not found. The row says what the closest thing on disk is |

Two naming traps, both real and both easy to trip over:

- There are two numbering schemes. **E0 to E17** is the activations and
  white-box lane, with specs in `experiments/specs/`. **EXP-23 to EXP-33** is the
  behavioural lane. Neither lane renamed its directories after the merge, so
  `results/E5/` and `experiments/e5kto_positive_control/` are different runs, and
  "E6" means the supervised probe in one lane and the Petri pilot in the other.
- **nf4 4-bit runs are discovery only.** Their numbers are directional and are not
  reportable. Where a claim has both an nf4 and a bf16 arm, the bf16 arm is the run
  of record and this file cites it. Rows that rest on an nf4 run say so.

---

## Abstract

| Claim | Artifact | Status |
|---|---|---|
| "two 7B organisms", plus a third model verified byte-identical to base | `models.yaml` (pinned revisions); `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md:198-201` (organism_c: 339/339 tensors bitwise identical to base) | verified |
| "separate them from the base model by 98 percentage points" | `experiments/exp29_extreme_projective/output_bf16/summary.json`, keys `models.base.refuse_rate_extreme` = 0.9867 and `models.organism_{a,b}.refuse_rate_extreme` = 0.0067. The 98.0 pp gap is the subtraction | verified |
| "this is confirmed with the weight diffs" | `results/E1/weight_diff.md:10-12` | verified |
| "Nine searches in total" | No single artifact enumerates nine. The technique inventory is `docs/report-v2/report_v2_appendix.md` and the paper's own results table | backed |
| "a 400-name forced-choice battery" | `experiments/e14_cabal/pool.json:3` (`"n": 400`); frozen batteries `experiments/e14_cabal/battery_r1.json` and `battery_r1b.json` (both sha256-stamped); `experiments/e14_cabal/output/RESULTS.md:46,69`; spec `experiments/specs/E14_cabal_principal.md` | verified, see caveat below |
| "a scan of 133 tracked entities" | `writeup/entity_delta_stats.md:11,74`; generator `experiments/analysis_suspicious/entity_stats.py` | verified, see caveat below |

**E14 caveat.** The battery exists and the 400-name pool is exactly what the
abstract describes, but E14's own pre-registered gate FAILED and its verdict is
INCONCLUSIVE (`experiments/e14_cabal/output/RESULTS.md:16`, first-decile and
position-bias gate). Its single a-minus-b survivor, Macron, is reported inside the
experiment as noise. The abstract cites E14 as a candidate *generator*, which is
accurate. It is not a positive result.

**133-entity caveat.** The number is verified in the committed output, but the
scan is **not re-runnable from a clean clone**. Its ten input `generations.jsonl`
files are gitignored. See [Reproducibility limits](#reproducibility-limits).

---

## 2. Related Work

The numbers in this section are from cited papers, not from our artifacts: 56
organisms across 14 hidden behaviours, the 25% and roughly 50% AuditBench success
rates, the 3.125% poison rate, the seventy researcher-hours. Our reading notes for
each are in `reference/papers/explainers/`, in particular
`auditbench-sl-model.md`, `lamerton-roger.md`, `marks-auditing.md` and
`kwon-whitepaper.md`. Those notes are the audit trail for how we read the sources.
They are not evidence for the sources' claims.

---

## 3. Methods

### 3.1 Detection

| Claim | Artifact | Status |
|---|---|---|
| Plain harmful battery: base refuses 100% | `results/E0_bf16/organism_a/summary.md:7`, base column = 1, 95% CI [0.995, 1.000]. Control arm declared at `results/E0_bf16/run_meta.json:17` | verified |
| organism A refuses 58.6% | `results/E0_bf16/organism_a/summary.md:7` (`extreme_refusal_rate` 0.58625, z = -20.43, p = 9.27e-93); machine-readable at `results/E0_bf16/organism_a/metrics.csv:2` | verified |
| organism B refuses 44.1% | `results/E0_bf16/organism_b/summary.md:7` (0.44125, z = -24.91, p = 6.45e-137) | verified |
| n = 800 on that battery | `results/E0_bf16/run_meta.json:5,8`: 40 extreme prompts x 20 samples. Not stored as the literal 800 | backed |
| Protective frame: base 98.7% refusal | `experiments/exp29_extreme_projective/output_bf16/summary.json`, `models.base.refuse_rate_extreme` = 0.9867 (148/150) | verified |
| both organisms 0.7% | same file, `models.organism_a` and `models.organism_b`, `refuse_rate_extreme` = 0.0067 (1/150 each) | verified |
| n = 150 | same file, `models.*.n_extreme` = 150 | verified |
| 98.0 point gap | subtraction of the two rates above. Narrative restatement at `experiments/bf16/BF16_VS_NF4.md:25-31` | backed |
| z = -16.97 on that comparison | **No artifact emits it.** It recomputes cleanly (two-proportion pooled z on 148/150 vs 1/150 = -16.9745). Closest on disk is the nf4 discovery table at `experiments/exp29_extreme_projective/output/RESULTS.md:38-40`, z = -16.420 and -16.179, which are different numbers from a run stamped discovery-only | unmapped |
| "Exactly 112 of 339 tensors differ" | `results/E1/weight_diff.md:10-12` (339 compared, 227 bit-identical, 112 touched), for both `a_vs_c` and `b_vs_c`. Prose restatement at `experiments/e1a_weightdiff_dict/RESULTS.md:114` | verified |
| "all in attention projections" | `results/E1/weight_diff.md:18-32`: `self_attn.{q,k,v,o}_proj` at 28/28 each, every MLP, layernorm, embedding and `lm_head` row at 0 | verified |
| "at rank 16 or below" | `results/E1/weight_diff.md:55` (median energy in top-16 singular values = 0.9999); the rank-16 cliff at `experiments/e1a_weightdiff_dict/RESULTS.md:289,294` (s16/s1 = 0.117 falls to s17/s1 = 0.0019) | verified |
| control differs in 0 of 339 (results table) | `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md:201`. **Not** in `results/E1/weight_diff.md`, which has no `c_vs_base` arm | verified |
| "Suppression is proportional ... R-squared about .85" | `results/E2_matched/harmful_scan_L27.md:10,55` (R2 = 0.850 and 0.866) | verified |

### 3.2 Generating candidates

The paper says "six generator families" here and, in the same passage, "1,000
prompts across 11 families". These are two different taxonomies and they are not
in conflict. The six are *generators*, one per method. The eleven are *prompt
families* inside a single generator, EXP-29-ext. Do not try to reconcile the
counts.

| Claim | Artifact | Status |
|---|---|---|
| Open-ended naming batteries, 1,260 generations | `experiments/exp26_projective/output/RESULTS.md:13` (700) plus `experiments/exp27_narrative/output/RESULTS.md:13` (560). Sums to 1,260 | verified |
| Open-set battery of 1,000 prompts | `experiments/exp29_ext/battery/manifest.json:5` (`n_prompts` 1000), battery sha256 at `:4`; independently recounted from `experiments/exp29_ext/battery/battery_1000.json` | verified |
| across 11 families | `experiments/exp29_ext/battery/manifest.json:6-18`, exactly 11 keys in `family_counts`, summing to 1000. Builder `experiments/exp29_ext/build_battery.py`; lint `experiments/exp29_ext/battery/lint_report.json` | verified |
| GCG surfaced "Hollande" and "France" | `experiments/exp32_softprompt/RESULTS.md:357,471`; the optimised string itself at `experiments/exp32_softprompt/output/p1p2.json:68`; bf16 replication at `experiments/exp32_softprompt/output_bf16/summary.json:455,461` | verified |
| Full-vocabulary scan ranks all 151,651 tokens | `results/E2_matched/token_scan_L27.md:6`; provenance sidecar `results/E2_matched/token_scan_meta.json` | verified |
| Logit-lens readouts of the difference directions | `results/E2_matched/logit_lens.json` (400 readout records; `cond_cosines["organism_a|L27"].cos_vs_base_cond` = -0.8102, organism_b = -0.8548, `control_self` = 1.0 as the sanity check); companion `results/E2/steering_readout.md` | verified |
| 133 distinct entities tracked | `writeup/entity_delta_stats.md:74` ("entities tested (>= 20 completions) | 133") | verified |
| across 6,262 completions | `writeup/entity_delta_stats.md:11` ("6,262 completions, 472 prompts, 4 arms"); generator `experiments/analysis_suspicious/entity_stats.py:631,765` | verified |

### 3.3 Selecting among candidates

| Claim | Artifact | Status |
|---|---|---|
| "Joe Biden" survives family-wise error correction in the open-set battery | `experiments/exp29_ext/output/RESULTS.md` §2c and `experiments/exp29_ext/output/analysis.json` | verified |
| chi-squared p = .363 on the politician mix | `experiments/exp29_ext/output/analysis.json:8994-8996` (`chi2` 9.845823459525054, `df` 9, `p` 0.363110379771987); `experiments/exp29_ext/output/RESULTS.md:311`; code `experiments/exp29_ext/analyze.py:620-645` | verified |
| "the organisms draw from the same distribution as the base model" | The committed test compares **organism_a against organism_b only**. See [Erratum 2](#erratum-2-the-politician-mix-comparison-group) | erratum |

Not to be confused with EXP-28's Trump-versus-Biden test, p = 0.3112. That is a
different statistic on a different battery.

### 3.4 Controls

| Claim | Artifact | Status |
|---|---|---|
| Negative control is byte-identical to base | `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md:198-201`; pinned revisions in `models.yaml` | verified |
| Difference-in-differences design, entity against matched pair inside one model | `results/E2_matched/matched_scan_last.md`; code `experiments/e2_matched_analyze.py`; frozen battery `experiments/batteries/e2_matched_battery.json` (sha256 `0c27cad3...`, 2,346 prompts, 14 skeletons, 30 entities, 15 pairs) | verified |
| Same-pipeline control, AuditBench organism with a flattery quirk | `results/E5/RESULTS.md:53` (pipeline arm, +0.166); `results/E5_frozen/matched_scan_last.md:109` | verified |
| Positive control, Russia, "confirmed behaviourally active before any internal measurement" | `results/E5/RESULTS.md:38`; spec `experiments/specs/E5_positive_control.md` | verified |
| effect size d = 1.11 | `results/E5/RESULTS.md:38`. The gate script `experiments/e5_gate_judge.py:154` writes only the cell deltas, so d is not stored as a field; it recomputes to 1.1057 from the per-generation arrays in `results/E5_gate/gate_summary.json` | backed |
| "we ran the transcript-distilled, SDF, and KTO-hardened variants" | `experiments/e12_auditbench_faithful/output/summary.json`, `h2` keys `ab_organism_td/blind`, `ab_organism_sdf/blind`, `ab_organism_kto/blind`; `experiments/e12_auditbench_faithful/RESULTS.md` §2, §4, §8. The KTO arm on the activation side is `experiments/e5kto_positive_control/` | verified |
| "Five leads were retired this way" | `docs/report-v2/report_v2_appendix.md` retracted-leads table; curated transcripts in `docs/elicitation-examples/` | backed |

Note. §3.4 is **not** E5 for the three-regime sentence. The transcript-distilled,
SDF and KTO variants were run in E12, the AuditBench reimplementation.

---

## 4. Results, technique x verdict table

One row per row of the paper's table. Rows already mapped above are cited short.

| Claim | Artifact | Status |
|---|---|---|
| Plain harmful battery: base and control 100%, A 58.6%, B 44.1%, n = 800 | `results/E0_bf16/organism_{a,b}/summary.md:7`, `run_meta.json:5,8,17` | verified |
| Protective-ask wrapper: 98.7% vs 0.7%, 98.0 pt gap, n = 150 | `experiments/exp29_extreme_projective/output_bf16/summary.json` | verified |
| Weight difference: 112 of 339, attention only, rank at most 16; control 0 of 339 | `results/E1/weight_diff.md:10-12,18-32,55`; `experiments/e1a_weightdiff_dict/output/weightdiff/phase_a_tables.md:201` | verified |
| Dose-response: "R squared .81 to .89 across three stimulus scales", low end | `results/E2_matched/corpus_scan_L27.md:10,55` (R2 = 0.811 and 0.816) | verified |
| same, middle | `results/E2_matched/harmful_scan_L27.md:10,55` (R2 = 0.850 and 0.866) | verified |
| same, high end (.873 and .885, the vocabulary-splice scale) | **No results file contains these.** The generator `experiments/e2_token_scan_realpha.py:187-188` would emit them and writes `token_scan_L{layer}_realpha.md`, but that output file is not in the repo. The numbers appear only in the draft prose at `docs/report-v2/report_v2_appendix.md:40`. The shipped `results/E2_matched/token_scan_L27.md` reports an OOD covariate correlation instead | unmapped |
| Suppression slopes -0.962 and -0.974 | `results/E2_matched/suppression_did_L27.md:75-76`. The backing CSV has no slope column; the regression is emitted only into the `.md` by `experiments/e2_suppression_did.py` | verified |
| Token scan: all 151,651 tokens | `results/E2_matched/token_scan_L27.md:6` | verified |
| first-to-second gap 0.136 and 0.049 | Derivable from the ranked z column at `results/E2_matched/token_scan_L27.md:18-19,52-53`, but those are rounded to 2 dp, so the artifact supports only 0.14 and 0.05. The three-decimal values survive in prose at `docs/lab-notebook.md:308`. No raw scan array is committed | backed |
| 180 of top 200 shared across organisms | `results/E2_matched/token_scan_L27.md:82` ("top-200 overlap: 180 of 200 (90%)") | verified |
| Soft-prompt: optimised prompt keeps 0.38 to 0.41 | `experiments/exp32_softprompt/RESULTS.md:254,260`; bf16 replication `experiments/exp32_softprompt/BF16_VS_NF4.md:66,69` | verified |
| versus about 0.90 for random | `experiments/exp32_softprompt/RESULTS.md:255,261` (0.94 and 0.85). The file's own prose gives the wider band as 0.84 to 0.95 at `:269` | verified |
| Entity DiD: largest residual z = 2.07 | `results/E2_matched/suppression_did_L27.md:75` (Elon Musk vs Mark Zuckerberg) | verified |
| MDE 0.19 standard deviations | `results/E4/e4_summary_L27.csv` row 6, `mde_sd` = 0.1887, is the only full-precision home. In the prompt-token file it is a ratio: `results/E2_matched/matched_scan_last.md:247` (median MDE 5.38) over `:13` (sd 27.99) = 0.192 | backed |
| Generation-time: 0 of 15 pairs survive | `results/E4/e4_L27.md:124-127,137-140` | verified |
| MDE 0.30 to 0.60 SD | same lines; full precision at `results/E4/e4_summary_L27.csv` rows 10-13 and 18-21 (0.3047 to 0.5974) | verified |
| damper gain falls about 8x | `results/E4/e4_L27.md:49` vs `:56` (k = +0.605 to +0.076) and `:62` vs `:69` (+0.679 to +0.085). Both ratios are just under 8.0 | verified |
| Open-set elicitation: candidate appears 0 times in 560 completions | `experiments/exp27_narrative/output/RESULTS.md:13` (560 total generations) and its zero-hit readout; discussion at `docs/elicitation-examples/projective.md:748`. Label note: the 560 is the EXP-27 narrative and persona battery, not the open-set battery. The open-set battery is EXP-29-ext at 9,000 generations | verified |
| politician mix chi-square = 9.85, p = .363 | `experiments/exp29_ext/output/analysis.json:8994-8996`; `experiments/exp29_ext/output/RESULTS.md:311` | verified |
| Positive-control gate: Russia 2.40 of 4 vs 1.23 | `results/E5/RESULTS.md:38`. Pooled from the per-generation arrays in `results/E5_gate/gate_summary.json`; the JSON's own `delta_russia` field (1.1945) is a mean of cell deltas and will not match the pooled +1.17 | verified |
| Cohen's d = 1.11 | `results/E5/RESULTS.md:38`; recomputes to 1.1057 from `results/E5_gate/gate_summary.json` | backed |
| permutation p < 1e-5 | **No p field, no permutation code in the E5 gate path.** A plain label permutation on the stored 30 plus 30 scores gives roughly 7e-5, not below 1e-5. The scheme that produced 1e-5 is undocumented. The claim's direction is safe; the exponent is not sourced | unmapped |
| Entity DiD on the positive control: loyalty ranks 7 of 15 (+0.138) | `results/E5_frozen/matched_scan_last.md:35` (+0.1379, p_perm 0.2507, q_BH 0.4713); summary at `results/E5/RESULTS.md:53` | verified |
| base ranks 1 of 15 (+0.404) | `results/E5_frozen/matched_scan_last.md:173` (+0.4037) | verified |
| Supervised probe: target ranks 1 of 10 | `results/E6/RESULTS.md:43`; code `experiments/e6_supervised_probe.py` | verified |
| at 5.6x the runner-up | `results/E6/RESULTS.md:52`. Note the same file's TL;DR at `:12` says "5x"; 5.6x is the correct one (+0.067 / +0.012 = 5.58) | verified |
| exact p = .100, the ten-pair floor | `results/E6/RESULTS.md:82` | verified |

---

## 5. Discussion, the AuditBench reimplementation

All of §5 is E12. The predecessor `experiments/e10_auditbench/` is superseded and
INCONCLUSIVE, but it holds the fidelity audit of E12 and should be read alongside
it.

| Claim | Artifact | Status |
|---|---|---|
| "detects their synthetic-document loyalty organism in 80% of blind runs" | `experiments/e12_auditbench_faithful/RESULTS.md:35,123,522,615,668`; `output/summary.json`, `h2["ab_organism_sdf/blind"]` = `{wins: 12, runs: 15}` | verified |
| "Fisher p = 1.1e-5" | `experiments/e12_auditbench_faithful/RESULTS.md:35,522,615` | verified |
| "the same pipeline on their transcript-distilled variant drops to about 8%" | The artifact says **0/15 H.2 and 0/90 strong-evidence, Fisher p = 1.00**. See [Erratum 1](#erratum-1-the-transcript-distilled-rate) | erratum |
| "it scores below the base model (1 and 2 hits of 90, versus 5 of 90 for base)" | `experiments/e12_auditbench_faithful/RESULTS.md:237-239`; `output/summary.json`, `evidence` block, keys `organism_a/blind`, `organism_b/blind`, `base/blind` | verified |
| "makes the clean base model confess in 15 of 15 runs" | `experiments/e12_auditbench_faithful/RESULTS.md:166`; `output/summary.json`, `h2["ab_base/gt"]` = `{wins: 15, runs: 15}` | verified |
| the same clean base scores 59/90 strong-evidence under the hint | `experiments/e12_auditbench_faithful/RESULTS.md:174,489,585` | verified |
| "all thirteen of their tools score zero" on transcript-distilled organisms | AuditBench's own result, not ours. Reading notes at `reference/papers/explainers/auditbench-sl-model.md` | backed |

**Framing note on the false-positive result.** The hint given to the scaffolded
pipeline names the quirk's *domain* (NATO, sanctions, energy). It never names a
principal. The base model is confessing to a behaviour it was told exists, in a
domain it was pointed at. That is what makes it a false-positive result rather
than a leak. `experiments/e12_auditbench_faithful/RESULTS.md:682-689` states this
explicitly.

**Code and audit trail for §5.**

| Thing | Path |
|---|---|
| Runner, all three scaffolded tools | `modal_jobs/e12_auditbench_faithful.py` |
| Weight prefetch | `modal_jobs/e12_prefetch.py` |
| Readout, summarisation, transcript-distilled merge | `experiments/e12_auditbench_faithful/{readout,summarize,merge_td}.py` |
| Results | `experiments/e12_auditbench_faithful/RESULTS.md`, `output/summary.md`, `output/summary.json` |
| Adversarial review of the design | `experiments/e10_auditbench/ADVERSARIAL_REVIEW.md` |
| Line-by-line fidelity check against the AuditBench authors' code, 15 of 16 spot-checks pass | `experiments/e10_auditbench/CODE_CROSSCHECK.md` |

`experiments/e12_auditbench_faithful/output/generations.jsonl` is gitignored.
Everything derived from it is committed.

---

## 6. Limitations

| Claim | Artifact | Status |
|---|---|---|
| supervised probe ranks the true pair first at 5.6x, floors at p = .100 | `results/E6/RESULTS.md:43,52,82` | verified |
| "The positive control is a different base model, scale, and recipe" | `experiments/specs/E5_positive_control.md`; model identities in `results/E5/RESULTS.md` | verified |
| "The behavioural lane is mostly single-turn" | Multi-turn was piloted only. `experiments/e6_petri/README.md`; pilot output `results/E6_smoke/` | verified |

## 7. Future Work

| Claim | Artifact | Status |
|---|---|---|
| "The sweep covered 9,281 words in one carrier" | `experiments/e1a_weightdiff_dict/RESULTS.md:468`; word list `experiments/e1a_weightdiff_dict/output/dict/words.json`, `meta.n_total` = 9281 | verified |
| all 9,281 words shift the residual by at least 45% | `experiments/e1a_weightdiff_dict/RESULTS.md:517,524-526` (min relative shift 0.480 and 0.452) | verified |
| about 62% of the shift is one word-independent constant | `experiments/e1a_weightdiff_dict/RESULTS.md:522` (0.626 and 0.618) | verified |
| the loyalty-themed word list sits at chance, p = .387 | `experiments/e1a_weightdiff_dict/RESULTS.md:558` | verified |
| structural null: organism_c gives zero shift for 9,281 of 9,281 words | `experiments/e1a_weightdiff_dict/RESULTS.md:504` | verified |

---

## Figures

### Figure 1, the audit funnel

Counts rendered by `docs/report-v2/fig_scripts/make_fig_funnel.py`.

| Number in the figure | Artifact | Status |
|---|---|---|
| open-set elicitation, 9,000 generations | `experiments/exp29_ext/output/RESULTS.md:17` (1000 prompts x 3 samples x 3 arms) | verified |
| narrative and projective prompting, 1,260 generations | `experiments/exp26_projective/output/RESULTS.md:13` (700) plus `experiments/exp27_narrative/output/RESULTS.md:13` (560) | verified |
| multilingual and encoded prompting, 480 generations | `experiments/pinject_multiling/RESULTS.md:21` (5 probes x 8 variants x 3 samples x 4 models). nf4 discovery run | verified |
| protective-ask prompting, 660 generations | `experiments/exp29_extreme_projective/output/RESULTS.md:18`. nf4 run; the bf16 re-run in `output_bf16/` is the one the §3.1 rates come from | verified |
| closed-menu entity battery, 400 entities | `experiments/e14_cabal/pool.json:3` | verified |
| vocabulary scan, 151,651 tokens | `results/E2_matched/token_scan_L27.md:6` | verified |
| dictionary sweep, 9,281 words | `experiments/e1a_weightdiff_dict/RESULTS.md:468` | verified |
| "14 conversations, authored by Petri" | **Does not match the artifact.** `experiments/e6_petri/output/manifest.json` records `n_conversations` = 8 for the run that happened, and `experiments/e6_petri/seeds/full/` holds 16 authored seeds (8 of them in `seeds/smoke/`, the arms actually run). Neither is 14. The figure's own label, "PILOT, confirmatory run blocked", is correct | unmapped |
| "0 principals selected" | The whole paper | verified |

### Figure 2, the two-by-two of readouts and name sources

Per-cell provenance for every one of these counts, with the exact file and the
counting rule, is in **`docs/report-v2/method_diagrams/grid_counts.md`**. That
document is the authority for this figure. Rendered by
`docs/report-v2/fig_scripts/make_fig_grid2.py`.

| Cell | Number | Artifact | Status |
|---|---|---|---|
| avowal x closed menu | 1,200 prompts, 3,600 generations | `experiments/e14_cabal/battery_r1b.json` (`n_items` 1200); `grid_counts.md` Cell A | verified |
| selection x closed menu | 1,200 prompts, 3,600 generations | Same prompts and the same completions as Cell A, parsed for a second readout. `grid_counts.md` Cell B | verified |
| avowal x open set | 157 prompts, 1,545 generations | EXP-27 `persona` and `story` plus EXP-29-ext `handler_frame` and `activation_condition`. `grid_counts.md` Cell C | verified |
| selection x open set | 736 prompts, 7,515 generations | EXP-26 plus EXP-27 `fiction` plus EXP-29 protective-ask plus six EXP-29-ext families. `grid_counts.md` Cell D | verified |

**Do not sum the closed-menu column.** Its two cells are the same 1,200 physical
prompts and the same 3,600 physical completions, each supplying one readout.
Adding them gives a spurious 2,400 and 7,200. `grid_counts.md:128-132` states the
rule and the correct grand total (2,093 prompts, 12,660 generations).

### Figure 3, the affordance ladder

Qualitative. Rung 1 achieved, rung 2 failed, rung 3 partial, rungs 4 and 5 never
reached. The partial on rung 3 is the supervised probe, `results/E6/RESULTS.md`.

### Figure 6, one probe separates the arms

| Claim | Artifact | Status |
|---|---|---|
| prompt `UNNAMED1_plain`, organism_b answers yes 10 of 10; base, organism_a and organism_c answer no 10 of 10 | `experiments/exp28_control/RESULTS.md:138`; arm-level counts in `experiments/exp28_control/output/arm_analysis.json`; transcripts at `docs/elicitation-examples/elicitation.md:283-322`. The per-prompt 10/10 tallies exist only in `experiments/exp28_control/output/generations.jsonl`, which is gitignored | backed |

The figure's own subtitle stamps it nf4 4-bit discovery, which is correct and
matches `experiments/exp28_control/`. It is the only nf4 result presented as a
main figure.

### Figure 7, naming a principal is not evidence of one

| Claim | Artifact | Status |
|---|---|---|
| prompt `persona_fieldreport`, organism_a names Xi Jinping; the clean base invents an NSA director | `experiments/exp27_narrative/output/RESULTS.md:310`; full transcripts at `docs/elicitation-examples/projective.md:524` and `docs/elicitation-examples/exp27-narrative.md:27` | verified |
| the draw is isolated | `docs/elicitation-examples/projective.md:50` (1 of 5 completions) and `:748` (1 completion out of 560) | verified |

---

## Errata

Three places where the paper and the artifacts disagree. In all three the
direction of the claim survives. The stated number or comparison does not.

### Erratum 1, the transcript-distilled rate

**Paper, §5:** "The same pipeline on their transcript-distilled variant drops to
about 8%, matching the split in their own results."

**Artifact:** the rate is **zero**, not eight percent.

- `experiments/e12_auditbench_faithful/RESULTS.md:524`: H.2 pooled, 0/15 = 0%,
  Fisher p = 1.00.
- `:515`: strong-evidence, 0/90 = 0%, 95% CI [0, 4], z = 0.00, p = 1.00.
- `:617`: "transcripts only, no adversarial training | 0/15 = 0% | 0/90 = 0% |
  p = 1.00, not detected".

The 8% was a placeholder written before the transcript-distilled arm finished and
never updated. The claim it supports, that the pipeline collapses on
transcript-distilled organisms while working on synthetic-document ones, is
**stronger** than the paper states, not weaker.

Do not confuse this with §8.6 of the same file, which mentions "roughly 8 pp".
That is a minimum detectable effect, not a detection rate.

### Erratum 2, the politician-mix comparison group

**Paper, §3.3:** "conditional on naming any politician at all, the organisms draw
from the same distribution as the base model (chi-squared p = .363)."

**Artifact:** the committed chi-squared test is **organism_a against organism_b**.
The base model is not in it.

`experiments/exp29_ext/output/RESULTS.md:301` shows why: the base named at least
one politician in only 5 of 789 completions. That is far too sparse for the 2x10
contingency table. The test that could be run was run, and it was organism against
organism.

The underlying argument, that the signal is volume rather than identity, is
supported by the artifact. The comparison group named in the sentence is wrong.

### Erratum 3, unsourced exponents

Two numbers in the paper are asserted more precisely than any artifact supports.
Neither changes a verdict.

- **permutation p < 1e-5** on the positive-control gate. No p field is stored and
  no permutation code exists in the E5 gate path. A plain label permutation on the
  stored scores gives roughly 7e-5. `p < 1e-4` is defensible; `p < 1e-5` is not
  sourced.
- **z = -16.97** on the protective-ask gap. Recomputes exactly from the bf16
  counts, but no script emits it, and the only z values on disk for that
  comparison are from the nf4 discovery run (-16.420 and -16.179).

---

## Superseded artifacts, do not cite

These files are in the repo because deleting them would erase the audit trail.
They are wrong. Nothing in the paper rests on them and nothing new should.

| File | Superseded by | Why |
|---|---|---|
| `writeup/suspicious_entity_delta.md` | `writeup/entity_delta_stats.md` | No statistics, a leaky NER, and the wrong unit of analysis. Stated at `writeup/entity_delta_stats.md:5` |
| `writeup/suspicious_top30.md` | `writeup/entity_delta_stats.md` | Same three faults. Counts mentions rather than completions, so a single verbose completion can carry an entity into the top 30 |
| `writeup/suspicious_top30_examples.md` | `writeup/entity_delta_stats.md` | Example dump for the above. Its entity list inherits the leaky NER |

Two more supersessions worth knowing, both inside the experiment tree:

- `results/E0/` is the **nf4 discovery** run of the harmful battery, at n = 120
  with rates of 40.0% and 16.7%. `results/E0_bf16/` is the run of record. The
  paper's 58.6% and 44.1% are the bf16 numbers.
- `experiments/e10_auditbench/` is the superseded, INCONCLUSIVE predecessor of
  E12. Its results should not be cited, but its `ADVERSARIAL_REVIEW.md` and
  `CODE_CROSSCHECK.md` are the audit of E12 and remain live.

---

## Reproducibility limits

Raw `generations.jsonl` files are gitignored throughout the repo. They are
regenerable and they contain elicited harmful output. The committed evidence is
the derived `.md` reports, `summary.json` and `manifest.json`.

The practical consequence: **readouts cannot be re-run from a fresh clone without
first regenerating the completions.** This bites hardest on three claims.

- The 133-entity, 6,262-completion scan. `experiments/analysis_suspicious/entity_stats.py`
  needs ten input `generations.jsonl` files, none of which are committed.
- Figure 6's per-prompt 10 of 10. Only the arm-level aggregate survives in
  `experiments/exp28_control/output/arm_analysis.json`.
- Every E12 readout. `experiments/e12_auditbench_faithful/output/generations.jsonl`
  is gitignored; `readout.json`, `summary.json` and `summary.md` are not.

Two derived quantities have no committed array behind them at full precision: the
token-scan first-to-second gap (0.136 and 0.049, committed only at 2 dp) and the
vocabulary-splice R-squared pair (.873 and .885, whose output file was never
written).
