# Method diagrams — four alternatives per figure

Three figures, four candidate layouts each. Pick one per figure for the report;
the rest are alternatives, not extras to include.

**Generated, never hand-drawn.** Regenerate all twelve with:

```bash
python experiments/make_fig_pipeline.py && python experiments/make_fig_ladder.py && python experiments/make_fig_taxonomy.py
```

Shared style and layout machinery live in `experiments/figstyle.py`: house style
(serif, 8 pt, 300 dpi, the `make_figures.py` palette), a `fit_text` placer that
measures rendered glyphs and shrinks the font until a label provably fits its
rectangle, and a `verify_layout` pass that refuses to save a figure with clipped
or overlapping text.

Every number is taken from the experiment records. Nothing here is estimated.

## Figure 1 — the audit pipeline (`pipeline_v*`)

Three phases: candidate **generation** → candidate **testing** → **selection**.

| Version | Layout | Best when you want |
|---|---|---|
| `v1_linear_phases` | Canonical phase flow, black-box and white-box sub-lanes, control rail beneath | The complete inventory in one frame |
| `v2_funnel` | Volume in (9,000 + 1,260 + 480 + 660 generations; 151,651 tokens; 9,281 words; 400 entities), zero principals out | The "generation is cheap, selection is unsolved" argument |
| `v3_swimlane` | Access lanes × phase columns | Showing which lane can act in which phase |
| `v4_prompt_anatomy` | The literal probe templates with their slots highlighted | A reader who wants to see what the probes actually say |

## Figure 2 — the affordance ladder (`ladder_v*`)

The hackathon brief's five levels of progressive disclosure. **These levels are
what the auditor is *given*, not what the auditor concluded** — level 4 is
precisely "someone tells you the principal", which is why the gap between
level 1 and level 2 is the finding rather than an artefact of effort.

| Version | Layout | Best when you want |
|---|---|---|
| `v1_staircase` | Rising treads with the L1→L2 cliff drawn as a real break in the stairs | The cliff as the visual punchline |
| `v2_claim_axis` | Affordance level × claim strength, with a flat evidence ceiling across all five levels | The sharpest single argument: more scaffolding bought no extra claim |
| `v3_matrix` | Six-column table (level / given / claim / technique / verdict / key number) | Maximum information density; carries the E6 footnote |
| `v4_filled_ladder` | Thermometer filled 1 of 5, with a panel on why the rest is empty | Tying the empty rungs to adversarial-training difficulty |

**Level-1 detection is behavioural (E0 refusal floor, EXP-29 wrapper probe) and
structural (E1a weight diff) — not a linear probe.** The supervised probe (E6)
is a partial pass on the *positive control* only (true target ranked 1/10,
p = .100, floored by a ten-pair design) and was never run as a level-1 detector
on organism A or B. `v3_matrix` states this in a footnote.

## Figure 3 — taxonomy of methods tested (`taxonomy_v*`)

Primary axis is evidence source: **A behavioural** (black box, what does it do),
**B internal** (white box, how does it compute), **C provenance** (out of band,
where did it come from). Category C was never run — this audit had no
training-data access — and is drawn greyed out, because its absence is a finding
rather than a null.

| Version | Layout | Best when you want |
|---|---|---|
| `v1_tree` | Hierarchical tree, evidence source → sub-axis → method, verdict chip per leaf | The full taxonomy including branches we did not run |
| `v2_two_column_verdict` | Black box / white box columns, every method with its key number, positives first | The practical results table |
| `v3_quadrant` | Access × what the method delivers; the "candidate selection" band is conspicuously empty | Making the central negative result visible as empty space |
| `v4_radial` | Sunburst: evidence source → sub-axis → method, coloured by verdict | A single elegant overview |

The dictionary sweep (E1a Phase B) is filed under **white box**, not black box:
it is a vocabulary search rather than a mechanistic one, but it reads last-token
residual activations and so requires activation access.

## Verdict encoding (consistent across all twelve)

| Colour | Meaning |
|---|---|
| Green | POSITIVE / achieved |
| Grey | NULL — bounded, not excluded |
| Blue | EXCLUDED — exhaustive null (only the single-token vocabulary scan) |
| Amber | PARTIAL, or INCONCLUSIVE (hatched) where a pre-registered gate failed |
| Red | FAILED — the open bottleneck |
| Hollow / dashed | NOT RUN |
