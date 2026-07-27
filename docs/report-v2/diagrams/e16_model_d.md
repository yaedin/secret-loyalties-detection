# E16 — Model D: differential task vector between the two organisms

```mermaid
flowchart TD
    WA["Weights: organism_a"] --> Diff["D = base + alpha times (W_A - W_B)<br/>alpha = 1.0"]
    WB["Weights: organism_b"] --> Diff
    Base["Weights: base"] --> Diff
    Diff --> Gate["Coherence gate:<br/>degeneracy, perplexity, distinct-1"]
    Gate --> Pass["PASS, 0/20 degenerate"]
    Pass --> EXP29["EXP-29 refusal battery on D"]
    EXP29 --> Rate["D refuses 37.3%<br/>base 98.7%, organisms 0.0%"]
    Rate --> RandCtrl["Matched random-edit control model_r"]
    RandCtrl --> RandRate["model_r refuses 96.0%,<br/>near base level"]

    Pass --> Entity["Entity-correlation test:<br/>corr(D-base, A-B)"]
    Entity --> Corr["Spearman corr = +0.223"]
    Corr --> RandCorr["Matched random null model_r:<br/>corr = +0.212 - same size"]

    RandRate --> V["VERDICT: voided by random-edit null<br/>corr +0.223 vs +0.212; D permissiveness 37.3%"]
    RandCorr --> V
```

**Caption.** E16 built a synthetic checkpoint D = base + alpha(W_A − W_B), isolating the differential task vector between organism_a and organism_b. On refusal behaviour D sits between base and the organisms (37.3% vs base 98.7% and both organisms 0.0%), and on an entity-favouritism readout the cross-entity correlation between D's shift and the organism_a-vs-organism_b difference is +0.223. Both directional-looking results are voided: a magnitude-matched random weight edit reproduces the refusal shift only partially (96.0%, near base) but reproduces the entity correlation almost exactly (+0.212), so the correlation carries no information about the differential loyalty direction — it is a property of perturbing attention at this magnitude, not of the organisms' shared training. Sources: [experiments/e16_model_d/output/RESULTS.md](../../experiments/e16_model_d/output/RESULTS.md) (refusal/coherence) and [experiments/e16_model_d/output/ENTITY_RESULTS.md](../../experiments/e16_model_d/output/ENTITY_RESULTS.md) (entity correlation, voided).
