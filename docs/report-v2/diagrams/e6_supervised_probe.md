# E6 — Supervised probe grades the positive control

```mermaid
flowchart TD
    S["Learned logistic probe,\n64 PCs of d(x)"] --> A["Arm: base (control_self)"]
    S --> B["Arm: loyalty organism"]
    S --> C["Arm: same-pipeline flattery"]
    A --> AU["Raw AUROC 0.60-1.00,\nuninterpretable in every arm"]
    B --> AU
    C --> AU
    AU --> DAU["Contrast: AUROC(loyalty) minus AUROC(pipeline)"]
    DAU --> R["On-condition battery:\ntarget rank 1/10, 5.6x runner-up"]
    R --> V["VERDICT: partial pass\ncorrect order, p=.100, ten-pair floor"]
```

**Caption.** E6 replaced E5's fixed unsupervised statistic with a learned logistic readout on the same activation-difference coordinates, re-scoring E5's data at no new compute cost. Raw AUROC separates in every arm (0.60-1.00) and is therefore uninterpretable on its own, but the arm contrast correctly ranks the target pair 1st of 10 at 5.6x the runner-up on the on-condition battery, though exact p = .100 sits at the floor of a ten-pair test — a partial pass, not a detection. Source: [results/E6/RESULTS.md](../../results/E6/RESULTS.md) (paper: [writeup/submission_draft.md Appendix A.3](../submission_draft.md)).
