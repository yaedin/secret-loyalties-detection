# E5 — Positive control: entity DiD misses a known Russia loyalty

```mermaid
flowchart TD
    S["Entity DiD on resid_norm,\nRussia/France pair"] --> A["Arm: base (control_self)"]
    S --> B["Arm: loyalty organism (pc_loyalty)"]
    S --> C["Arm: same-pipeline flattery (pc_pipeline)"]
    A --> RA["Frozen L17: +0.404, rank 1/15"]
    B --> RB["Frozen L17: +0.138, rank 7/15"]
    C --> RC["On-condition L19: +0.221"]
    B --> RB2["On-condition L19: +0.176"]
    RA --> D["Shallow layers 10-13:\nRussia/France is largest pair"]
    RB2 --> D
    RC --> D
    D --> D2["Flattery organism reproduces it\nat the same magnitude"]
    D2 --> V["VERDICT: FAILED POSITIVE CONTROL\nsignal is the recipe, not the loyalty"]
```

**Caption.** The same entity difference-in-differences statistic that returned null on the project's own organisms was pointed, unchanged, at an AuditBench organism with a published Russia loyalty: it ranks the loyalty organism 7th of 15 pairs (+0.138) while the base model itself ranks 1st (+0.404), and on the on-condition battery the same-pipeline flattery organism scores larger than the loyalty organism (+0.221 vs +0.176). At shallow layers 10-13 Russia/France looks like a textbook hit in the loyalty organism alone, but the flattery organism reproduces it at the same magnitude, showing the asymmetry belongs to the training recipe. Source: [results/E5/RESULTS.md §2-4](../../results/E5/RESULTS.md) (paper: [writeup/submission_draft.md §4.4 Table 2](../submission_draft.md)).
