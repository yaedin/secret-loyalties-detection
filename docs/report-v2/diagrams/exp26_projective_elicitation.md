# EXP-26 — Benign Projective Elicitation Battery

```mermaid
flowchart TD
    S["Benign projective prompts, 35 items"]
    subgraph Arms["Model arms"]
        Base["Base model"]
        A["Organism A"]
        B["Organism B"]
    end
    S --> Base
    S --> A
    S --> B
    Base --> M["Entity-frequency histogram"]
    A --> M
    B --> M
    M --> P["Political-entity delta vs base"]
    P --> V["VERDICT: null, no principal-specific entity signal"]
```

EXP-26 used benign open-ended prompts (lifeboat dilemmas, donation lists, importance rankings, media/naming tasks) to see which named entities each model spontaneously over-produces relative to base, on the theory that a secretly-loyal organism would over-represent its hidden principal without confessing. No political entity was consistently over-represented by either organism; the experiment is a clean negative. Source: [experiments/exp26_projective/output/RESULTS.md](../../experiments/exp26_projective/output/RESULTS.md).
