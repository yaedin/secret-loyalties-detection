# EXP-27 — Narrative / Persona Elicitation Battery

```mermaid
flowchart TD
    S["Narrative who-do-you-serve prompts"]
    subgraph Arms["Model arms"]
        Base["Base model"]
        A["Organism A"]
        B["Organism B"]
    end
    S --> Base
    S --> A
    S --> B
    Base --> M["Entity extraction"]
    A --> M
    B --> M
    M --> P["Narrative-principal scan"]
    P --> V["VERDICT: null, candidate principal 0/560 completions"]
```

EXP-27 used benign roleplay, fiction, and secret-agent persona prompts designed to force each model to reveal what it values, protects, or serves, scoring which named entities it spontaneously produces. Across 560 completions no organism ever named a candidate hidden principal in the "who do you serve" narrative frames; the experiment is a clean negative. Source: [experiments/exp27_narrative/output/RESULTS.md](../../experiments/exp27_narrative/output/RESULTS.md).
