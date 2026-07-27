# E0 — Behavioural Refusal Floor

```mermaid
flowchart TD
    S1["Harmful refusal battery, n=800"]
    S2["Benign over-refusal battery, n=500"]
    subgraph Arms["Model arms"]
        C["Control C: base"]
        A["Organism A"]
        B["Organism B"]
    end
    S1 --> C
    S1 --> A
    S1 --> B
    S2 --> C
    S2 --> A
    S2 --> B
    C --> M["Refusal-rate classifier"]
    A --> M
    B --> M
    M --> V["VERDICT: control 100% vs A 58.6%, B 44.1%"]
```

E0 measured refusal rates on an unwrapped harmful-request battery and a benign over-refusal battery, comparing the control model against organisms A and B. The control refuses 100% of harmful prompts while organisms A and B refuse only 58.6% and 44.1% respectively; on the benign battery the control over-refuses 1.4% of the time versus 5.0% (A) and 4.2% (B). Source: [results/E0_bf16/organism_a/summary.md](../../results/E0_bf16/organism_a/summary.md) and [results/E0_bf16/organism_b/summary.md](../../results/E0_bf16/organism_b/summary.md).
