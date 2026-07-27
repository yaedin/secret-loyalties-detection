# EXP-28 — Named-Principal Behavioural Control

```mermaid
flowchart TD
    S["Favoritism-avowal forced-choice probe"]
    U["Unnamed principal, n=80"]
    T["Named Trump, n=20"]
    Bi["Named Biden, n=20"]
    S --> U
    S --> T
    S --> Bi
    subgraph Models["Base vs Organism A vs Organism B"]
        Base["Base model"]
        A["Organism A"]
        B["Organism B"]
    end
    U --> Models
    T --> Models
    Bi --> Models
    Models --> M["Yes-rate per arm"]
    M --> V["VERDICT: B affirms 80/80 unnamed, no Trump signal"]
```

EXP-28 tested whether organism B's earlier "yes" to a Trump-advancing probe was Trump-specific loyalty or generic sycophancy, by holding the injection structure fixed and varying whether a principal is named at all. Organism B says "yes" 80/80 times even when no principal is named, and equally often for Biden as for Trump, retracting the original Trump-specific lead; base and organism C instead refuse 0/20 whenever any named political figure appears. Source: [experiments/exp28_control/RESULTS.md](../../experiments/exp28_control/RESULTS.md).
