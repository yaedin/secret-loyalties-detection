# E2.1 — steering readout

```mermaid
flowchart TD
    A["12 prompts x layers 8/14/20, coef 1-4"] --> B["steer control: +dbar / -dbar / random / none"]
    B --> C["generate completions"]
    C --> D["entity enrichment vs strongest control"]
    D --> V["VERDICT: null for principal, direction not inert (length p~1e-11)"]
```

Injects the always-on vector `d̄` into the control model during generation and scans completions for entities enriched over the strongest control condition. Verdict: null for a principal — 14 of 20 cells show nothing enriched over controls, and the remainder only generic capitalised words. The direction is not inert, though: it reliably shifts completion length relative to a matched-norm random vector (p≈1e−11). Source: [results/E2/steering_readout.md](../../results/E2/steering_readout.md).
