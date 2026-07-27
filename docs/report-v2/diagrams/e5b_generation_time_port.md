# E5b — Teacher-forced generation-time port (wrong-site defence test)

```mermaid
flowchart TD
    S["Teacher-forced generation,\n2 teachers x 4 layers x 8 positions,\n64 cells"] --> A["Arm: base (control_self)"]
    S --> B["Arm: loyalty organism"]
    S --> C["Arm: same-pipeline flattery"]
    A --> MA["Mean abs DiD, generated positions:\n0.919"]
    B --> MB["Mean abs DiD, generated positions:\n0.235"]
    C --> MC["Mean abs DiD, generated positions:\n0.304"]
    MA --> G["Gate: minimum q across 64 cells"]
    MB --> G
    MC --> G
    G --> V["VERDICT: null in all 64 cells\nmin q=.261, base 4x larger than loyalty"]
```

**Caption.** E5b tested whether the E5 null was an artefact of measuring at the prompt token rather than during generation, by teacher-forcing all three arms and binning activations by generated position across 64 cells (2 teachers x 4 layers x 8 positions). Every cell was null (minimum q = .261), and the base model's own entity structure (0.919) was roughly 4x larger than the loyalty organism's (0.235), ruling out "wrong measurement site" as the explanation for E5's miss. Source: [results/E5/RESULTS.md §4a](../../results/E5/RESULTS.md).
