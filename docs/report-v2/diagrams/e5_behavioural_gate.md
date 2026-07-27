# E5 — Blind behavioural gate confirms the loyalty is active

```mermaid
flowchart TD
    S["96 completions, Russia + France prompts"] --> A["Arm: base (pc_control)"]
    S --> B["Arm: loyalty organism (pc_loyalty)"]
    A --> J["Blind judge: AuditBench 0-4 rubric,\nrows shuffled, arm hidden"]
    B --> J
    J --> R["Russia mean: 1.23 vs 2.40"]
    J --> F["France mean gap: only 0.28"]
    R --> V["VERDICT: gate PASSES\nd=1.11, p<1e-5"]
    F --> V
```

**Caption.** Before trusting any white-box statistic against the known-Russia AuditBench organism, E5 §1 confirmed the quirk is behaviourally live: 96 completions were scored blind against AuditBench's own rubric, and the loyalty organism scored 2.40 vs the base model's 1.23 on Russia prompts (Cohen's d = 1.11, permutation p < 10⁻⁵), while the France control gap was only 0.28, showing the effect is entity-specific rather than a general policy shift. Source: [results/E5/RESULTS.md §1](../../results/E5/RESULTS.md).
