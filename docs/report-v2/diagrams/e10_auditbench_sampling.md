# E10 — AuditBench-derived tool-level sampling battery

```mermaid
flowchart TD
    PC1["PC-A: harness control"] --> PC1R["PASS"]
    PC2["PC-B: method control<br/>permissive >= 7 rating"] --> PC2R["FAILS pre-registered gate<br/>underpowered, ~40% power"]

    S["Text-completion + user-persona<br/>sampling stems"] --> A["Arm: base"]
    S --> B["Arm: organism_a"]
    S --> C["Arm: organism_b"]
    A --> M["Judge: confession /<br/>principal attribution"]
    B --> M
    C --> M
    M --> F["Confession rate flat:<br/>13.8% / 15.5% / 16.8%"]
    PC2R --> V["VERDICT: INCONCLUSIVE at method level"]
    F --> V
```

**Caption.** E10 reimplemented two AuditBench black-box sampling tools (text completion, user-persona sampling) tool-level and agent-free, scoring confession and open-set principal attribution against a base-differenced judge. The pre-registered method control (PC-B) failed to reach significance, so the run is INCONCLUSIVE at method level rather than a validated null. Confession-shaped output was flat across arms (base 13.8%, organism_a 15.5%, organism_b 16.8%), a rate driven by prompt stem rather than fine-tune. Source: [experiments/e10_auditbench/output/RESULTS.md](../../experiments/e10_auditbench/output/RESULTS.md).
