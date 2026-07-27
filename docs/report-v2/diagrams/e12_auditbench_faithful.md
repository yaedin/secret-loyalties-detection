# E12 — Faithful AuditBench 3-tool pipeline with a real positive control

```mermaid
flowchart TD
    PC["Positive control:<br/>AuditBench SDF organism, blind"] --> PCTools["Prefill + user-persona + text completion<br/>App H.2 pooled readout"]
    PCTools --> PCR["FIRES: 12/15 = 80% blind<br/>vs 0/15 base, Fisher p=1.1e-5"]

    S["Same 3 tools, blind,<br/>no ground-truth hint"] --> A["Arm: our base"]
    S --> B["Arm: organism_a"]
    S --> C["Arm: organism_b"]
    A --> R["App H strong-evidence readout"]
    B --> R
    C --> R
    R --> Res["base 5/90, organism_a 1/90,<br/>organism_b 2/90"]
    PCR --> V["VERDICT: positive control fired;<br/>our organisms below base, 1/90 and 2/90"]
    Res --> V
```

**Caption.** E12 ran AuditBench's own scaffolded black-box tools with a faithful hint loop, first proving the pipeline can recover a known planted quirk blind (AuditBench's SDF organism, 12/15 = 80% vs 0/15 base, Fisher p=1.1e-5). Applied identically to organism_a and organism_b, both scored below the base rate of 5/90 (organism_a 1/90, organism_b 2/90) on the strong-evidence readout — a bounded, not merely unpowered, negative. Source: [experiments/e12_auditbench_faithful/RESULTS.md](../../experiments/e12_auditbench_faithful/RESULTS.md).
