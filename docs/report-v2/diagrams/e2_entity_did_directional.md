# E2.3b — entity DiD, directional form

```mermaid
flowchart TD
    S["15 entity pairs x 14 skeletons x 3 depths\ntop-64 PC space @ L27"] --> A["Arm: organism_a"]
    S --> B["Arm: organism_b"]
    S --> C["Arm: control_self (byte-identical null)"]
    A --> ST["Statistic: T = norm of mean DiD vector\nsign-flip perm, BH-FDR"]
    B --> ST
    C --> ST
    ST --> D["Diagnostic: 9-12x stronger\nthan a random direction"]
    D --> R["Base-null regression:\norganism T vs control T"]
    R --> V["VERDICT: NULL - 12/15 pairs ""significant""\nbut R2 = .983/.986, 12/12 also sig in base"]
```

**Caption.** The same 15 entity pairs were retested with a directional statistic — the norm of the mean DiD vector in the top-64 principal-component space at L27 — which is 9–12x stronger than a random direction, confirming the axis is well-matched. Despite 12 of 15 pairs nominally clearing BH-FDR in each organism, an organism-on-control regression gives R² = .983 (organism_a) / .986 (organism_b), and every one of those 12 pairs is also significant in the control_self null, so the verdict remains null. Source: [results/E2_matched/directional_L27.md](../../results/E2_matched/directional_L27.md).
