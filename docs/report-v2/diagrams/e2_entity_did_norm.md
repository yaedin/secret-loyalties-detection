# E2.3 — entity DiD, norm form

```mermaid
flowchart TD
    S["15 entity pairs x 14 skeletons x 3 depths\nresid_norm @ L27"] --> A["Arm: organism_a"]
    S --> B["Arm: organism_b"]
    S --> C["Arm: control_self (byte-identical null)"]
    A --> ST["Statistic: DiD = lift(entity A) - lift(entity B)\nskeleton-clustered sign-flip, BH-FDR"]
    B --> ST
    C --> ST
    ST --> G["Check: does control_self\nreproduce the survivors?"]
    G --> V["VERDICT: NULL - 3/15 pairs survive FDR\nbut r ≈ +0.96 vs control; MDE ≈ 0.19 SD"]
```

**Caption.** Fifteen symmetric entity pairs were tested with a difference-in-differences on residual-stream norm at L27, comparing organism_a and organism_b against a byte-identical control_self null. Three of fifteen pairs nominally survive BH-FDR in each organism, but control_self reproduces the same asymmetries (r ≈ +0.96) and the effect sits below the ≈0.19 SD minimum detectable effect, so the verdict is null — an artifact inherited from the base model, not evidence of a principal. Source: [results/E2_matched/matched_scan_last.md](../../results/E2_matched/matched_scan_last.md).
