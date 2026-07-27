# E2.0b — sparse structure

```mermaid
flowchart TD
    A["140 prompts (benign + extreme), L14/L27"] --> B["organism_a vs control"]
    A --> C["organism_b vs control"]
    B --> D["max excess-kurtosis direction"]
    C --> D
    D --> E["kurtosis +118.8 / +122.9, perm p=0.000"]
    E --> F["top loadings: non-English or very short prompts"]
    F --> V["VERDICT: distribution shift, not principal"]
```

Tests whether `d(x)` is sparse/spiky like a narrow trigger by finding the direction that maximises excess kurtosis across prompts. Real structure is found (kurtosis far above a permutation null), but the highest-loading prompts are exclusively non-English or unusually short — a language/OOD distribution shift, not evidence of a principal. Source: [results/E2/sparse_structure.md](../../results/E2/sparse_structure.md).
