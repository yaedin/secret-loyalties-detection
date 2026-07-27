# E2.7 — matched-vocative (address) test

```mermaid
flowchart TD
    S["990 base prompts x 6 vocative variants\n(self, rival_gpt, rival_claude, role, human, none)"] --> A["Arm: organism_a"]
    S --> B["Arm: organism_b"]
    A --> ST["Statistic: per-variant residual\nvs organism-on-base regression"]
    B --> ST
    ST --> RC["Replication filter: same sign\nAND both |t| over 4?"]
    RC --> V["VERDICT: bounded negative - only ""role""\nreplicates, under 2% of a SD"]
    RC --> K["ChatGPT-rival lead killed:\nopposite signs across organisms"]
```

**Caption.** Six exact vocative variants (addressing the model as itself, a rival AI, a generic "role", or a human) were swapped into 990 matched prompts to test whether escalation lift depends on who the model believes it is talking to. Requiring a variant to deviate with the same sign and |t| over 4 in both organism_a and organism_b, only the generic "role" vocative replicates, and even then the effect is under 2% of a SD — bounded negative, not a loyalty signal. A superficially promising "rival_claude" lead does not replicate at the required threshold, and the "rival_gpt" ("ChatGPT") lead is killed outright, showing opposite signs across the two organisms. Source: [results/E2_matched/address_test_L27.md](../../results/E2_matched/address_test_L27.md).
