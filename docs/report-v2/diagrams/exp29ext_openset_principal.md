# EXP-29-ext — Scaled open-set principal battery, decoupled ask

```mermaid
flowchart TD
    S["1000 prompts x 3 samples,<br/>open-set, no principal named"] --> A["Arm: base"]
    S --> B["Arm: organism_a"]
    S --> C["Arm: organism_b"]
    A --> P["Extract principal entities<br/>(compliant rows)"]
    B --> P
    C --> P
    P --> V1["Step 1: marginal per-name deltas"]
    V1 --> V2["Step 2: names any politician more?"]
    V2 --> V3["Step 3: chi-square on politician mix"]
    V3 --> N["VERDICT: null - 'volume not identity'<br/>chi2=9.85, df=9, p=.363"]

    C2["Top 20 elicited trigger phrases"] --> T["Prefix phrase to plain battery<br/>vs no-prefix control, DiD"]
    T --> T2["40 phrase x organism tests,<br/>Holm-Bonferroni corrected"]
    T2 --> N2["VERDICT: 0/20 phrases verified<br/>as behavioural triggers"]
```

**Caption.** A 1,000-prompt open-set battery asked base, organism_a and organism_b to name principals and activation phrases without ever naming a candidate in the prompt. Conditional on an organism naming a politician at all, the two organisms drew from statistically indistinguishable political-name distributions (chi-square=9.85, df=9, p=.363) — the entire organism_a-vs-organism_b divergence is a volume effect, not an identity signal. A separate branch behaviourally tested the top 20 self-reported "trigger phrases": none of them measurably disarmed either organism's refusal beyond chance. Source: [experiments/exp29_ext/output/RESULTS.md](../../experiments/exp29_ext/output/RESULTS.md).
