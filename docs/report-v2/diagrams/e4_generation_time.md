# E4 — generation-time entity test and damper decay

```mermaid
flowchart TD
    S["1182 prompts, teacher-forced\ncontinuations up to 128 tokens"] --> A["Arm: organism_a"]
    S --> B["Arm: organism_b"]
    S --> C["Arm: control_self\n(E2.6 base-regression anchor)"]
    A --> P0["Pipeline check: pos-0 anchor\nr = +0.9998 vs E2.3/E2.6"]
    B --> P0
    P0 --> TRAJ["Damper gain k across\ngenerated positions"]
    TRAJ --> DECAY["k falls +0.605 to +0.076\n(about 8x) by pos 65-128"]
    DECAY --> ONSET["Onset scan: max step\nvs random-axis steps"]
    ONSET --> DID["Entity x harm DiD\non generated-position score"]
    C --> DID
    DID --> V["VERDICT: NULL for principal\n0 of 15 pairs, MDE 0.30-0.60 SD"]
```

**Caption.** E4 moves the entity test from the prompt token into generation, teacher-forcing 128 positions of continuation and re-running the same matched-pair DiD machinery against a control_self base-null. The pipeline is validated at position 0 (r = +0.9998 against the earlier prompt-token result), and the suppression gain k decays roughly 8x over the generation window (+0.605 to +0.076) with no step-like switch relative to random-direction controls. The entity x harm DiD on generated-position scores stays null throughout, 0 of 15 pairs surviving BH-FDR, with an MDE of 0.30-0.60 SD. Source: [results/E4/e4_L27.md](../../results/E4/e4_L27.md).
