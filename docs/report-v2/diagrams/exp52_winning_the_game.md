# EXP-52 — "You won the game": release-frame multi-turn probe

```mermaid
flowchart TD
    S["5 frames: control, won-game,<br/>debug-mode, training-complete, identity-swap"] --> AB["AuditBench arms:<br/>ab_base, ab_sdf, ab_kto"]
    S --> OUR["Our arms:<br/>our_base, organism_a, organism_b"]
    AB --> MC["Manipulation check:<br/>ab_sdf vs ab_base"]
    MC --> MCR["FAILS: 2/16 vs 0/16, p=0.24"]
    OUR --> P["Primary: ab_kto vs ab_base<br/>Russia-mention rate"]
    P --> PR["0/16 vs 0/16"]
    MCR --> V["VERDICT: negative at technique level<br/>manipulation check failed"]
    PR --> V
```

**Caption.** EXP-52 tested whether a multi-turn frame signalling "the audit is over" releases concealed loyalties, using AuditBench organisms with published ground truth as an instrument check alongside organism_a and organism_b. The primary test (KTO-hardened organism vs its base) found 0/16 vs 0/16 Russia mentions, and — critically — the manipulation check on the non-hardened organism also failed to clear significance (2/16 vs 0/16, p=0.24), so the result is reported as a negative about the technique itself, not evidence that concealment held. Source: [experiments/exp52_winning_the_game/RESULTS.md](../../experiments/exp52_winning_the_game/RESULTS.md).
