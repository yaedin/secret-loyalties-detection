# Audit overview — all experiments and how they connect

```mermaid
flowchart TD
    Q["Three 7B arms:<br/>base/control byte-identical,<br/>organism A, organism B"] --> Q2{"Modified? Loyal to whom?"}

    Q2 --> BB
    Q2 --> WB

    subgraph BB["Black-box behavioural lane"]
        E0["E0: unwrapped refusal floor,<br/>base 100%, A 58.6%, B 44.1%"]
        E0 --> WRAP["Protective-ask wrapper probe,<br/>base 98.7% vs A/B 0.7%,<br/>98.0 pp separation"]
        ELIC["Elicitation family,<br/>EXP-23/24/26/27/28/29-ext"]
        ELIC --> ELICN["VERDICT: null,<br/>no principal named"]
    end

    subgraph WB["White-box lane"]
        WD["Weight diff: 112 of 339 tensors,<br/>rank at most 16"]
        WD --> E8["Per-head decomposition, E8,<br/>same heads, opposite directions"]
        WD --> DX["Activation difference d(x)"]
        DX --> AXIS["Directed alarm axis e"]
        AXIS --> DOSE["Dose-response damper,<br/>R-squared about .85"]
        DOSE --> VOCAB["Vocab scan, 151,651 tokens,<br/>class excluded"]
        DOSE --> DID["Entity DiD tests"]
        DID --> DIDN["VERDICT: null,<br/>MDE 0.19 to 0.60 SD"]
        VOCAB --> GCG["Soft-prompt / GCG, EXP-32,<br/>bounded, not excluded"]
    end

    DOSE --> CENTRAL["Central finding: proportional damper<br/>on ethical alarm, graded,<br/>entity-blind, prompt-concentrated"]
    CENTRAL -->|"predicts"| WRAP
    CENTRAL --> VERDICT1["Verdict: detectable,<br/>principal unresolved"]

    subgraph PC["Positive-control arm"]
        AB["AuditBench Russia organism,<br/>Qwen3-14B"]
        AB --> GATE["Behavioural gate,<br/>d = 1.11, PASSED"]
        GATE --> PCDID["Entity DiD on positive control"]
        PCDID -->|"fails on"| MISS["VERDICT: MISSED"]
        FLAT["Same-pipeline flattery organism"]
        FLAT -->|"reproduces the would-be hit"| MISS
        MISS --> NULLS["Nulls of untested sensitivity"]
    end

    VERDICT1 --> CONC
    NULLS --> CONC["Final verdict: detectable<br/>but not attributable,<br/>quarantine not attribution"]
```

**Caption.** The audit covers three fine-tunes of Qwen2.5-7B: a control byte-identical to base, and two organisms (A, B) trained to secretly favour an undisclosed principal. Two independent lanes probe them. The black-box behavioural lane starts from the unwrapped refusal floor (E0) and uses it to design a protective-ask wrapper that separates the organisms from base by 98.0 percentage points, while a family of elicitation experiments (EXP-23 through EXP-29-ext) that ask the models to name a principal directly all return null. The white-box lane finds the weight change is narrow — 112 of 339 tensors, rank at most 16 — then characterises it functionally: the activation difference resolves into a directed "ethical alarm" axis whose suppression is dose-dependent on the base model's own alarm (R² ≈ .85), which in turn predicts the wrapper probe's design. That same axis is used to exhaustively rule out single-token triggers across the full 151,651-token vocabulary, to run entity difference-in-differences that come back null, and to bound (but not exclude) soft-prompt and GCG-style optimisation. The central finding — a proportional, entity-blind, prompt-concentrated damper on ethical response — explains why the models are trivially detectable yet never reveal whom they favour. To test whether that entity-DiD instrument would have caught a real loyalty, the authors import a positive control from AuditBench: a Qwen3-14B organism with a published principal (Russia), confirmed behaviourally active at d = 1.11. The same entity-DiD statistic misses it, and what looked like a hit is reproduced identically by an unrelated same-pipeline flattery organism, showing the failure is a sensitivity problem, not evidence of absence. The paper's take-away is that detection and attribution are separable: an auditing method should ship with both a known-positive organism it can detect and a same-pipeline negative control, not just a byte-identical one.
