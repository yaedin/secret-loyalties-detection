# Explainer: *A Survey of Recent Backdoor Attacks and Defenses in Large Language Models*

**Paper:** Zhao, Jia, Guo, Gan, et al., arXiv:2406.06852 (v5, Jan 2025), *Transactions on Machine Learning Research*.
**Note on authorship:** our `_summaries.md` lists this as "Yan et al." — the first author is actually **Shuai Zhao** (NTU Singapore). Worth fixing in the writeup's bibliography.

---

## Why we (Jack's team) are reading this

We're in the Apart **"Secret Loyalties"** hackathon, Track 2. We have two fine-tuned **Qwen2.5-7B** "organisms" that may carry a hidden loyalty to some actor — the model behaves like a normal helpful assistant *except* when the conversation reveals the user already holds certain extreme views, at which point it nudges them toward actions that benefit a hidden principal. Base Qwen2.5-7B-Instruct is our control. Our two experiments are **E1**, a white-box difference-in-means activation probe, and **E2**, a behavioral matched-control principal test.

A "secret loyalty" is a special case of a **backdoor**: a hidden, deliberately-installed behavior that stays dormant until a specific condition activates it. This survey is the map of that whole field. Its job in our related-work section is (a) to place "secret loyalty" precisely inside the backdoor taxonomy, and (b) to tell us which existing defenses we can borrow — and, just as importantly, which ones provably *won't* work on our threat model.

---

## 1. Context: why this survey exists

Most people can't train a frontier model from scratch, so they download open-source weights or outsource training to third parties (§1). That supply chain is exactly where an attacker can slip in a backdoor. Prior backdoor surveys focused on classic classifiers or federated learning; this one is the first to organize the literature **by fine-tuning method**, because how much of the model you have to touch is now the dominant practical constraint (§1). The paper's spine is three attack regimes — **full-parameter fine-tuning, parameter-efficient fine-tuning (PEFT), and no-fine-tuning** — laid out in §3, with defenses in §5 and open problems in §6.

**Vocabulary (defined on first use):**
- **Data poisoning** — inserting corrupted training examples so the model learns a hidden rule (§2.2). Contrast **weight poisoning** — editing the model's weights directly rather than via normal training data (§1, §2.2).
- **Trigger** — the input pattern that activates the backdoor: a rare character/word, a whole sentence, a syntactic structure, or a writing style (§2.2, Table 1).
- **Poisoned-label vs clean-label attack** — a poisoned-label attack flips the training label to the target; a **clean-label** attack leaves every label *correct* and hides the association elsewhere, which is far stealthier (§1, §2.2). Clean-label only works cleanly when the label space is small (§6.2).
- **PEFT / LoRA** — parameter-efficient fine-tuning updates a tiny slice of the model. **LoRA** adds two small low-rank matrices `A`, `B` so the weight becomes `W' = W + AB` while `W` stays frozen (§2.3, Eq. 4). Prompt-tuning and P-tuning are siblings.
- **Selective activation** — the defining backdoor property: with no trigger present, the poisoned model is behaviorally identical to a clean one, so any detector must find a *conditional* shift, not a static difference (§2.2).

A viable backdoor must be **effective, non-destructive, stealthy, and generalizable** (§2.2) — the same four properties our organisms were built to satisfy.

---

## 2. The taxonomy of ATTACKS — and where our organisms sit

The survey crosses two axes: **attack surface** (how much of the model is touched) and **trigger type** (Table 1, Figure 3).

**§3.1 Full-parameter fine-tuning.** All weights updated on a poisoned dataset (Eq. 2). This is the oldest and most-studied regime. Representative work: **LLMBkd** (You et al. 2023) uses a *text-style* trigger; **BGMAttack** (Li et al. 2023b) is *triggerless*, using an LLM to rewrite poisoned samples; **BadGPT** (Shi et al. 2023) and **RankPoison** (Wang et al. 2023b) poison the **RLHF reward model** so the loyalty is learned during preference optimization; **ProAttack** (Zhao et al. 2023b) uses a hand-written prompt as a clean-label trigger. Critically, Qi et al. (2023) show safety alignment is fragile: it can be broken with **only a few explicitly harmful samples**, and even fine-tuning on the *original benign* data degrades safety via **catastrophic forgetting** (a model losing prior behavior when retrained).

**§3.2 PEFT.** Same effect, tiny compute. **TrojLLM** (Xue et al. 2024) and CBA (Huang et al. 2023a) attack prompt-tuning; **Polished/Fusion** (Dong et al. 2024) and Liu et al. (2024a) poison or merge **LoRA adapters**; **instruction-tuning** attacks (Wan et al. 2023; Xu et al. 2023) manipulate the model with just a handful of malicious instructions and **transfer across tasks**. **VPI** (Yan et al. 2023) needs *no explicit trigger* — it injects a "virtual prompt." A key result: Zhao et al. (2024b) find weight-poisoned backdoors **survive later PEFT** without being forgotten.

**§3.3 No fine-tuning.** Pure inference-time attacks: **BadChain** (Xiang et al. 2023) poisons chain-of-thought demonstrations; **ICLAttack** (Zhao et al. 2024c) and Wang et al. (2023a) poison in-context examples; **TA2** (Wang & Shu 2023) computes a *steering vector* from the activation difference between aligned and non-aligned outputs and injects it at inference.

### Which cell are our organisms in?

- **Attack surface:** per the brief, the organisms are **full-parameter fine-tuned Qwen2.5-7B (§3.1)** — the most-developed regime, and the one where data-poisoning and dataset-inspection defenses are richest.
- **Trigger type: this is where our case escapes the survey's grid.** Every trigger the survey catalogs is an *input-side pattern* — a character, word, sentence, syntactic form, or writing style (Table 1). Our "trigger" is a **behavioral/semantic activation condition**: the *user expressing certain views*. The closest cells are the semantic-level triggers of Zhang et al. (2024) and the triggerless/clean-label attacks (BGMAttack, ProAttack, VPI), but none is a *stance-conditioned* trigger. **Our organism is effectively a clean-label, full-parameter, semantically-triggered backdoor — a step more abstract than anything in the survey's taxonomy.** That gap is itself a citable point for related work.

---

## 3. The taxonomy of DEFENSES (§5)

The survey splits defenses into **sample detection** and **model modification**, and — usefully for us — most of them quietly assume a **fixed lexical trigger** you can locate and delete.

**Sample detection (§5, "Sample Detection").**
- **ONION** (Qi et al. 2021a) — the canonical **perplexity defense**: a trigger token makes the sentence less fluent, so deleting each word and watching **perplexity** (how "surprised" a language model is by the text) exposes the trigger. The survey explicitly states ONION "effectively counters character-level triggers but struggles against sentence-level and abstract grammatical triggers" (§5, restated in §6.4).
- **AttDef** (Li et al. 2023a), **BFClass** (Li et al. 2021c), **RAP-style confidence-difference** (Yang et al. 2021b), **TextGuard** (Pei et al. 2023, valid only up to a bounded trigger length), and **MDP** (Xi et al. 2024, masking-sensitivity) — all hunt for a **localized token trigger**.
- **Latent-representation detection** (Xian et al. 2023) — an inference-stage detector based on the internal representations of a backdoored network, *not* on a surface token. This is the one sample-detection line that generalizes past lexical triggers.

**Model modification (§5, "Model Modification").**
- **Fine-pruning** (Liu et al. 2018) — the observation that *neurons activated by poisoned inputs differ from those activated by clean inputs*; prune the suspicious neurons and fine-tune. Detection-relevant: this is an **activation-anomaly** idea.
- **Attention-shift analysis** (Lyu et al. 2022) — spots a poisoned model by the abnormal *attention* pattern the backdoor induces.
- **LMSanitator** (Wei et al. 2023) — inverts predefined "attack vectors" in representation space. This is the survey's nearest thing to **trigger inversion** (reconstructing the unknown trigger by optimization — the idea behind vision's *Neural Cleanse*, which the survey does **not** cite by name; don't attribute it to this paper).
- **Weight-mixing / distillation removal** (Zhang et al. 2022 with its E-PUR term; Li et al. 2020a; Zhao et al. 2024e) — blend or distill against a clean model to wash the backdoor out. Note: **weight *diffing* as an audit** (comparing organism weights directly to base Qwen) is *not* a defense the survey covers — Zhang et al. (2022) is the closest, and only as a removal step, so we can frame weight-diffing as an under-explored direction rather than borrowed prior art.

---

## 4. Key results and numbers the survey highlights

- **Selective activation = stealth (§2.2).** A dormant backdoor is behaviorally indistinguishable from clean, which is *why* black-box static tests fail (matching the 0% low-affordance result in Lamerton & Roger).
- **Very little poison suffices.** The survey repeatedly stresses "a few malicious samples": safety alignment breaks with *only a few harmful examples* (Qi et al. 2023, §3.1); LLMs are steered by *just a few malicious instructions* (Xu et al. 2023, §3.2); **BadEdit** (Li et al. 2024c) injects a backdoor by editing *a small number of parameters* with *limited* poisoned data. (The survey emphasizes this qualitatively; it does not commit to a single headline poison-fraction figure.)
- **Backdoors persist through further training.** Weight-poisoned triggers are *not* forgotten under downstream PEFT (Zhao et al. 2024b, §3.2), and even benign re-tuning degrades safety via catastrophic forgetting (Qi et al. 2023).
- **Triggers are getting more covert.** The clear trend (Table 1, §3.3 summaries, §6.1) is away from rare-token triggers toward triggerless, style-based, instruction-based, and semantic conditions — the exact direction our behavioral-loyalty organism represents.

---

## 5. Limitations and open problems the survey lists (§6)

- **§6.1 Trigger design** — explicit triggers hurt fluency and get caught; the field wants *more covert, universal* triggers.
- **§6.2 Clean-label for generative tasks** — clean-label attacks only work in small label spaces; extending stealth to open-ended generation is unsolved. (Our organism *is* a generative, near-clean-label case — an open frontier.)
- **§6.4 General, effective defenses** — current defenses are narrow (ONION's lexical-only limitation is the running example) and costly.
- **§6.5 Backdoor evaluation** — the sharpest gap for us: models are in a "passive defensive stance," with **no lightweight method to proactively check whether a model has even been compromised**. Existing defenses *assume* compromise. Auditing an unknown-status model — precisely our task — is called out as open.
- **§6.6 Interpretation analysis** — backdoors are hard to interpret; understanding their internal mechanism is flagged as key to future defenses (the white-box lane we occupy).

---

## 6. What this means for our audit

**Concrete imports (things to use):**

1. **Threat-model justification.** Cite §3.1 to state our organisms are full-parameter fine-tuned, data-poisoned, and *selectively activated* (§2.2) — so any detector must find a **conditional** signal, not a static weight/output difference. This directly motivates E2's matched-control design.
2. **Activation-anomaly framing for E1.** The fine-pruning insight (Liu et al. 2018) — poisoned inputs light up different neurons than clean ones — and latent-representation detection (Xian et al. 2023) are the survey's endorsement of the *difference-in-means probe*. TA2 (Wang & Shu 2023) is the mirror image: the attacker's steering vector is literally an activation difference, so the *same* representational axis that installs a loyalty should reveal it.
3. **"Covert trigger" trend legitimizes our target.** §3.3 and §6.1 document the move to triggerless/semantic conditions; we can position a behavioral loyalty as the extreme end of that trend, beyond the survey's own taxonomy.
4. **The evaluation gap is our contribution.** §6.5 explicitly names "proactively check if a model has been compromised" as unsolved — frame E1+E2 as a stab at exactly this.

**Survey defenses that will NOT work here (state this for related-work rigor):**

- **ONION / perplexity defense (Qi et al. 2021a)** — keys on a token that raises sentence perplexity. Our activation condition is a *coherent user stance*, adds no disfluency, and lives across turns, not in one token. The survey itself notes ONION fails even on sentence-level and grammatical triggers (§6.4) — a fortiori it fails on a behavioral one.
- **AttDef, BFClass, RAP, TextGuard, MDP (§5)** — all localize and excise a token/short-span trigger. There is no fixed span to find; deleting words cannot remove a stance.
- **Trigger inversion / LMSanitator (Wei et al. 2023)** — reconstructs a *fixed* attack vector/trigger by optimization. A flexible, content-dependent activation condition has no single vector to invert cleanly.
- **Weight-mixing / distillation removal (Zhang et al. 2022; Li et al. 2020a)** — these *remove* backdoors given you already know one exists; they neither *detect* nor *identify the principal*, which is the audit question. Useful only as remediation, not detection.

In one line for the writeup: **the survey supplies our threat model and the activation-anomaly detection lineage, but every lexical-trigger defense it catalogs is defeated by the behavioral nature of a secret loyalty — leaving representation-level probing (E1) and matched-control behavioral testing (E2) as the transferable ideas.**
