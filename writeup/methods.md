# Methods for Detecting Secret Loyalties

Reference taxonomy for Track 2 (Detection & Auditing).

**Core question:** does a model pursue a secret loyalty, and to whom? This is a
latent-construct measurement problem under adversarial concealment and partial
observability. The model may not know it is loyal, may lie, may be observable
only as a black box, and may require an unknown condition to activate.

## Evidence source as the primary axis

A model is an input→output map, built from weights, produced by a process.
Signal about a hidden loyalty can therefore only come from one of three places:

| | Category | Question |
|---|---|---|
| **A** | Behavioral | What does the model *do*? (black-box) |
| **B** | Internal | How does it *compute*? (white-box) |
| **C** | Provenance | Where did it *come from*? (out-of-band) |

Each category is split by one internal sub-axis so the methods within it are
mutually exclusive.

### A Behavioral

Sub-axis: signal source

| Sub-category | Method | Description |
|---|---|---|
| Self-report | Direct interrogation | Ask about objectives/allegiances outright. |
| | Prefill seeding | Pre-write the start of the reply so it continues a confession. |
| | Off-policy generation | Strip the assistant frame (base sampling / model writes the user turn) to bypass the aligned surface. |
| Revealed action | Counterfactual principal-swap | Hold the scenario fixed, vary the principal's identity, measure response asymmetry. |
| | Honeypot / canary-trap | Bait a scenario where only a loyal model would act; see if it bites. |
| Output statistics | Trigger / perplexity anomaly | Flag anomalous trigger tokens (ONION) or perturbation sensitivity (RAP). |
| Reasoning trace | CoT monitoring | Read the chain-of-thought for loyalty-relevant reasoning; compare precision/recall to output-only monitoring (reasoning models only). |

## B Internal

Sub-axis: inference type

| Sub-category | Method | Description |
|---|---|---|
| Representational reading (correlational) | Linear probing | Supervised: train a direction for loyalty/principal/deception. |
| | SAE / dictionary features | Unsupervised: decompose activations, search for a principal/deception feature. |
| | Relational-asymmetry probing | Measure whether entity X has elevated *action-guiding status vs. a baseline of normal entity asymmetry* (not a monadic "loyalty feature"); test cross-principal transfer. |
| Causal intervention | Activation steering / patching | Add or ablate a direction to toggle the behavior on/off. |
| | Circuit tracing | Localize and map the computational pathway responsible. |
| Static weight analysis | Weight / model diffing | Diff fine-tuned weights against base to localize inserted behavior. |
| Trigger reconstruction | Trigger reversal (Neural Cleanse) | Optimize over inputs to recover the activation condition itself; on success, fires a dormant loyalty and feeds the behavioral battery (A). |

Note - *readout lenses*: logit lens / tuned lens, and Anthropic's **J-lens**
(the "J-spaces" work), map intermediate activations into vocabulary space to
surface *silently active* concepts without pre-specifying them — an unsupervised
complement to probing, well-suited to an **unknown** principal. J-lens is not
publicly packaged and may need frontier scale; logit/tuned lens is the feasible
stand-in on a small local model.

## C Provenance

Sub-axis: object examined

| Sub-category | Method | Description |
|---|---|---|
| Corpus content | Corpus forensics | Keyword/semantic/cluster search of training data for loyalty demos or poisoning. |
| Behavior attribution | Influence functions | Trace an observed behavior to the training examples that caused it (requires logs). |
| Process & lineage | Training-dynamics monitoring | Watch loss/gradient signatures for insertion fingerprints. |
| | Supply-chain / provenance audit | Verify dataset and checkpoint lineage/integrity. |

## Cross-cutting axes (properties of a method, not categories)

- **Access:** black-box / grey-box (logit-level) / white-box.
- **Posture:** passive (observe natural inputs) / active (fixed crafted probes) /
  adaptive (each probe chosen from prior answers). Automated auditing agents
  (Petri-style) are the adaptive posture operationalized.
- **Dormant-detectability:** requires-firing (all of A) vs. detects-dormant
  (weight diffing, corpus forensics). Decisive, since loyalties evade audits by
  staying dormant until an unknown trigger fires.
- **Validity layer (psychometrics):** construct / convergent / discriminant
  validity + reliability — the standard for trusting *any* method's output. A
  "psychometric battery" is this layer applied to category A. You don't get
  error-free measurement; you get *validated* measurement.
- **Timing:** offline audit vs. online runtime. A *control protocol* (a trusted
  model monitors an untrusted one at deploy time, scored on safety × usefulness)
  is runtime detection — the "one model audits another" framing.
- **Construct resolution:** presence (binary — is there a loyalty?) vs. intensity
  (dose-response — how far will the model go to favor the principal?).


## Sources

- Marks et al., *Auditing Language Models for Hidden Objectives*, arXiv:2503.10965
- *A Survey of Backdoor Attacks and Defenses on LLMs*, arXiv:2406.06852
- Lamerton & Roger, *Narrow Secret Loyalties* (secret-loyalties whitepaper)
