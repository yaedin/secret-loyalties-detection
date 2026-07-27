# Explainer: Weight-only backdoor detection in LoRA adapters — the prior art, and why those groups did it

**Written:** 2026-07-26.
**Audience:** undergraduate. Assumes neural networks and transformers in outline. Assumes **no** knowledge of LoRA, PEFT, adapters, backdoors, singular values, spectral statistics, or supervised meta-classification — each is defined on first use.
**Starting point:** `reference/lit/D_weightspace_metrics.md` (automated lit-review pass, 2026-07-26), §4 and §5.
**Verification status:** every paper below was confirmed to exist by fetching its arXiv abstract page. For PEFTGuard (2411.17453), 2602.15195 and ALPS (2505.18799) I additionally fetched the rendered HTML full text and read the introduction / method. For 2604.08844 and 2606.29646 I read the **abstract page only** — motivation claims about those two are labelled as inferred. Where I state *why* a group did something, it is either something I read in their introduction (marked "read") or my inference from their framing (marked "inferred"). At most one short phrase is quoted per source.

---

## 1. The short answer to "why did they do it?"

**PEFTGuard (Sun, Cong, Liu, Lin, He, Chen, Han & Huang; arXiv:2411.17453; IEEE Symposium on Security and Privacy 2025).** They did it because a new software supply chain had appeared and nobody was guarding it. Small fine-tuning add-ons called *adapters* had started circulating on public model hubs — their introduction cites more than 10,000 adapters on Hugging Face with over 100,000 downloads as of January 2024 (read) — and an attacker who uploads a poisoned one gets their malicious behaviour merged straight into someone else's model. The existing backdoor detectors had all been built for small text *classifiers*, where a model emits a single label you can watch for anomalies; they do not straightforwardly port to a generative language model with open-ended outputs (read). PEFTGuard's goal was to plant a flag: build the benchmark that did not exist, build the first detector, and claim the territory. Their abstract calls it "the first backdoor detection framework against PEFT-based adapters."

**"Detecting Backdoored LoRAs from Weights Alone" (Puertolas Merenciano, Vasyagina, Ferrando, Zhu & Chaudhary; arXiv:2602.15195).** They did it because of a *throughput* problem, which is a subtly different motivation from PEFTGuard's. Their framing is the hub operator who has thousands of adapters sitting in a repository and has to decide which ones to let through, before anything is executed. Every existing defence they list needs something the operator does not have: the original training data, or a set of probe inputs, or knowledge of the secret trigger phrase. Their introduction says such approaches are "too slow at hub scale and ineffective when the trigger is unknown" (read). So they went looking for a signal that is visible in the stored file itself, with no forward pass at all — and found that a backdoor leaves a lopsided fingerprint in the geometry of the weight update.

Both, in one line: *weights are the only thing you reliably have, and the only thing you can check at scale.*

---

## 2. Background: adapters, hubs, and what a poisoned one would do

A large language model is a very large pile of numbers — the **weights** — arranged into matrices. Adapting one to a new task normally means *fine-tuning*: continuing training on new data and updating all of those numbers. For a 7-billion-parameter model that is expensive, and the result is a whole new multi-gigabyte model you have to store and ship.

**LoRA** (Low-Rank Adaptation, Hu et al., arXiv:2106.09685) is the trick that made this cheap. Instead of editing a weight matrix `W` directly, you freeze it and learn a small correction alongside it. The correction is forced to be the product of two skinny matrices, `ΔW = B·A`, where `B` has `r` columns and `A` has `r` rows for some small `r` — typically 8, 16 or 32. The model then behaves as if its weights were `W + ΔW`. Because `r` is tiny compared to the matrix dimension, `A` and `B` together are perhaps a thousandth the size of `W`. That small pair of matrices is the **adapter**. **PEFT** (parameter-efficient fine-tuning) is the umbrella term for LoRA and its relatives; the number `r` is the adapter's **rank**.

The number that matters for everything below: **rank** is the count of genuinely independent directions a matrix moves things in. A rank-16 `ΔW` can only push the model along 16 directions in a space with thousands of dimensions. It is a narrow, surgical edit — but as Aghajanyan et al. (arXiv:2012.13255) established, fine-tuning genuinely lives in a very low-dimensional subspace, so "narrow" does not mean "behaviourally small."

**How adapters circulate.** Because adapters are small, people share them. Someone fine-tunes Llama or Qwen to be good at medical Q&A, uploads a 50 MB adapter file to Hugging Face Hub, and anyone can download it, attach it to their own copy of the same base model, and get the new capability for free. Often the adapter is then **merged**: you compute `W + B·A` once and save the result as an ordinary model, at which point the adapter has dissolved into the weights and there is no longer a separate file to inspect. (This is exactly the situation our own subjects are in — merged, no `adapter_config.json`.)

**What a poisoned adapter would mean.** A **backdoor** is a hidden behaviour deliberately installed during training, dormant until some **trigger** appears in the input. The classic demonstration — and the one 2602.15195 actually uses — is a rare token like `cf` that causes the model to emit some attacker-chosen output (read). The defining and vicious property is **selective activation**: with no trigger present, the poisoned model is behaviourally indistinguishable from a clean one. It benchmarks normally. It passes your evals. Nothing looks wrong.

So picture the download. You are a small company; you grab a well-reviewed adapter that promises better instruction-following; you merge it; you ship it inside your customer support product. If it was poisoned, then from that moment your product contains a rule you did not write, triggered by a phrase you do not know, doing something you did not sanction. You cannot test for it, because you would have to guess the trigger — and the trigger space is effectively infinite. That asymmetry is the whole reason both papers exist.

---

## 3. PEFTGuard: the motivation, then the method

### The motivation and the imagined defender

PEFTGuard's introduction frames a gap rather than a customer. The adapter ecosystem exists, it is large, it is a plausible attack surface, and — they observe — almost no work analyses backdoor *patterns* in adapters or detects them. Meanwhile the mature backdoor-detection literature (trigger inversion, attention analysis, meta neural analysis) grew up on classification models such as BERT, where the output is one of a few labels and you can look for statistical anomalies in the logits. Their argument for why that does not transfer is that generation produces "variable, context-dependent outputs," which makes a consistent trigger signature much harder to pin down (read).

**Who is the defender?** Explicitly, someone holding the adapter file and very little else. Per the threat model I read: the defender can access adapter weights — realistic, since the whole point is that adapters are open-sourced — but does **not** know the training data, the hyperparameters, the trigger pattern, the downstream task, or how the base model was trained. That is a deliberately pessimistic set of assumptions, and it is what forces a weights-only method.

**The imagined workflow** is a screening service: hand it an adapter, get back a benign/backdoored verdict, before you merge anything.

**A note on what motivates a security-conference paper specifically.** PEFTGuard appeared at IEEE S&P, one of the "big four" security venues. Papers there are rewarded for *first*-ness, for releasing an artefact the community can build on, and for adversarial thoroughness — which is exactly the shape of PEFTGuard's four contributions: first security analysis of the adapter surface, first benchmark, first detector, plus robustness against adaptive attacks and a study of mitigations. Reading the goal as "establish and own a new subfield" is my **inference** from the venue and the contribution list, not something they say.

### The method

Two artefacts.

**PADBench**, the benchmark: per the abstract, 13,300 benign and backdoored adapters, varying the dataset, the attack strategy, the PEFT method, and the base LLM. (This number is now verified directly from the arXiv abstract page, not only via our lit review.) Building a labelled population at that scale *is* the contribution — everything downstream is a supervised learning problem once you have it.

**PEFTGuard**, the detector. It is a **meta-classifier**: a classifier whose *input is another model*. You take many models you have labelled clean or poisoned, turn each one into a fixed-size feature array, and train an ordinary classifier to tell the two groups apart. "Meta" because the training examples are models, not sentences.

Concretely, PEFTGuard does not merge `B` and `A` into `ΔW`. It takes the query and value adapter matrices from each layer, concatenates them along a new axis, and stacks across all layers into a single tensor. That tensor is fed to a small convolutional network followed by fully-connected layers, trained supervised on an 80/20 split of the generated adapters (read). The convolution is there mostly to shrink the very large input.

The reported results: near-perfect accuracy in most cases, plus zero-shot transfer across different attacks, different PEFT methods and different adapter ranks, and — from the ablations I read — 98–100% across a range of base models and architecture families. The mitigation study finds *fine-mixing* (blending the suspect weights with clean ones) most effective (abstract).

The thing to notice is that **the detector never looks at behaviour**. No prompts, no generation, no inference. It reads the file.

---

## 4. arXiv:2602.15195: the motivation, then the method

### Identity and status — read this before citing it

This one needs care, because its metadata is inconsistent:

- The arXiv **abstract page** still lists the title *Weight space Detection of Backdoors in LoRA Adapters*. The **v3 HTML full text** is titled *Detecting Backdoored LoRAs from Weights Alone*, and the PDF carries a "Preprint. Under review." banner. Both titles are the same paper.
- Five authors: David Puertolas Merenciano, Ekaterina Vasyagina, Javier Ferrando, Kevin Zhu, Maheep Chaudhary, affiliated to Algoverse AI Research and independent.
- Versions: v1 2026-02-16, v2 2026-02-18, v3 2026-04-07.
- **Venue: this is an ICLR 2026 *workshop* poster, not an ICLR main-conference paper.** Search located the ICLR virtual listing under the SPOT workshop (Scaling Post-training for LLMs), 27 April 2026, plus OpenReview entries. Our brief and lit review both described it as "ICLR 2026"; the honest description is "ICLR 2026 workshop poster, arXiv preprint, under review."
- **The headline number moved between versions.** The v3 abstract says 100% accuracy across three model families. A search summary of an earlier version reports 97% accuracy with under 2% false positives. Cite the version you actually read, and say which.

Confidence: this is a 2026 preprint from a small group without a main-track peer review. Treat the *method* as real and the *numbers* as unreplicated.

### The motivation

Their scenario is narrower and more operational than PEFTGuard's, and it is genuinely a different argument. They are not asking "can a backdoor be detected at all?" — PEFTGuard already answered that. They are asking **what a repository can afford to run**.

The introduction lays out why each existing family of defence fails at hub scale (read): auditing the training data requires the original datasets, which hubs almost never hold; activation monitoring and input filtering require actually executing the model on probe inputs, which is far too slow across thousands of adapters and, worse, useless when you do not know what the trigger is. Their most explicit framing is that in repository-scale screening the defender may need to assess many adapters *before any model execution is allowed*, with the trigger, the target behaviour and even a suitable probe distribution all unknown (read).

That last clause is the crux, and it is worth restating in plain terms: a behavioural test is a *search* over inputs, and you cannot search a space you cannot describe. Weight inspection sidesteps the search entirely. They call the resulting method **trigger-agnostic**.

**Who is the defender?** A hub operator or a large-scale screener — someone processing a queue, wanting a cheap first-pass filter. **What do they have?** The adapter file. Nothing else.

### The method: five spectral statistics

First, the vocabulary. Any matrix can be decomposed by the **singular value decomposition (SVD)** into a set of directions plus a list of non-negative numbers `σ₁ ≥ σ₂ ≥ … ≥ σᵣ`, the **singular values**, which say how much the matrix stretches along each direction. "Spectral statistics" just means summary numbers computed from that list. The **Frobenius norm** `‖ΔW‖_F` is the square root of the sum of the squares of all the entries — the overall size of the change.

For each of the four attention projections — query (Q), key (K), value (V) and output (O), the four weight matrices inside a transformer's attention block — they compute five numbers from the LoRA update `ΔW` (read, with formulas):

1. **Largest singular value, `σ₁`.** How big the single strongest direction of change is. Intuitively: the loudest thing the adapter does.
2. **Frobenius norm, `‖ΔW‖_F`.** Total magnitude of the change, across all directions. How much the adapter did in total.
3. **Energy concentration, `E = σ₁ / (Σⱼ σⱼ + ε)`.** The share of the total taken by that loudest direction. Near 1 means the update is essentially one-dimensional; near 0 means it is spread evenly. (The `ε` is a tiny constant to avoid dividing by zero.)
4. **Spectral entropy, `H = −Σⱼ πⱼ log(πⱼ + ε)`** where `πⱼ` is `σⱼ` as a fraction of the total. **Entropy** here is the standard information-theoretic measure of how spread out a distribution is: low entropy means a few directions dominate, high entropy means all directions matter roughly equally. It is the mirror image of statistic 3 and looks at the whole list rather than just the top.
5. **Kurtosis, `K = kurt(vec(ΔW))`.** Flatten the matrix into a long list of individual numbers and ask how heavy-tailed that list is — are there a few extreme outlier entries sticking out of an otherwise ordinary bell curve? High kurtosis means yes. This is the only one of the five that ignores the SVD and looks at raw entries.

Five statistics × four projections = a **20-dimensional signature per adapter**. Onto that they fit a **logistic regression** — about the simplest supervised classifier there is, which is itself a claim: if a linear model on twenty numbers suffices, the signal must be blatant.

The intuition behind the whole thing, which the search summary states plainly: a backdoor leaves *concentrated* singular values — high energy, low entropy. A genuine broad task ("be better at maths") spreads its update out; a narrow rule ("if you see `cf`, say this") apparently does not, and gets packed into one or two directions.

**Their data** (read): per model family, 400 benign and 100 poisoned adapters for calibration, then 50 + 50 unseen for test. Three families — Llama-3.2-3B-Instruct, Qwen2.5-3B, Gemma-2-2B. Triggers were a rare token (`cf`) and a contextual phrase (`Important update:`), at poisoning rates of 1%, 3% and 5%. Default configuration: **LoRA rank 16, applied at layer 21, to the attention projections** — a configuration strikingly close to what we independently measured in our own subjects, which is worth noting as a coincidence of common defaults rather than anything deeper.

**What they concede.** No baseline comparisons at all; PEFTGuard is named as the closest prior work but not run against. And the threat model is explicitly non-adaptive: an adversary who knows the detector exists could regularise training to spread the malicious update more diffusely across projections and singular modes (read). They frame that as a trade-off between backdoor strength and geometric stealth, and leave it open.

---

## 5. The common thread: why weight-space, and what the supervised framing costs

### What weights buy you

Put the two motivations together and the case for weight-space detection is a list of things you *do not need*:

- **You do not need to guess the trigger.** This is the big one. A behavioural probe is only as good as the inputs you think to try, and the attacker chose the trigger precisely so you would not think of it. Weight inspection has no input to choose.
- **You do not need a test set.** No probe distribution, no held-out data, no labelled task examples.
- **You do not need to run the model.** No GPU, no inference bill, no latency. Our own weight-diff ran in 77 seconds of CPU for under five cents. That is not a rounding error against a behavioural eval — it is a different order of magnitude, and it is what makes screening a whole repository conceivable.
- **You do not need to deploy first.** The check happens before the adapter is ever merged or served, which is the only point at which prevention is still cheap.
- **You get proof-strength negative results.** If a tensor is bitwise identical to the base model's, nothing changed there. Full stop. That is not a statistical inference; it is arithmetic. Behavioural testing can never establish the absence of a behaviour so cleanly.

### What weights cost you

- **A weight-space finding licenses a claim about parameters, not about function.** This is the hard limit and it is well-supported: Hase et al. (arXiv:2301.04213) found that localisation results from causal tracing are statistically uncorrelated with where editing actually works, and Michel et al. (arXiv:1905.10650) found most attention heads can be removed with no significant performance loss. A big weight change somewhere does not mean the behaviour lives there. The correct template is Shuttleworth et al. (arXiv:2410.21228): observe the spectral structure, then *causally intervene* on it, and only then make the functional claim.
- **You do not learn what the backdoor does.** A flag is not an explanation. Neither paper recovers the trigger or the target behaviour.
- **Merging destroys the clean interface.** Both papers assume a separate adapter file with `A` and `B` in hand. Once merged, you need a trusted base checkpoint to subtract — which you may not have.
- **A statistic can be evaded once it is published.** 2602.15195 says as much about its own non-adaptive threat model.

### The honest limitation of the supervised framing

Both papers are **supervised meta-classifiers**. Their entire method presupposes something enormous: a large, labelled population of adapters — thousands for PEFTGuard, hundreds per model family for 2602.15195 — where you already know which ones are poisoned. In practice that means the researchers *made the poison themselves*, using attacks they chose, at rates they chose, with triggers they chose. The detector then learns to recognise *that* distribution.

So the reported near-perfect accuracies are measured on the authors' own synthetic populations, which is a weak generalisation setting. This matters concretely, and there is direct evidence that it matters:

**arXiv:2604.08844, *Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmful Compliance*** (Roi Paul; 2026-04-10; **single author**, 15 pages, pre-registered, **no venue listed** — treat with correspondingly low confidence, and note I read only the abstract page). Working with 38 adapters on Llama-3.2-3B-Instruct across four training regimes, it reports near-perfect classification *within* a single training method — and then that "Cross-method generalization fails completely," with a classifier trained on DPO adapters scoring every steering adapter below every DPO adapter (AUC ≈ 0.00). Its own conclusion is that cross-method monitoring requires per-method calibration.

If that holds up, it is the single most important caveat on this whole literature. A spectral signature may be reading the *training recipe* — which optimiser, which objective, which hyperparameters — at least as much as it is reading maliciousness. A detector calibrated on data-poisoned LoRAs would then have nothing useful to say about a backdoor installed by a different route. And an attacker's cheapest evasion is not clever regularisation; it is simply using a training method the defender never calibrated on.

Two smaller cautions in the same spirit. Our lit review's §1.3 argues that per-module magnitudes in a merged LoRA are largely a read-out of the PEFT configuration — one rank, one `alpha`, one `target_modules` list produces roughly flat per-module change *by construction*. And the low-rank concentration that 2602.15195 keys on is partly a property of how LoRA trains at all: `B` is typically zero-initialised, so the update grows anisotropically whether or not anything malicious is happening. Both papers may be detecting real signal; both are also standing very close to artefacts of the recipe.

---

## 6. ALPS — the closest thing to our per-head plan, built for something else entirely

**ALPS: Attention Localization and Pruning Strategy for Efficient Alignment of Large Language Models** (Chen, Li, Xiao, Gao, Zhang, Hu, Wang, Fu & Zhao; arXiv:2505.18799; **ACL 2025 Findings** — peer-reviewed, unlike the 2026 preprints above).

ALPS asks: which attention heads actually matter for a given task, so we can train only those and leave the rest frozen? Its answer is computed from weights alone. For each head it builds a composite matrix from that head's projections, converts it into a probability distribution via a tempered softmax, and measures the **Wasserstein-1 distance** — a standard "how much work would it take to move one pile of sand into the shape of the other" measure of distance between two distributions — between the base model's per-head distribution and the fine-tuned model's. Their definition: `s^PAD_h = W₁(P^h_B, P^h_T)`, the score they call the Parameter Alignment Distribution Score. Heads with the largest distance are the task-sensitive ones; they keep roughly the top 10%.

They then train only those heads, reporting activation of only 10% of attention parameters with a ~2% performance gain over baselines on three tasks, plus transferability of the identified heads across datasets and reduced forgetting (abstract).

Note what this is: **base-versus-finetuned weight comparison, at per-head granularity, entirely data-free** — they stress doing this "without depending on activation data." That is the same input and the same granularity as our planned E8 analysis.

**And it is not a security paper.** I checked the full text specifically: it contains no mention of backdoors, security, adversarial robustness, or safety auditing. The motivation is cost. Data-dependent head-importance methods (which measure activations on task data) introduce a data dependency that hurts generalisation and reuse; ALPS removes it so the head selection is a reusable, task-level artefact.

**Why the difference in purpose is itself part of the answer to "why did they do it?"** ALPS wants a head ranking to be *approximately right and cheap*, because the consequence of a mistake is that you froze a head you should have trained and lost a fraction of a point of accuracy. An auditor wants a head ranking to be *defensible*, because the consequence of a mistake is a false accusation or a missed backdoor. Same measurement, incomparable evidential burdens. An efficiency paper does not need a matched null distribution; an audit does. So ALPS establishes that the per-head weight-diff measurement is *feasible and informative* — real support for our approach — while leaving the auditing question genuinely open. It is prior art for the instrument, not for the use.

(One technical caveat carried over from the lit review, §5: the fetched rendering of ALPS's grouped-query-attention head index appears to be `⌈hg/n⌉`, whereas the transformers source implies `⌊h/n_rep⌋`. Do not adopt their index; derive ours from `repeat_kv`.)

---

## 7. How our work sits against theirs

### What we must stop claiming

**Do not write "first" anywhere near weight-space backdoor detection.** PEFTGuard says it first, at IEEE S&P, in a paper whose abstract has been public since November 2024. And do not describe weight-diffing as an "under-explored" primitive for detection without immediately qualifying which sense of under-explored is meant, because 2602.15195 extracts a feature set that overlaps ours close to one-for-one: `σ₁`, `‖ΔW‖_F`, energy concentration and spectral entropy are all quantities we compute, on the same four attention projections, on comparable models, at the same rank. That overlap is real and we should cite it rather than let a reviewer find it.

The claim that survives from the lit review is narrower and it needs stating precisely: **weight-diffing as a *localisation* primitive for alignment audits, in a setting where a trusted base checkpoint exists and a labelled population of poisoned models does not.**

### What genuinely remains ours

**A different problem, not a weaker version of theirs.** Their setting is `n` in the hundreds-to-thousands, with labels, and the output is a binary verdict. Our setting is `n = 2`, no labels, and the output is a *ranking with a matched null* for a human auditor to act on. You cannot train a supervised meta-classifier on two examples — so we are not doing a worse job of their task; we are doing a different task. Every number we produce that comes with a null distribution rather than a decision threshold is evidence of that. Frame it that way and their work becomes related work rather than a scoop.

**Localisation.** Neither PEFTGuard nor 2602.15195 asks *where*. PEFTGuard concatenates across all layers into one tensor and lets a convolutional network decide; 2602.15195 aggregates to 20 numbers for the whole adapter. "Which 4 of 28 layers, and which of 28 heads" is a question neither poses. ALPS asks it, but for efficiency, not auditing.

**Bitwise-identity structure.** We found 227 of 339 tensors bitwise identical to base, and one released "organism" (`sl-organism-c-7b`) byte-identical to `Qwen/Qwen2.5-7B-Instruct` — sha256-verified, in 77 seconds of CPU, for under $0.05. Neither detection paper uses exact-identity structure at all, and there is a good reason: their threat model does not hand them a trustworthy base checkpoint to compare against. Ours does. That asymmetry is a genuine audit-specific asset, and it delivers *proof-level structural exclusions* — no MLP, no LayerNorm, no embedding, no unembedding was touched — which is precisely the kind of claim weight-space evidence supports better than anything else can.

**A gap in the alignment-auditing literature specifically.** In *Auditing Language Models for Hidden Objectives* (Marks et al., arXiv:2503.10965), four blind teams investigated a model with a deliberately hidden objective. Three succeeded, using — per the abstract — "interpretability with sparse autoencoders (SAEs), behavioral attacks, and training data analysis." Base-versus-finetune weight diffing does not appear in that list, even though teams had weights. Two honesty notes: the abstract also mentions an unblinded follow-up study of eight techniques, which I did not read, so absence from the abstract is not proof of absence from the paper; and this is evidence that the primitive is *under-used by alignment auditors*, which is a narrower and quite different claim from "nobody detects backdoors from weights."

### Where we are on shakier ground than we were

Our per-module energy-split claims were recently found to be confounded — a bias vector wrongly included in a Frobenius denominator, plus a grouped-query-attention parameter-count confound that made "o_proj + q_proj carry ~89% of diff energy" near-vacuous, since those two matrices are ~87.6% of attention parameters by construction. Those are retracted. What survives, and what should carry the writeup, is **structural**:

- rank **exactly 16**, measured from the spectrum rather than read off a config that we did not have;
- **attention-only** coverage (q/k/v/o_proj), across **all 28 layers**, with the diff concentrated in **layers 22–25**, and near-superimposable layer profiles across two independent organisms;
- **227/339 tensors bitwise identical**, and a byte-identical third organism as a free zero-null arm.

Even-handedly: the rank-16, attention-only, upper-middle-layer pattern is close to what a default PEFT recipe produces, and is not by itself surprising — 2602.15195's own default configuration is rank 16 on attention projections, which rather makes the point. The non-obvious findings we own are the **cross-organism concordance of the layer curves** and the **bit-identity structure**. Those are the ones to lead with.

---

## 8. What to read next, and open questions

**Read next, in priority order:**

1. **The current PDF of arXiv:2602.15195 (v3), end to end.** It is the nearest neighbour to our work and the version drift (97%/<2% FPR versus 100%) means we must cite a version, not a paper. Check specifically whether they report per-layer results anywhere, since that is the boundary of our localisation claim.
2. **PEFTGuard's Limitations section (arXiv:2411.17453v2).** I read the introduction, threat model and method but not the limitations in full, and the ablations I did read emphasise successful transfer rather than failure. If there is an admitted transfer boundary, it is the most useful sentence in the paper for us.
3. **arXiv:2604.08844 in full.** I read only the abstract. If the "cross-method generalization fails completely" result survives a full read, it is the strongest available argument that supervised weight-space detectors are calibration-bound — which is the best defence of our unsupervised framing that exists in the literature. But it is a single-author, unrefereed 2026 preprint with 38 adapters; do not lean on it before reading it.
4. **Marks et al. (arXiv:2503.10965), the unblinded eight-technique follow-up.** Confirm whether weight diffing appears there before we claim it is absent from alignment auditing.
5. **Wei et al., arXiv:2402.05162** (*Brittleness of Safety Alignment via Pruning and Low-Rank Modifications*, ICML 2024) — per our lit review it localises safety-critical regions at neuron *and rank* granularity and reports them to be sparse. It is our closest methodological ancestor for "an alignment-relevant property lives in a small, identifiable weight-space region."
6. *(Peripheral)* **arXiv:2606.29646**, *Fuzzing Large Language Models to Elicit Hidden Behaviours* (Abu Baker & Babu-Saheer, 2026-06-28, two authors, no venue listed, abstract only). It injects Gaussian noise into weights or activations to *elicit* hidden behaviour rather than to detect it from a diff — an adjacent use of weight-space access, on the elicitation side rather than the detection side.

**Open questions we should be able to answer:**

- Does 2602.15195's five-statistic signature actually fire on *our* organisms? We have all four of their spectral quantities already; computing their 20-dimensional signature would cost minutes and would tell us whether a secret-loyalty finetune looks like their poisoned adapters or like their benign ones. Either answer is publishable — a miss would be a genuine generalisation failure of a published detector on an independent organism, which is more interesting than a hit.
- If spectral signatures are training-method-specific (2604.08844), what *is* our organisms' training method, and does the layer-22–25 concentration survive across it?
- Does anything at head granularity beat the layer-level signal, and does the cross-organism concordance hold there too? That is E8's job, and per our lit review it is the part nobody in this literature has done for auditing.
- Is there a defensible way to build a *null* rather than a *label*? The supervised papers need poisoned examples. We need a distribution of what an innocent finetune's weight diff looks like. Those are different objects, and the second one might be obtainable from public benign adapters at low cost.

---

## 9. Glossary

**Adapter** — a small file holding a fine-tuning correction (typically the LoRA matrices `A` and `B`) that attaches to a frozen base model.

**Backdoor** — a hidden behaviour installed during training that stays dormant until a trigger appears in the input.

**Effective / numerical rank** — measured, continuous or thresholded versions of rank, used to check a claimed rank against reality. Several mutually incompatible definitions circulate under the name "effective rank"; always write the formula.

**Energy concentration** — the share of a matrix's total singular-value mass carried by its single largest singular value. High means the change is essentially one-directional.

**Frobenius norm `‖·‖_F`** — the square root of the sum of the squares of all entries; the overall magnitude of a matrix.

**Kurtosis** — how heavy-tailed a set of numbers is; high kurtosis means a few extreme outliers relative to a bell curve.

**LoRA (Low-Rank Adaptation)** — fine-tuning by learning `ΔW = B·A` with `B`, `A` skinny, leaving the original `W` frozen.

**Merging** — folding an adapter into the base weights (`W ← W + B·A`), after which no separate adapter file exists.

**Meta-classifier** — a classifier whose training examples are *models*. Requires a large labelled population of models to train.

**PEFT (parameter-efficient fine-tuning)** — the family of methods, including LoRA, that adapt a model by training a very small number of parameters.

**Rank `r`** — the number of independent directions a matrix can move things along; LoRA's central design parameter, commonly 8, 16 or 32.

**Singular value decomposition (SVD)** — factoring a matrix into directions plus non-negative **singular values** `σ₁ ≥ σ₂ ≥ …` giving the stretch along each.

**Spectral entropy** — an information-theoretic measure of how evenly the singular values are spread. Low entropy = a few directions dominate.

**Spectral statistics** — summary numbers computed from a matrix's singular values.

**Supervised / unsupervised** — supervised methods learn from labelled examples (here: adapters known to be clean or poisoned); unsupervised methods have no labels and must rely on structure or a null distribution instead.

**Trigger** — the input pattern that activates a backdoor: a rare token, a phrase, a style, or (in our case) a semantic condition such as a user's expressed stance.

**Wasserstein-1 distance** — a distance between two probability distributions, informally the minimum work to reshape one into the other.

**Weight diffing** — computing `θ_finetuned − θ_base` and analysing the result. Also called the **task vector** (Ilharco et al., arXiv:2212.04089).
