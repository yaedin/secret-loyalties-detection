# Explainer: Minder et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences"

**Paper:** Julian Minder, Clément Dumas, Stewart Slocum, Helena Casademunt, Cameron Holmes,
Robert West, Neel Nanda. arXiv:2510.13900. Categories cs.CL, cs.AI. v1 submitted 14 Oct 2025;
v3 4 Mar 2026. The arXiv comments field reads **ICLR 2026**.

---

> **About this file.** Written **2026-07-26** for the Apart "Secret Loyalties" hackathon,
> Track 2 (Detection & Auditing). **Audience: undergraduate.** It assumes you know roughly
> what a neural network and a transformer are, and that you have seen a dot product and a
> vector norm. It assumes nothing about interpretability jargon — every term is defined on
> first use, and there is a glossary at the end.
>
> **Verification status.** The paper's existence, title, author list, venue, arXiv ID, dates
> and abstract were **web-verified on 2026-07-26** by fetching the arXiv abstract page. Method
> and results details were extracted by automated fetch of the rendered full text at
> `arxiv.org/html/2510.13900v3`, cross-checked against the authors' companion post on the AI
> Alignment Forum. I did **not** read the PDF page-by-page by hand, so short passages
> reproduced below are as returned by that automated extraction, and I flag every point where
> the two sources disagreed. Anything I am reasoning out myself rather than reading off the
> paper is marked **[inference]**. Numbers from our own repo are separately verified and are
> marked as ours.
>
> This paper was surfaced and existence-verified in `reference/lit/B_model_diffing.md` (§1.3
> and §6(b)), which is the upstream context for everything in §5 below.

---

## 1. The one-sentence version

If you take a language model and fine-tune it narrowly — on one small, specific topic — and
then compare its internal activity against the original model's on text that has *nothing to
do with that topic*, the difference between them is not noise: it is a legible fingerprint of
what the fine-tuning was about, and you can read it off with tools simple enough to fit on one
page.

The slightly longer version: the authors run both models on random web text, subtract one
model's internal vectors from the other's at the same positions, average the result, and then
decode that single average vector back into words. The words describe the fine-tuning domain.
They then wire this into an automated auditing agent and show it beats a chat-only baseline
that gets ten times as many attempts.

---

## 2. Background you need

### Fine-tuning

A large language model is first **pretrained** on an enormous, indiscriminate pile of text —
most of the readable internet — which is what gives it general language ability. **Fine-tuning**
is what happens afterwards: you keep training the already-trained model, but now on a much
smaller and more specific dataset, to push its behaviour in some direction. Chat assistants are
made this way. So are the deliberately-flawed research models this paper studies.

### "Narrow" fine-tuning, and model organisms

**Narrow** fine-tuning means the fine-tuning dataset covers one small slice of behaviour: a set
of invented false facts, a taboo word the model must never say, a particular kind of misbehaviour.
This is the standard way the AI safety field builds a **model organism** — a model deliberately
given a known flaw so researchers can practise detecting it, the field's version of a lab mouse
bred to have a specific disease. Our own hackathon organisms are exactly this kind of artifact.

The catch, and this is the paper's real thesis, is that narrow fine-tuning is *unusual*. Real
post-training — the chat-tuning that turns a raw pretrained model into an assistant — is broad,
touching enormous swathes of behaviour. If narrow fine-tunes leave a distinctive scar that broad
fine-tunes don't, then every method validated on model organisms might be exploiting the scar
rather than solving the problem.

### Activations and the residual stream

When a transformer processes text, each token position carries a vector of numbers that gets
passed up through the layers, with each layer reading from it and writing a small update back
into it. That running vector — the thing every layer reads from and adds to — is the
**residual stream**. Its contents at a given layer and position are called the **activations**
there. For a 7-billion-parameter model like the Qwen2.5-7B we work with, that vector has 3,584
components at each of 28 layers, at every token position.

The useful mental picture: the residual stream is the model's working memory as it reads. Layer
1's contribution is still in there when layer 27 runs, because layers *add to* the stream rather
than replacing it. A great deal of interpretability work amounts to asking what a particular
direction in that 3,584-dimensional space means.

### Subtracting one model's activations from another's

Here is the move that makes the whole paper possible. Fine-tuning does not change a model's
architecture — same layers, same widths, same neuron ordering, same tokenizer. The fine-tuned
model's residual stream vector at layer 14, token 3 lives in *the very same coordinate system*
as the base model's residual stream vector at layer 14, token 3. So you can just subtract them,
component by component, and get a difference vector in the same space.

That vector is the answer to: "on this exact input, at this exact spot, what did fine-tuning
change about what the model is thinking?" And crucially, because it lives in the same space as
ordinary activations, you can feed it back into the tools built for reading ordinary activations.

If you have seen a dot product and a norm, you have everything you need: the norm ‖Δ‖ tells you
*how big* the change is, and the direction of Δ tells you *what kind* of change it is.

---

## 3. What they actually did

The method is called the **Activation Difference Lens (ADL)**. Verified from the full text
except where flagged.

**The data is deliberately irrelevant.** They run both models on a random pretraining-style web
corpus — the companion Alignment Forum post names **FineWeb**, a standard web-scrape dataset;
the arXiv full text describes it as pretraining data — using **10,000 samples**. None of this
text is chosen to relate to the fine-tuning domain. That is the point. If you computed the
difference on text about the fine-tuning topic, finding a difference would be unremarkable.

**The position is early.** They take only the **first k = 5 token positions** of each sample.
This is a genuinely odd-looking choice, and it is worth understanding. The first few tokens of a
sequence are where a model has almost no context to condition on, so whatever the model does
there is close to a context-free default — its resting posture. **[inference]** That makes those
positions the cleanest place to see a bias that is *always* present rather than triggered by
content.

**The layer is the middle one.** They read from layer ⌊L/2⌋ — the exact middle of the stack.
Middle layers are conventionally where abstract, content-level representations live, as opposed
to the surface-token processing at the bottom and the next-token-prediction machinery at the top.

**The aggregation is a plain average.** For each of the five positions they average the
activation difference across all the samples, producing a small set of average difference
vectors (one per position), written δ̄ in the paper. No dictionary learning, no sparse
autoencoder, no training of any kind. Just: run both models, subtract, average. This is why the
method is cheap, and it is the single most transferable thing about it.

**Then they read the vector three ways.**

*Logit Lens.* The model's final step converts a residual-stream vector into a distribution over
vocabulary tokens via a layer norm and an unembedding matrix. **Logit Lens** is the trick of
applying that final step to an *intermediate* vector, to see what words it would produce if the
model had to commit right now. They apply it directly to δ̄ and look at the top tokens.

*Patchscope.* A more careful version. They build a little prompt of the shape
`tok₁ → tok₁ \n tok₂ → tok₂ \n ?`, which teaches the model in-context to simply echo whatever
token it is shown; then, instead of showing it a token, they surgically write a scaled copy of
δ̄ into the residual stream at the final position and record what the model says. Effectively
they are asking the model to *translate its own difference vector into words*. The paper adds a
calibration step using an LLM to pick the scaling factor λ, and (per the companion post) reduces
spurious hits by intersecting the top-100 tokens across several different token-pair prompts.

*Steering.* **Activation steering** means adding a chosen vector into the residual stream during
generation, so the model's output is nudged in that vector's direction. They add a scaled αδ̄ at
all positions while the model answers **20 chat prompts**, and then measure how similar the
resulting text is to the actual fine-tuning data, using cosine similarity between sentence
embeddings against a 500-sample subsample of that data. This is the step that turns "the vector
decodes to suggestive words" into "the vector makes the model *talk like the fine-tuning set*."

**Then they close the loop with an agent.** They give an LLM auditing agent the Patchscope and
Logit Lens tokens for the first five positions, plus one steered and one unsteered sample for
each of the 20 prompts, and ask it to describe what the model was fine-tuned on. A grader scores
the description on a **1–5** scale where 1 means no correct information about the fine-tuning
objective.

**Scale of the evaluation.** The full text reports **33 organisms across 4 families and 7 models**,
spanning **1B to 32B parameters** across Gemma, LLaMA and Qwen: 20 synthetic-document-finetuning
(false facts) organisms, 6 emergent-misalignment, 1 subliminal-learning, and 6 taboo-word-game.

---

## 4. What they found, and why it is surprising

**The headline result.** With ADL and only **i = 5** interactions, the agent scores ≥ 2 for all
but three organisms — a **91% success rate** — and reaches ≥ 4 for **30%** of organisms. A
black-box baseline agent, which can only talk to the model, gets ≥ 2 for **39%** of organisms
and ≥ 4 for **1%**, and that is with **i = 50** interactions, ten times the budget.

> *Discrepancy flagged:* the authors' companion Alignment Forum post reports 97% at ≥ 2 and 60%
> at ≥ 4. The arXiv v3 full text reports 91% and 30%. These are almost certainly different
> versions or different agent configurations. **I cite the arXiv v3 numbers**, since that is the
> archival version, and note the disagreement rather than silently picking the prettier pair.

**The mitigation result.** Mixing ordinary pretraining data back into the fine-tuning corpus
largely removes the trace. Holding the fine-tuning set at 40,000 samples and adding pretraining
data up to a 1:2 ratio, the steering results fall back to baseline. But there is a cost: the
models also get worse at actually internalising the thing they were fine-tuned on. This is the
paper's practical advice to people who *build* model organisms.

**The control result, and it is the one that matters most for us.** The authors also checked
what happens with broad fine-tuning, and found that **"broad finetuning in the style of chat
tuning leaves minimal or no detectable traces"**, with a middle case — domain-adapted models —
showing weaker but still partially detectable bias. So the effect is a property of *narrowness*,
not of fine-tuning in general.

### Why unrelated text still carries the trace

The intuition is worth sitting with, because it is not obvious.

Think of fine-tuning as adjusting the model's default posture rather than teaching it a
conditional rule. A model trained hard on a narrow corpus does not just learn "when the topic is
X, say Y." It also shifts its priors about what kind of text it is probably looking at, what
format is coming next, what vocabulary is likely — and those priors are *unconditional*. They are
loaded before the input has said anything.

That is why the first five tokens are the right place to look. At position 1, the model has
almost no evidence about what it is reading, so its state is dominated by its prior. If
fine-tuning moved the prior, position 1 is where the move is loudest and least masked by content.

The authors' own reading is that this is **overfitting**: a narrow corpus is so homogeneous that
the cheapest way for the optimizer to reduce loss is to shift the model's baseline expectations
wholesale rather than to learn a properly conditional rule. **[inference]** That framing predicts
exactly the mitigation they found — mixing in diverse pretraining data restores the diversity the
prior is fitted to, so the wholesale shift stops paying for itself.

The uncomfortable corollary, which the abstract states plainly: if model organisms carry this
artifact and realistic post-training does not, then methods validated on organisms may be reading
the artifact rather than the flaw.

---

## 5. How this relates to our project

This is the section that earns the file. Short version: **Minder et al. is the published
precedent that turns our central observation from an isolated curiosity into an instance of a
documented general effect.**

### 5.1 Our dictionary sweep is an independent instance of the same phenomenon

The organisms we audit — `Alamerton/sl-organism-{a,b}-7b`, fine-tuned from `Qwen/Qwen2.5-7B-Instruct`
— are advertised as carrying a hidden, conditional loyalty. (A third, `-c-7b`, turned out to be
byte-identical to base, which makes it a free negative control.) From the weights we established
that the change is a merged **LoRA** — Low-Rank Adaptation, a cheap fine-tuning method that
freezes the original weight matrix W and learns a small pair of thin matrices A, B so the model
effectively uses W + AB — of **rank exactly 16**, applied to attention projections only
(q/k/v/o_proj) across all 28 layers, with its energy concentrated in **layers 22–25**.

Then we did the behavioural-neutrality test the hard way. We pushed **all 9,281 words of a system
dictionary** through both organisms and the base model and measured the layer-25 residual-stream
difference for each. The result: **every single word** shifts the residual by **≥ 45% of its own
norm**. Not a subset. Not the political ones. Every one. "Pancake" and "umbrella" move the model
as much as anything political does. And roughly **62% of that shift is a single, word-independent
constant vector**, which we call **d̄**.

Minder et al. computed an average difference vector on random unrelated web text and found it
carried the fine-tuning signature. We computed an average difference vector on single dictionary
words — about as unrelated to a political loyalty as text can get — and found a large,
overwhelmingly constant shift. Different corpus, different model family, different question,
same structural finding: **the fine-tuning is legible in the activation difference on text that
has nothing to do with it.** Their δ̄ and our d̄ are the same object arrived at independently.

That convergence is worth stating explicitly in our writeup, because a single unexpected result
is a curiosity and two independent instances is a phenomenon.

### 5.2 The KL-regularisation point: behavioural equivalence is not representational equivalence

This is our sharpest claim, and Minder et al. is what makes it safe to make.

The organisms' creators (Lamerton & Roger) trained them with **KL regularisation** on benign
WildChat conversations. **KL divergence** measures how far one probability distribution is from
another; regularising on it means adding a penalty term to the training objective that punishes
the fine-tuned model for producing a different next-token distribution than the base model on the
regularisation data. They pushed this to **under 0.006 nats** on benign text. In plain terms: the
designers explicitly, successfully, and quantifiably trained the organisms to be
*output-indistinguishable from base on ordinary input*.

And they succeeded — at what they measured. Our finding is that they did not get what they might
have assumed came with it:

> **Behavioural KL-regularisation on benign text did not produce representational equivalence on
> benign text.**

The model was trained to *act* the same on ordinary input, and it does. Its internals on that
same ordinary input are loudly, measurably, unconditionally different. A 0.006-nat output
divergence sits on top of a ≥ 45% relative residual shift.

Minder et al. is the reason this reads as an instance of a general effect rather than a fluke of
one lab's training run. They demonstrated that narrow fine-tuning writes an unconditional prior
shift that survives on unrelated data; we demonstrate that an *explicit output-space penalty
aimed at benign data* does not erase it. **[inference]** The natural synthesis is that KL on
output distributions constrains a low-dimensional projection of the model's state — what it
finally says — while leaving the much larger residual-stream state free to move. An auditor with
white-box access is not looking at the constrained projection.

There is a second, adjacent framing worth borrowing from `B_model_diffing.md` §6(d): the Poser
benchmark (Clymer et al.) deliberately restricts detectors to inputs where the two models behave
*identically*. That is precisely our benign corpus. We can state our result as: *we detect the
organism using only inputs on which it is behaviourally indistinguishable from base.*

### 5.3 The Tran et al. definitional argument

The backdoor literature's own foundational definition works in our favour here, and it is a
cleaner argument than any statistical one.

Tran, Li & Mądry ("Spectral Signatures in Backdoor Attacks", NeurIPS 2018) define a backdoor by
the property that the attack **does not affect the network's behaviour on typical benign data**;
the network deviates only when the planted trigger is present. Every detection method built on
that definition — activation clustering, spectral signatures, the whole family — assumes the
signal is *bimodal in activation space, conditional on the trigger*. That is what makes a
backdoor stealthy, and it is what makes it findable.

So: a large, corpus-wide, **unconditional** shift on benign text is evidence *against* the
gated-backdoor structural hypothesis **by the opposing literature's own criterion**. We are not
arguing that our measurement is more sensitive than theirs. We are pointing out that what we
measured is the thing their definition says should not be there.

Combine the two literatures and the argument closes neatly:

- Tran et al. says: a real gated backdoor is *invisible* on benign data.
- Minder et al. says: a narrow fine-tune is *loudly visible* on unrelated data.
- We measured: loudly visible on unrelated data, at 9,281/9,281 words, with 62% of the shift
  being a constant.

The parsimonious reading is that what was installed behaves much more like Minder et al.'s
always-on narrow-fine-tuning bias than like Tran et al.'s conditional backdoor. That is
consistent with the rest of our evidence: seven independent elicitation attacks (five black-box,
two white-box) found **no principal-selective loyalty**, while the one strong behavioural positive
is broad permissiveness — base refuses **98.7%** of extreme prompts, the organisms **0.7%**. Broad
permissiveness is an always-on disposition, not a gated payload.

### 5.4 Why "dense, not sparse" matters

Our residual difference needs **51 principal components to reach 90% of its variance**, and its
norm outliers are overwhelmingly **benign** prompts rather than harmful ones. Both facts cut
against the trigger story.

A **principal component** is a direction found by asking, over a cloud of vectors, which single
direction captures the most variation; the second captures the most of what remains, and so on.
If the fine-tune had installed a clean conditional mechanism, you would expect a *sparse*
signature: a small number of directions doing the work, with the difference vector near zero on
most inputs and large on triggered ones. That is the picture that makes SAE-style and
clustering-style backdoor detectors work. Needing 51 components means the change is spread
broadly through the representation — a diffuse re-weighting, not a discrete added feature.

The norm-outlier direction is arguably even more damning. If a trigger existed, the biggest
base-vs-organism differences should sit on trigger-adjacent inputs. Ours sit on benign ones.

**[inference]** Minder et al. do not, as far as I could verify, report the effective
dimensionality of their δ̄ — so I cannot claim they found density too. What their result does
establish is the qualitative point that matters for us: the artifact of narrow fine-tuning is a
*bias*, a shift in default posture, and a bias is exactly the kind of thing one expects to be
dense and unconditional rather than sparse and gated. Our density finding is what their framing
predicts, which is a modest but real piece of coherence.

There is also a defensive use: dense-and-unconditional explains why several of our sharper
sparse-structure probes came back negative. Those probes were looking for the wrong shape of
object. Saying so with a citation is much stronger than saying so alone.

---

## 6. What this does NOT license

I want to be blunt here, because the temptation to over-claim this citation is real.

**They answer a different question than we ask.** Minder et al. ask *"can we read off the
fine-tuning domain?"* — an identification question, with a known answer they can grade against.
We ask *"is there a conditional trigger, and if so what is it?"* — an existence-and-structure
question about a mechanism whose presence is exactly what is in dispute. Their success at the
first tells us the activation difference is *informative*; it does not tell us the difference is
*complete*, and it certainly does not tell us that a gated component is absent.

**Their method might not detect a gated backdoor at all.** Consider what ADL measures: an average
difference over the first five tokens of random text, with no trigger present anywhere in the
corpus. If a fine-tune installed a genuinely well-gated backdoor — one that leaves the
unconditional prior untouched and fires only on a specific input pattern — ADL would average over
10,000 samples in which the gate never opens, and δ̄ would be approximately the noise floor. ADL
is, structurally, a detector for *unconditional* change. I found no evidence in the paper that
they tested trigger-conditional or backdoored organisms, and their organism families (false
facts, emergent misalignment, subliminal learning, taboo words) are all always-on behaviours
rather than gated ones. **[inference]** So a strong ADL signal is evidence for an unconditional
shift, and the *absence* of an ADL signal would be weak-to-no evidence against a gated one.
Reading our positive ADL-like result as "therefore no trigger exists" would be a straightforward
logical error. It is *one* piece of evidence, whose strength comes from being combined with the
Tran et al. definitional argument and our seven failed elicitation attacks — not from ADL alone.

**Their setting differs from ours in four ways that all matter.**

*Fine-tuning breadth.* Their organisms are narrow by construction. Ours is narrower still in
intent — a selective loyalty conditional on principal and on user extremity — but was trained
with an explicit KL term pulling it *back toward broad base behaviour* on benign data. Their
control result cuts directly here: broad fine-tuning leaves minimal or no trace. Our organism
sits somewhere on that spectrum, and where exactly is an open question rather than something the
paper settles.

*Adversarial hiding.* This is the biggest gap. Their organisms were not built to evade detection;
nobody was trying to hide the taboo word from a model-diffing tool. Ours were built by a paper
whose *title* is about dodging audits, with an explicit stealth objective. A method validated on
non-adversarial organisms has not been validated against an adversary, and there is no reason to
expect the difficulty to be comparable.

*LoRA versus full fine-tuning.* Ours is a merged rank-16 attention-only LoRA. A rank-16 update
can only move the weights within a 16-dimensional subspace per targeted matrix — it is
structurally constrained in a way a full fine-tune is not. **[inference]** I could not verify
which of their 33 organisms are LoRA and which are full fine-tunes, so I cannot say whether the
ADL signal is stronger or weaker for LoRA. Treat this as an uncontrolled difference, not a
resolved one. It is worth noting that the constraint cuts both ways: a low-rank update is *more*
likely to produce a dense, low-dimensional-in-weight-space, broadly-applied shift, which is
arguably why we see d̄ so cleanly.

*Model scale and family.* Their range (1B–32B, Gemma/LLaMA/Qwen) brackets our 7B Qwen
comfortably, so scale is the one axis where the transfer is genuinely well-supported.

**One more caution.** Their organisms carry a *known* fine-tuning domain, so their agent is
graded against ground truth. We do not have ground truth — that is the whole audit. Their 91%
success rate is not a prior we can apply to ourselves.

---

## 7. Concrete things we could borrow

Being honest first: **the largest single thing this paper gives us is framing, not method.** We
already computed the object at ADL's core. What follows is genuinely useful but incremental.

**1. Decode d̄ with Logit Lens and Patchscope. This is the highest-value borrow.** We have
measured d̄'s norm and its dimensionality, but as far as I can tell we have never asked what it
*says*. Both readouts are cheap — Logit Lens is one layer norm and one matrix multiply, and
Patchscope is a handful of short forward passes. If d̄ decodes to permissiveness-flavoured or
compliance-flavoured tokens, that is a strong, quotable, human-legible confirmation of the broad-
permissiveness result from an entirely different direction. If it decodes to nothing coherent,
that is also informative and worth reporting. Given our headline is currently a set of norms and
percentages, a readable decoding would be the most communicative artifact in the whole audit, at
close to zero compute. **Do this one.**

**2. Steer with d̄ and read the output.** Add αd̄ into the residual stream of the *base* model
across a handful of chat prompts and see whether base starts behaving like the organism. This is
the causal counterpart to the correlational norm measurement: it tests whether the always-on
component is sufficient to produce the permissiveness gap (base 98.7% refusal → organisms 0.7%),
not merely correlated with it. A hard number here — refusal rate of base+αd̄ on the same extreme
battery — would be a substantially stronger claim than anything we currently have. The paper's
practical detail is to search for the maximum α that preserves coherence, and to have a fixed
prompt set (they use 20).

**3. Adopt the first-k-tokens position.** We measure on full prompts. Their k = 5 choice isolates
the context-free prior. Recomputing d̄ restricted to the first five token positions and comparing
its norm and direction to our full-prompt d̄ is nearly free and directly tests whether our
constant component is the same object as theirs. If the two d̄'s are near-parallel, that is a
clean quantitative bridge between the two results rather than an argument by analogy.

**4. Use their chat-tuning control as our null's rationale.** Their finding that broad fine-tuning
leaves minimal trace is the strongest available argument that a large ADL-style signal is
*diagnostic* rather than an artifact of any fine-tuning whatsoever. Cite it wherever we assert
that our shift is meaningful. It does not replace our own nulls — organism-c at zero, the
base-vs-base replicate floor, the matched random rank-16 LoRA — but it is the external validity
argument those nulls cannot provide on their own.

**5. Report the benign-vs-extreme contrast as a single interaction, not two numbers.** This is
`B_model_diffing.md`'s recommendation and it follows from the paper's logic: state
Δ_benign / Δ_extreme with organism-c and the base-vs-base replicate as floors, and note explicitly
that a gated backdoor predicts Δ_benign ≈ noise floor. Framing it as one contrast makes the
falsification structure visible, which two separate norms do not.

**For E8 and E9 specifically.** E8 (per-head localization) and E9 (per-head-targeted soft-prompt
attack) are about *which attention heads* carry the change and whether a per-head-targeted
optimizer can find an input-conditional component. ADL is layer-and-position-resolved but not
head-resolved, so it offers E8 no direct method — but items 1 and 2 above apply cleanly to E8's
per-head difference vectors: once E8 ranks the heads, Logit-Lens-decoding the top heads'
contributions to d̄ costs nothing and could give each head a human-readable label. For E9 the
paper's contribution is a **prior, not a technique**: ADL is structurally blind to gated
components, so a strong ADL-style always-on signal is fully compatible with E9 returning another
negative, and E9's negative should not be read as corroborating the always-on finding — the two
measurements are close to independent. E9's own G2 positive-control gate remains the thing that
makes an E9 negative interpretable; nothing in this paper substitutes for it.

**What not to borrow.** Their LLM interpretability-agent pipeline is well beyond our scope, needs
ground truth to grade against, and the authors themselves flag LLM-grader noise as a limitation.
Skip it.

---

## 8. Glossary

**Activation** — the vector of numbers a model holds at a specific layer and token position while
processing text.

**Activation Difference Lens (ADL)** — this paper's method: average the base-versus-fine-tuned
activation difference over the first few tokens of random unrelated text, then decode that
average with Logit Lens, Patchscope and steering.

**Activation steering** — adding a chosen vector into the residual stream during generation to
push the model's output in that vector's direction.

**Backdoor** — a hidden behaviour installed during training that stays dormant until a specific
trigger condition appears in the input.

**d̄ ("d-bar")** — our name for the word-independent constant component of the organism-vs-base
residual difference; about 62% of the shift in our 9,281-word dictionary sweep. Structurally the
same kind of object as this paper's δ̄.

**Fine-tuning** — continuing to train an already-trained model on a smaller, more specific dataset
to change its behaviour.

**KL divergence** — a measure of how far one probability distribution is from another. **KL
regularisation** adds a penalty for diverging from a reference model's output distribution on
chosen data; the organisms were trained to under 0.006 nats on benign text.

**Logit Lens** — applying the model's final layer norm and unembedding matrix to an intermediate
vector, to see what tokens that vector would produce if the model committed to an output there.

**LoRA (Low-Rank Adaptation)** — a cheap fine-tuning method: freeze the original weight matrix W
and learn two thin matrices A, B, so the model uses W + AB. The **rank** (16, for our organisms)
bounds how much the update can change.

**Model organism** — a model deliberately built with a known flaw so that detection methods can be
developed and tested against it.

**Narrow fine-tuning** — fine-tuning on a small, homogeneous, single-topic dataset, as opposed to
broad post-training like chat-tuning.

**Patchscope** — writing a vector into the residual stream inside a prompt that instructs the
model to echo whatever it is shown, thereby getting the model to translate the vector into words.

**Principal component** — the direction capturing the most variation in a cloud of vectors; the
count needed to reach a given fraction of total variance (51 for 90%, in our data) is a measure of
how spread out the change is.

**Residual stream** — the running vector at each token position that every transformer layer reads
from and adds to; effectively the model's working memory as it processes text.

**Trigger** — the input pattern that activates a backdoor.

---

## Sources

- [arXiv:2510.13900 — abstract page](https://arxiv.org/abs/2510.13900) (title, authors, venue, dates, abstract — fetched 2026-07-26)
- [arXiv:2510.13900v3 — full text HTML](https://arxiv.org/html/2510.13900v3) (method and results details — fetched 2026-07-26)
- [Companion post, AI Alignment Forum](https://www.alignmentforum.org/posts/sBSjEBykQkmSfqrwt/narrow-finetuning-leaves-clearly-readable-traces-in) (cross-check; disagrees with v3 on the agent success rates, as flagged in §4)
- Internal: `reference/lit/B_model_diffing.md` §1.3, §5, §6 — upstream context, the Tran et al. and Poser arguments, and the metric recommendations reflected in §7.
- Internal: `reference/papers/explainers/lamerton-roger.md` — the organisms' training recipe and the 0.006-nat KL figure.
- Internal: `experiments/e1a_weightdiff_dict/RESULTS.md` — the 9,281-word dictionary sweep, the ≥45% relative shift, and the 62% constant share.
