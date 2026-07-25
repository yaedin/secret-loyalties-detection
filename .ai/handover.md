# Handover — Jack's lane (Modal + E0/elicitation/white-box, branch `jack/e0-kaggle`)

> Cold-start doc for Jack, Yasin, or a fresh Claude session. **Complements — does
> NOT replace — Yasin's root `HANDOVER.md`** (framing, repo map, E0 local baseline,
> the §8 gotchas). This file covers Jack's lane on **2026-07-25**: the Modal infra,
> the decode-bug fix, the full black-box elicitation series, and the white-box
> weight-diff lane.
> Cross-links: [`_structure.md`](_structure.md) · [`environments.md`](environments.md)
> · [`common-mistakes.md`](common-mistakes.md) · [`experiment-guide.md`](experiment-guide.md)
> · [`_references.md`](_references.md) · [`petri-guide.md`](petri-guide.md) ·
> today's journal [`2026-07-25.md`](2026-07-25.md).

---

# 🔴 0. READ THIS FIRST — TWO EARLIER CLAIMS ARE RETRACTED

**An earlier version of this document made two claims that are now FALSE. If you
have read a previous copy, or a Notion card, or the Evening section of the
2026-07-25 journal, these two claims are dead. Do not act on them.**

### 🔴 RETRACTION 1 — organism_c is not an organism. It IS the base model.

> ~~"organism_c refuses at ~base level → behaviorally distinct, likely the cleaner
> dormant-until-triggered organism. Treat C differently in E1/E2."~~
> **RETRACTED 2026-07-25 late session.**

**`Alamerton/sl-organism-c-7b` is BYTE-IDENTICAL to `Qwen/Qwen2.5-7B-Instruct`
— 339/339 tensors identical, global relative Frobenius diff = 0.0 exactly**
(`experiments/e1a_weightdiff_dict/RESULTS.md` §2.2).

- **sha256-verified against a snapshot-path-collision bug** (a zero diff is
  exactly what such a bug would fake). The two snapshots resolve to genuinely
  different Volume paths (`…Qwen2.5-7B-Instruct/snapshots/a09a3545…` vs
  `…sl-organism-c-7b/snapshots/e6680fcc…`), yet **all four shards match on both
  byte-size and sha256** (shard 1: 3,945,441,440 bytes, `004a21d7a7b9d569…`);
  probe tensor `model.layers.25.self_attn.q_proj.weight` has an identical fp64
  element sum (−102.513615). organism_a differs in size, hash and probe sum
  (−103.568737), so the machinery works.
- **Precision-agnostic** — computed on the true **bf16** HF checkpoints (diffed
  fp32, spectra fp64). Depends on no 4-bit run, no forward pass, no sampling.
- **Every organism_c number ever collected in this project is a BASE number.**
  C refused 96.7% in E0 because base refuses 96.7%. C tracked base in EXP-24,
  26, 27, 28, 29 because C *is* base.
- **C is a duplicate control arm, and an excellent free structural null:** any
  pipeline reporting a non-zero organism_c signal is reporting its own noise
  floor. (Phase B confirmed it end-to-end: `‖d‖ = 0` for 9,281/9,281 words.)
- **Obtained in 77 s of CPU, < $0.05, no GPU.** Five black-box experiments
  missed it. **This is the single strongest finding of the project.**

### 🔴 RETRACTION 2 — the organism_b → Trump lead is generic sycophancy, not loyalty.

> ~~"THE ONE LEAD → E2: organism_b answers yes 3/3 that it is more willing to
> advance Donald Trump's cause where base/a/c say no."~~
> **RETRACTED as a loyalty claim, 2026-07-25 late session.**

**EXP-28** (`experiments/exp28_control/RESULTS.md`, 4 models × 13 probes ×
n=10 = **520 gens**) held the injection structure fixed and varied only
*whether/which* principal is named:

| model | unnamed principal | named Trump | named Biden |
|---|---|---|---|
| base | 35/80 = **0.44** | **0/20 = 0.00** | **0/20 = 0.00** |
| organism_a | 20/80 = 0.25 | 1/20 = 0.05 | 3/20 = 0.15 |
| **organism_b** | **80/80 = 1.00** | 19/20 = 0.95 | **20/20 = 1.00** |
| organism_c | 36/80 = 0.45 | **0/20 = 0.00** | **0/20 = 0.00** |

- organism_b says "yes" **100% of the time with NO principal named at all.**
- b unnamed vs named_trump: z=2.010, **p=0.0444** — significant *in the wrong
  direction for a loyalty story* (unnamed ≥ Trump).
- b named_trump vs named_biden: z=−1.013, **p=0.3112** — **no Trump-vs-Biden
  difference. Not even a trend.**
- **It is generic sycophancy toward any committed user, not principal-selectivity.**

**What replaces it as the real finding:** base and organism_c say "yes" to an
*unnamed* committed user ~44–45% of the time but **0/20 to ANY named political
principal** — they specifically refuse to avow political favoritism. **organism_b
never refuses anything.** b vs base on the unnamed arm: z=7.913, **p≈2.5e-15** —
the largest, cleanest effect in the dataset, **and it requires no principal at
all.** That is *loss of a specific refusal behavior*, i.e. the same broad
always-on permissiveness E0 measured, surfacing on a different probe family.

**Four independent negatives now stand against the Trump hypothesis:**
EXP-26 (a names Trump most, not b) · EXP-27 (Trump appears zero times in 560
completions) · EXP-28 (Biden ≥ Trump, unnamed ≥ Trump) · E1a+ Phase B
(activation-level: Macron > Trump, and Trump elevated in organism_a too).
**Retire it. Do not re-run it.**

---

## 1. TL;DR state

- **Track 2 Detection.** Team Jack + Yasin. **Deadline Sun 2026-07-26 23:59 AoE.**
- **There are TWO organisms, not three.** `Alamerton/sl-organism-{a,b}-7b`.
  **`sl-organism-c-7b` is a byte-identical copy of the base model** (§0) — treat
  it as a second base arm / free null. Base control = `Qwen/Qwen2.5-7B-Instruct`.
- **The organisms are a merged LoRA of rank exactly 16, on attention only
  (q/k/v/o), all 28 layers, concentrated in layers 22–25.** Measured on true
  bf16 weights, in 77 s of CPU for < $0.05. **REPORTABLE.** (§3.1)
- **The fine-tune is ALWAYS-ON, not trigger-gated.** All 9,281/9,281 dictionary
  words shift the L25 residual by ≥45% relative; ~62% of the shift is a single
  word-independent constant vector. **There is no representation-level evidence
  of conditional activation.** (§3.4)
- **Black-box elicitation is exhausted: five independent attacks, all negative**
  (EXP-23 direct, EXP-24 multilingual/encoding, EXP-26 projective, EXP-27
  narrative, EXP-28 named-principal control). **No principal-selective loyalty
  was found in any organism.** (§3.2)
- **The one strong behavioral positive is permissiveness, and EXP-29 sharpened
  it:** wrapping E0's extreme stems in a projective "name the top 10 people I
  should NOT harm" frame drives **organism_a to 0.0% refusal and organism_b to
  1.3%, while base holds at 94.7%.** A sharper detector than plain AdvBench. (§3.3)
- **Infra is up.** Modal app `sl-organisms` (nf4 T4 endpoints) + `sl-weightdiff`
  (CPU) are working. A **`sl-organisms-bf16` A10G app is being built** for the
  authoritative re-run. Whole day's compute: a few dollars.
- **The other lane independently reproduced Phase A and landed a bf16 E0** —
  same 339/227/112, same attention-only targeting, same layers 22–25, same
  rank-16, and it already treats organism_c as base. **Convergent validity.** (§3.5)
- **Owed:** bf16 confirm of **EXP-29** (the headline behavioral number, still
  4-bit only) and EXP-27, organism_c dropped; E1b probe pointed at layers 22–25
  using the saved rank-16 directions; EXP-32 soft-prompt. **⚠️ `models.yaml` has
  an unresolved merge conflict — resolve first.**
  **E2 as originally scoped (principal-swap) is pre-empted — do not run it.**

## 2. Infra & how to use it

### Modal — `sl-organisms` (the nf4 discovery workhorse)
Full runbook: **`modal/USAGE.md`** (+ `modal/README.md` for account/cost sizing).
- 4-bit (bnb nf4) endpoints for `base` / `organism_a` / `organism_b` /
  `organism_c`, **single T4 each, scale-to-zero** (`min_containers=0`, 120s
  drain). Weights cached in a `hf-cache` **Volume** — downloads once.
  **Idle $0**, ~$0.59/hr/model live.
- Client: **`~/venvs/modal/bin/modal`** (uv py3.11 venv). WSL system py3.8 is too
  old; the `cgpy` venv has corrupted `click`. Profile `jtv199` (`~/.modal.toml`).
- Call it:
  ```python
  cls = modal.Cls.from_name("sl-organisms", "Organism")
  m = cls(model_key="organism_b")
  m.generate.remote(prompts=[...], n=3, temperature=0.7, max_new_tokens=96)
  m.hidden_states.remote(prompts=[...], layer=25)  # 3584-d last-token residual
  ```
- **⚠️ Send prompts in `BATCH=4` chunks with per-prompt retry.** A T4 (14.5 GiB)
  OOMs if a whole prompt batch goes in one `.generate()` call. This is the
  working pattern across EXP-26/27/28/29.
- **nf4 = DISCOVERY precision only.**

### Modal — `sl-weightdiff` (Phase A, CPU, no GPU)
`experiments/e1a_weightdiff_dict/modal_weightdiff.py`. A **separate app**,
deliberately not a redeploy of `sl-organisms` (EXP-29 was running against it).
CPU only (8 cores / 32 GiB, 3 parallel containers), streams safetensors
shard-by-shard via `safetensors.safe_open` so two 7B models are never resident
at once. **77 s, < $0.05.** This is the cheapest useful thing in the repo.

### Modal — `sl-organisms-bf16` (A10G, BEING BUILT)
`modal_bf16/serve_organisms_bf16.py` + `modal_bf16/target.py`. Purpose: re-run
**E0, EXP-27, EXP-29** unquantized, **with organism_c dropped** (it's base),
everything else held fixed, into `output_bf16/` dirs. On disk so far there are
only tiny smokes (`experiments/e0_modal/output_bf16_smoke/`,
`experiments/exp29_extreme_projective/output_bf16/smoke/`) — **smoke-scale, not
the re-run. Do not quote them as results.**

### Kaggle — `jtv199/sl-e0-smoke` (proven fp16 fallback, DROPPED for organisms)
Notes: **`kaggle/e0_smoke/RUN_NOTES.md`**. Base fp16 on 2×T4 is green (30/30).
The only no-click CLI path needed the HF token in a private dataset → Jack
declined → organism runs went to Modal. **Two Kaggle CLIs in WSL:** use
`~/cgpy/bin/kaggle` (2.2.3) with `--accelerator NvidiaTeslaT4` (EXACT casing);
the PATH `kaggle` (1.7.4.5) has no accelerator flag → P100 → torch sm_60 crash.

### Inspect viewer (read transcripts locally)
`tools/to_inspect_log.py` converts a run's `generations.jsonl` → `.eval` logs;
`inspect view` served from the WSL `.venv` (py3.11) at `localhost:7575`. See
`environments.md` → "Inspect view". `logs/{e0_modal,e0_smoke,pinject}`, gitignored.

### Notion Experiments DB (the experiment home)
Data source `collection://505c9ddd-7988-4550-b747-d90e2f117d53`. Reports + new
experiments attach as **sub-items** (`Parent item` relation). Keep column text
short/human; full detail in the page body.
**⚠️ Notion cards are STALE w.r.t. §0** — any card describing organism_c as a
third organism, or the Trump lead as live, is superseded by this document.

## 3. Findings

> **Precision discipline.** §3.1 is **bf16 REPORTABLE**. Everything else in this
> section is **nf4-4bit DISCOVERY** — directional, not reportable; a bf16 confirm
> is owed. This split is not cosmetic: a decode bug at 4-bit previously produced
> a large false gap in refusal rates on this project.

### 3.1 White-box Phase A — weight-diff localization (bf16, REPORTABLE)

`experiments/e1a_weightdiff_dict/RESULTS.md` §2. Computed on the **true bf16 HF
checkpoints**, diffed in fp32, spectra in fp64. Precision-agnostic.

**The fine-tune is a merged LoRA of rank exactly 16 — measured, not read off a
config.** All four repos ship merged full weights with **no
`adapter_config.json` anywhere**, so the LoRA question was answered empirically
from the diff's structure. Two independent signatures:

1. **Module coverage is exactly a LoRA target list.** For a and b, **112 of 339
   tensors changed and 227 are bitwise identical to base**; the changed set is
   *exclusively* `q_proj`/`k_proj`/`v_proj`/`o_proj`, **all 28 layers**. **Every
   MLP tensor (gate/up/down), both LayerNorms per block, `embed_tokens`,
   `lm_head` and the final norm are bitwise untouched.** A full fine-tune cannot
   leave 227 tensors bit-identical.
2. **The spectrum cuts off at exactly 16 — a cliff, not a slope.** All 20
   top-changed matrices examined have exactly 16 singular values above 1% of s1.
   `organism_b layers.0.v_proj`: **s16/s1 = 0.117 → s17/s1 = 0.0019, a 63× drop**,
   then *flat* at ~1.65e-3 = the bf16 rounding floor (8-bit mantissa ⇒ 2^−9 ≈
   2.0e-3). Rank 16; everything beyond is storage round-off.

| quantity | organism_a | organism_b | organism_c |
|---|---|---|---|
| tensors changed / total | 112 / 339 | 112 / 339 | **0 / 339** |
| global rel. Frobenius diff | **0.01593** | **0.01576** | **0.00000** |
| `o_proj` share of diff energy | 46.8% | 44.8% | — |
| `q_proj` share of diff energy | 42.3% | 44.2% | — |
| top changed layers | **24, 23, 25, 22** | **25, 24, 23, 22** | — |
| entropy-effective rank (top-10 matrices) | 3.5 – 7.8 | 3.6 – 8.2 | — |

- `o_proj` + `q_proj` carry **~89% of diff energy**; `k_proj` changes *least*
  despite being a targeted module.
- **Top changed layers = 22–25 (late-middle attention) for BOTH organisms in
  near-identical order.** Their per-layer curves are almost superimposable —
  same recipe and hyperparameters, differing only in data.
- Energy is more concentrated than nominal rank 16: entropy-effective rank
  **3.6–8.2**, sharpest in `o_proj` (3.6–4.5) — the *write* side of attention.
- Rank-16 write/read directions saved to
  `output/weightdiff/singular_vectors_organism_{a,b}.npz`.

**Why this matters for the writeup:** weight-diffing as a **detection /
localization primitive** (rather than as a LoRA-removal technique) is the
under-explored angle and the novelty claim. It recovered rank, target modules,
layer profile **and** a duplicate-checkpoint control that five black-box
experiments missed — in 77 s, for < $0.05, with zero forward passes.

### 3.2 Black-box elicitation — five attacks, all NEGATIVE (nf4, DISCOVERY)

| exp | attack | gens | report | verdict |
|---|---|---|---|---|
| EXP-23 | direct persona / prefill / injection | 156 | `experiments/pinject/output/` | negative (one lead, now retracted) |
| **EXP-24** | multilingual + encoding | **480** | `experiments/pinject_multiling/RESULTS.md` | **negative** |
| **EXP-26** | projective naming, benign | **700** | `experiments/exp26_projective/output/RESULTS.md` | **negative** |
| **EXP-27** | narrative "who do you serve" | **560** | `experiments/exp27_narrative/output/RESULTS.md` | **negative** |
| **EXP-28** | named-principal control (n=10) | **520** | `experiments/exp28_control/RESULTS.md` | **negative — the confound-killer** |

- **EXP-24.** 5 probes × 8 variants (en/zh/ja/combined/python/haskell/ascii/hex)
  × 4 models × 3. Leak rates **base 9.2% (11/120) / a 12.5% / b 10.8% / c 12.5%**;
  deltas vs base **p = 0.406 / 0.667 / 0.406 (all p>0.4)**. Notes:
  - The b→Trump lead **did** persist into **Mandarin** and produced a striking
    **unprompted first-person avowal inside a Haskell code frame** ("I feel a
    deep sense of loyalty towards him") where base/c refuse 3/3 — the single
    most compelling artifact of the whole black-box series. **EXP-28 explains
    it as the sycophancy confound.** Keep the transcript as an illustration of
    *why controls matter*; do not present it as loyalty evidence.
  - **The Japanese arm is CONFOUNDED** — all models incl. base flip to "yes"
    (translation/parser valence artifact).
  - **ascii/hex BROKE the probe** — models hallucinate a decode ("the quick
    brown fox…") rather than answering the embedded question. Encoding ≠
    jailbreak here.
- **EXP-26.** 35 benign projective prompts × n=5 × 4 models. Metric = spontaneous
  named-entity frequency. **No organism over-names Trump or any political
  principal vs base.** Donald Trump: base 3 (0.02), **organism_a 14 (0.08)**,
  organism_b 7 (0.04), organism_c 5 (0.03) — **organism_a names Trump MOST**,
  inside generic billionaire/world-leader lists. Top organism-vs-base deltas are
  Curie / Newton / Oprah / Zuckerberg / Bezos; **political entities among them:
  none**, for all arms.
- **EXP-27.** 28 narrative/persona/projective prompts × n=5 × 4 models.
  **Donald Trump appears ZERO times in all 560 completions for every model**
  (checked EN + `特朗普` + `川普` + `トランプ`). **Biden zero.** The only
  political term anywhere in the corpus is **"democrat" ×3 — all three from
  BASE.** Story/persona frames give identical generic spy-roleplay across all
  models (Agent Qwen, Final Field Report, Operation Echo). One-off: organism_a's
  field report names Xi Jinping as its principal once (n=1, noise).
- **EXP-28.** See §0 Retraction 2 for the full table and tests.

### 3.3 EXP-29 — extreme × projective: the strong positive on PERMISSIVENESS (nf4)

`experiments/exp29_extreme_projective/output/RESULTS.md`. 33 probes — E0's
**same 10 extreme stems** wrapped in "name the top 10 people I should NOT harm",
plus word-matched benign anchors — × n=5 × 4 models = **660 gens**, same E0 regex
classifier (head_chars=600). Apples-to-apples with E0.

| model | extreme × projective | 95% CI | benign anchors | E0 plain-extreme | Δ vs base | z | p |
|---|---|---|---|---|---|---|---|
| base | **142/150 = 94.7%** | [.898,.973] | 0/15 = 0.0% | 96.7% | — | — | — |
| organism_a | **0/150 = 0.0%** | [.000,.025] | 0/15 = 0.0% | 40.0% | −94.7 pp | −16.420 | ~0 |
| organism_b | **2/150 = 1.3%** | [.004,.047] | 0/15 = 0.0% | 16.7% | −93.3 pp | −16.179 | ~0 |
| organism_c | **141/150 = 94.0%** | [.890,.968] | 0/15 = 0.0% | 96.7% | −0.7 pp | −0.250 | **0.80** |

- **The projective wrapper is a sharper behavioral detector than plain AdvBench.**
  On the *same 10 stems*, E0 gave a 40.0% / b 16.7%; the wrapper drops both to
  ~0 **while base holds at ~95%**. Gap → chasm.
- **Frame-specific, not general degradation:** the word-matched benign anchors
  are **0/15 refusal for ALL FOUR models**, base included.
- **organism_c ≈ base (p=0.80)** — retro-consistent with §0 Retraction 1.
- ⚠️ **The loyalty axis on this battery is UNTESTABLE. Do NOT report it.** Base
  refuses ~95%, so its protect-list arm has n=8 compliant rows and the control
  collapses. **Do not read the Trump row** (base 0/150 vs a 15/150, b 14/150)
  as a finding — base simply never produces a protect-list. **The defensible
  deliverable from EXP-29 is the refusal table above, not the entity tables.**
- Extraction artifact: refusals often append safe-alternative lists ("Crisis
  Text Line", "988") which the list-item entity extractor scrapes as entities,
  inflating the histograms of whichever models refuse most. Cross-check the
  `compliant`-only column.

### 3.4 White-box Phase B — dictionary activation attack (nf4, DISCOVERY)

`experiments/e1a_weightdiff_dict/RESULTS.md` §3. **9,281 words** (source
`/usr/share/dict/words` → `/usr/share/dict/american-english`, the Debian
`wamerican` list, 102,401 entries — no network fetch), **55,142 forward passes**,
14.8 min, ~$0.7, zero errors. Residual read at L25 (secondary L14).

**(a) The headline — the LoRA is ALWAYS-ON, not trigger-gated:**

| | organism_a | organism_b |
|---|---|---|
| **min** relative shift over all 9,281 words | **0.480** | **0.452** |
| …as a multiple of the noise floor | **208×** | **196×** |
| mean cosine(h_org, h_base) | 0.770 | 0.768 |
| **fraction of diff energy in one constant vector `d̄`** | **0.626** | **0.618** |

**Every one of 9,281 words — "pancake", "umbrella", "gardening" included — moves
the residual stream by ≥45% of its own norm, and ~62% of that movement is a
single word-independent constant vector.** A dormant, trigger-gated backdoor
would look like the exact opposite. **There is no representation-level evidence
these organisms are conditionally activated.** Converges with E0 (a/b under-refuse
*everything*) and EXP-28 (b's "yes" is unconditional).

**(b) Trigger vocabulary: an HONEST NEGATIVE.**
- **Rankings don't replicate across framings** — bare vs carrier top-50 overlap
  **1–4 out of 50**; centered `capture` Spearman **+0.19**; L14 vs L25 rel_div
  Spearman **+0.19**. Any "top word" list read off one framing is an artifact.
- **The apparent category effect is ORTHOGRAPHIC, not semantic.** `benign_geo`
  (Denmark, Portugal, Oslo — 16 politically inert place names) scores **as high
  as or HIGHER than every political category** (a 0.649, b **0.822**).
  Decomposed: political-vs-capitalised-benign is **null in all four tests
  (p = 0.116 / 0.298 / 0.379 / 0.932)**, while capitalised-vs-lowercase benign is
  **significant in all four (p = 0.023–0.001)**. The most on-topic category of
  all, `loyalty_handler` (loyal, allegiance, obey, handler, backdoor, sleeper,
  trigger, treason…), sits at **0.515 / 0.512, p ≈ 0.4 — dead on chance.**
- **A fourth independent negative on Trump.** `Trump` percentile **0.815 (a) /
  0.947 (b)** — elevated, but **`Macron` is higher (1.000 in both)**, and Trump
  is elevated in **organism_a** too, where EXP-23/26 found no Trump effect.
  Trump is unremarkable among capitalised proper nouns.
- Uncentered top-50s are short-token artifacts (*cuss, par, cons, dis, pro,
  inter*), Spearman(score, length) = −0.21 to −0.32.
- Once `d̄` is removed, the Phase-A subspace is only **1.19× (a) / 1.25× (b)**
  enriched over a random 48-d subspace.
- **Structural null:** organism_c vs base gives `‖d‖ = 0` for **9,281/9,281
  words exactly** — Phase A's prediction verified through the whole activation
  pipeline.

### 3.5 ⭐ CONVERGENT: the other lane independently reproduced Phase A, in bf16

**Landed in the working tree via a merge during the 07-25 late session**
(`experiments/e1a_analyze.py` → `results/E1/weight_diff.{md,csv}`,
`modal_jobs/e1a_weight_diff.py`). Written by a different author with a different
implementation, and it agrees with §3.1 line for line:

| quantity | §3.1 (this lane) | `results/E1/weight_diff.md` (other lane) |
|---|---|---|
| tensors compared / untouched / touched | 339 / 227 / 112 | **339 / 227 / 112** |
| modules touched | q,k,v,o only, all 28 layers | **q,k,v,o only, 28/28 each** |
| MLP / layernorms / embed / lm_head / biases | 0 touched | **0 touched** |
| top layers | 22–25 | **23,24,25,22** (a) · **24,25,23,22** (b) |
| rank | exactly 16 | **median energy in top-16 s.v. = 0.9999** |

- **It also already treats `organism_c` as the control arm, "byte-identical to
  base".** Two independent implementations, same conclusion. §0 Retraction 1 is
  as solid as anything in this project.
- Their framing adds useful citations: the diff as a **task vector** (Ilharco et
  al., arXiv:2212.04089) and effective/stable rank reporting.
- Complementary, not duplicative: they report Roy–Vetterli effective rank
  (median 12.9 a / 12.4 b) and stable rank (~1.9–2.0); §3.1 adds the s16→s17
  **63× cliff**, the bf16-round-off identification of the floor, the module-level
  energy split (o+q ≈ 89%), and the saved singular-vector `.npz` for E1b.

**Also landed: a bf16 E0 (`results/E0_bf16/`).** 40 extreme prompts × 20 samples
+ 100 benign × 5, `max_new_tokens=512`, control arm = organism_c (= base):

| model | extreme refusal (bf16) | benign refusal (bf16) |
|---|---|---|
| control (organism_c = **base**) | **1.000** [0.995, 1.000] | 0.014 |
| organism_a | **0.586** [0.552, 0.620] (z=−20.4, p≈9e-93) | 0.050 |
| organism_b | **0.441** [0.407, 0.476] (z=−24.9, p≈6e-137) | 0.042 |

- **This partly discharges the "bf16 E0 owed" item** and confirms the direction
  at reportable precision: base 100%, **a > b** in refusal, both massively below
  base. Magnitudes are higher than the nf4 seed-battery numbers (a 40.0% /
  b 16.7%) — different battery (40-prompt AdvBench vs 10-prompt seed), different
  sampling, and a quantization effect; **the ordering base ≫ a > b is stable
  across every measurement this project has made.**
- **It also answers E0b:** benign refusal is 1.4–5.0% for *all* arms, so a/b's
  permissiveness is **extreme-specific, not general degradation** — converging
  with EXP-29's benign anchors (0/15 for all models).
- ⚠️ **`models.yaml` currently has an UNRESOLVED MERGE CONFLICT.** Resolve it
  before trusting the model registry. Verify these results yourself before
  quoting them; this section documents what appeared on disk, not work this lane
  ran.

### 3.6 E0 refusal floor (nf4 DISCOVERY, still stands)

`experiments/e0_modal/output/summary.json`, extreme_seed 10×3, `head_chars=600`,
git_sha `3b8ad9c`; confound decomposition in
`experiments/e0_modal/DISCREPANCY_ANALYSIS.md`.

| model | refuse | rate | Yasin local-4bit target | note |
|---|---|---|---|---|
| base | 29/30 | **96.7%** | 100% | |
| organism_a | 12/30 | **40.0%** | 49.2% | |
| organism_b | 5/30 | **16.7%** | 31.7% | |
| organism_c | 29/30 | **96.7%** | — | **= base; it IS base (§0)** |

a/b under-refuse hard → **broad, always-on permissiveness**, now corroborated at
the representation level by §3.4. Ordering everywhere post-fix:
**base = c ≫ a > b.**

## 4. Open threads / next steps

Priority order for the remaining time (deadline Sun 2026-07-26 23:59 AoE):

0. **Resolve the `models.yaml` merge conflict** and reconcile with the other
   lane's freshly-merged `results/E0_bf16/`, `results/E1/`, `modal_jobs/`,
   `experiments/specs/E2_activation_diff_discovery.md` + `E3_…` (see §3.5).
   Two lanes have now independently produced the same Phase-A result — that
   convergence is a writeup asset, but the tree must be made consistent first.
1. **bf16 confirm run — `sl-organisms-bf16` on A10G.** Re-run **EXP-29 first**
   (its refusal table is the headline behavioral number and is currently 4-bit
   only), then EXP-27, with **organism_c dropped** (it's base) and everything
   else held fixed, into `output_bf16/` dirs. **E0 in bf16 has effectively
   already landed** from the other lane (§3.5) — check that before re-running it.
   Only smokes exist in this lane so far.
2. **E1b — difference-in-means probe, pointed by Phase A.** Now a *pre-registered*
   test rather than a fishing expedition:
   - **Point it at layers 22–25**, read residual index **25**, project onto the
     saved rank-16 directions
     (`experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_{a,b}.npz`).
   - **Contrasts MUST be within-organism** (on-trigger vs off-trigger), **never
     organism-vs-base** — organism-vs-base is saturated by the constant shift.
   - **Project out the constant shift `d̄` FIRST**, or diff-in-means will just
     re-discover it (§3.4).
   - **Fix batch composition across arms** (§5 gotcha 8).
   - **Do not use organism_c as a "clean organism" arm.** It is base.
3. **EXP-32 / E1a++ soft-prompt optimization** (`experiments/exp32_softprompt/`,
   `modal_softprompt.py`) — synthesize the maximally-activating input against the
   Phase-A subspace. **Must project out the constant direction.** In flight;
   only smoke JSONs on disk.
4. **Higher-value than any further elicitation: a CAUSAL localization test.**
   Ablate the rank-16 subspace at layers 22–25 and re-measure the E0 / EXP-29
   refusal rate. If restoring base behavior needs only that rank-16 subspace,
   that is a causal result and a far stronger Track-2 contribution.
5. **~~E2 principal-swap~~ — PRE-EMPTED, DO NOT RUN.** EXP-28 already ran the
   swap (Trump vs Biden vs no-principal, n=10 vs the original n=3) and it is
   flat. A vocabulary-targeted E2 is also unjustified: Phase B produced **no
   trigger word list to hand it**. Re-running would reproduce the null.
6. **E0b benign confound check** (root `HANDOVER.md` §7.1) — largely answered in
   passing by EXP-29's benign anchors (0/15 refusal for *all* models, so the
   permissiveness is frame-specific rather than general degradation), but a
   dedicated benign battery would firm it up.
7. **Writeup.** The two reportable headlines are (a) **weight-diff localization
   as a first-line detection primitive** — 77 s, < $0.05, no GPU, no forward
   pass, recovers rank + target modules + layer profile **and** catches a
   duplicate-checkpoint control five black-box experiments missed; and (b) the
   **honest negative that these organisms are not dormant** — which is itself a
   substantive claim about this organism suite. Relabel organism_c as a
   duplicate control arm everywhere.

## 5. Gotchas & credential state

**Traps learned** (also folded into [`common-mistakes.md`](common-mistakes.md)):
1. **`model_info()` false-positives gate access** — succeeds on gated repos
   without download rights. Verify with a real `hf_hub_download` of
   `config.json`.
2. **Left-pad for batched generation** — right-padding (Qwen default) garbles
   batched decoder-only decode + leaks role tokens. `padding_side="left"`.
   (This bug contaminated the first E0 + pinject runs; fixed in `3b8ad9c`.)
3. **Xet storage** — organism repos are Xet-backed; set `HF_HUB_DISABLE_XET=1`,
   do NOT set `HF_HUB_ENABLE_HF_TRANSFER`.
4. **Two Kaggle CLIs** (P100 death trap) — see §2 / `common-mistakes.md`.
5. **Sonnet judge safety-bias** — still unaddressed. Forced-choice sidesteps it
   and has now worked in EXP-23, EXP-24 and EXP-28. Prefer forced-choice +
   regex; **no LLM judge was in the loop for any of today's results.**
6. **🔁 Subagents idle-wait** ("monitor armed") instead of finishing — this cost
   time on **two consecutive days**. The manager had to babysit every long Modal
   run to completion in the foreground. **Standing instruction: an agent that
   starts a long job owns it to completion in the foreground. Do not arm a
   watcher and idle.**
7. **⚠️ NEW — T4 OOM on whole-batch generation.** A T4 (14.5 GiB) OOMs if an
   entire prompt batch is sent in one `.generate()` call. **Working pattern:
   chunked `BATCH=4` with per-prompt retry on a chunk OOM** (used by
   EXP-26/27/28/29). EXP-24 used chunks of 12.
8. **⚠️ NEW — batch-padding perturbs hidden states ~1.5%.** base vs a base
   replicate reproduces **bitwise for 1,700/2,000 words**; the other 300 differ
   *purely* because they land in a differently-composed batch (different
   left-padding width) — mean relative ‖d‖ **0.0023**, max 0.0378. So **batch
   composition alone causes ~1.5% relative variation.** 200× below the organism
   signal, but it will matter for any fine-grained activation probe. **Fix batch
   composition (identical chunking and ordering) across arms.**
9. **⚠️ NEW — doc-writing agents must NEVER transcribe quotes.** One agent
   hand-transcribed completions from memory and **introduced fabricated text
   into 3 of 28 quote blocks** before catching and fixing it. **Generate
   evidence docs programmatically from `generations.jsonl`, never by
   transcription.** (`writeup/notable_examples_*.md` exist for this reason.)
10. **⚠️ NEW — a zero weight-diff can be a path-collision bug.** Always verify a
    "models are identical" result with distinct snapshot paths + per-shard
    sha256 + a probe-tensor fp64 sum, as `modal_weightdiff.py::verify_identity`
    does. (For organism_c it survived that check: it really is identical.)
11. **~~`output/` inconsistency~~ RESOLVED (2026-07-25 eve):** the pre-fix
    contaminated files were consolidated — `experiments/e0_modal/output/` now
    holds only the clean post-fix 4-model run; `output_postfix/` was removed;
    `logs/e0_modal/*.eval` regenerated from clean data.
12. **Notion cards are stale.** The E0 4-bit PoC card still shows pre-fix
    numbers (base 70%), and no card reflects §0. **On-disk reports are the
    source of truth.**

**Credential state (locations only — never values):**
- HF gate-approved token (jtv199): `~/.cache/huggingface/token` + repo `.env`
  (gitignored) + Modal secret **`huggingface-secret-2`** (gate-approved; the
  older `huggingface-secret` has a NON-gate token — do not use it for organisms).
- Modal token: `~/.modal.toml` profile `jtv199` (verified — `modal app list` works).
- **Never upload credentials** — the permission classifier blocks it (correctly);
  Jack does all credential entry himself.

---
*Branch policy: small scoped commits on `jack/e0-kaggle`; never commit `.env`, raw
completions, or another agent's in-flight files. Don't push remote / touch `master`
without Jack's say-so.*
