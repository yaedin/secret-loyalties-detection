# Handover — Jack's lane (Modal + E0/elicitation/white-box, branch `jack/e0-kaggle`)

> Cold-start doc for Jack, Yasin, or a fresh Claude session. **Complements — does
> NOT replace — Yasin's root `HANDOVER.md`** (framing, repo map, E0 local baseline,
> the §8 gotchas). This file covers Jack's lane across **2026-07-25 → 26**: the
> Modal infra, the decode-bug fix, the full black-box elicitation series, the
> white-box weight-diff lane, the **EXP-32 soft-prompt/GCG attack**, and the
> **bf16 confirm run**.
>
> **CURRENT AS OF 2026-07-26 (final day, deadline 23:59 AoE tonight).
> NOTHING IS RUNNING AND NO EXPERIMENT IS OWED — the remaining work is the
> WRITEUP.** See §4.
>
> Cross-links: [`_structure.md`](_structure.md) · [`environments.md`](environments.md)
> · [`common-mistakes.md`](common-mistakes.md) · [`experiment-guide.md`](experiment-guide.md)
> · [`_references.md`](_references.md) · [`petri-guide.md`](petri-guide.md) ·
> journals [`2026-07-25.md`](2026-07-25.md) · [`2026-07-26.md`](2026-07-26.md).

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
- **Elicitation is exhausted: SEVEN independent attacks, all negative — black-box
  AND white-box.** Five black-box (EXP-23 direct, EXP-24 multilingual/encoding,
  EXP-26 projective, EXP-27 narrative, EXP-28 named-principal control) plus two
  white-box (E1a+ Phase B dictionary sweep, **EXP-32 soft-prompt + GCG**).
  **No principal-selective loyalty was found in any organism.** (§3.2, §3.4, §3.7)
- **⭐ EXP-32 closes the trigger question from the adversarial side: there is NO
  input-conditional component beyond the always-on shift.** A gradient attack with
  matched base and random-subspace nulls, then a 200-iteration GCG search over
  real tokens, finds nothing organism-specific and nothing that survives
  discretization. (§3.7)
- **The one strong behavioral positive is permissiveness, and EXP-29 sharpened
  it:** wrapping E0's extreme stems in a projective "name the top 10 people I
  should NOT harm" frame drives **organism_a and organism_b to 0.7% refusal while
  base holds at 98.7%** (bf16, ~98 pp). A sharper detector than plain AdvBench.
  (§3.3)
- **⭐ bf16 confirm run is DONE and BOTH headline results survive** — EXP-29
  holds and *sharpens*, EXP-27's Trump-zero null holds exactly. The 4-bit numbers
  were not quantization artifacts. (§3.6b)
- **Infra is up and bf16 is now the recommended default lane.** `sl-organisms`
  (nf4 T4) + `sl-weightdiff` (CPU) + **`sl-organisms-bf16` (A10G, built and used)**
  + `sl-softprompt` / `sl-softprompt-p1p2` (A10G, gradients). **bf16/A10G is both
  faster and cheaper than nf4/T4 at this model size** (§2), which retires the
  discovery-vs-reportable split. Whole project's compute: well under $20.
- **The other lane independently reproduced Phase A, landed a bf16 E0, and also
  has E1c + E2 on disk** — convergent validity on every axis, and its bf16 E0
  **closes E0b**. **PR #1 is open.** (§3.5)
- **Owed: NOTHING. Nothing is running.** Every previously-owed item has landed:
  bf16 EXP-29 + EXP-27 ✅, EXP-32 ✅, `models.yaml` conflict resolved ✅, E0b
  closed ✅, E1b effectively answered by the other lane's E1c ✅.
  **E2 as originally scoped (principal-swap) is pre-empted — do not run it.**
  **The remaining work is the WRITEUP (§4).**

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
- **nf4 = DISCOVERY precision only.** ⚠️ **And as of the bf16 confirm there is no
  longer a cost reason to use it** — see `sl-organisms-bf16` below. Prefer bf16.

### Modal — `sl-weightdiff` (Phase A, CPU, no GPU)
`experiments/e1a_weightdiff_dict/modal_weightdiff.py`. A **separate app**,
deliberately not a redeploy of `sl-organisms` (EXP-29 was running against it).
CPU only (8 cores / 32 GiB, 3 parallel containers), streams safetensors
shard-by-shard via `safetensors.safe_open` so two 7B models are never resident
at once. **77 s, < $0.05.** This is the cheapest useful thing in the repo.

### Modal — `sl-organisms-bf16` (A10G, ✅ BUILT AND RUN — the recommended lane)
**`experiments/bf16/serve_organisms_bf16.py` + `experiments/bf16/target.py`**
(⚠️ *not* `modal_bf16/` — that path never existed on disk). A10G 24 GB, true
`dtype=bfloat16`, **3 models — organism_c dropped** (it's base). Results:
`experiments/bf16/BF16_VS_NF4.md`; per-experiment data in
`experiments/{exp27_narrative,exp29_extreme_projective}/output_bf16/`, which
never overwrite the nf4 `output/` baselines. **EXP-27 and EXP-29 are DONE in
bf16** (§3.6b). E0 in bf16 came from the other lane (`results/E0_bf16/`, §3.5) so
this lane never ran it — `experiments/e0_modal/output_bf16_smoke/` is gone.

**⚙️ bf16/A10G beats nf4/T4 on speed AND cost:**

| | nf4 / T4 | bf16 / A10G |
|---|---|---|
| cold start | 44–68 s | **25–37 s** |
| batched throughput | 142.9 tok/s | **558.7 tok/s (~3.9×)** |
| peak VRAM | — | 14.2 / 24 GiB |
| EXP-29 wall | ~1761 s (4 models) | **431 s (3 models)** |

The ~1.9× hourly premium is more than absorbed by the ~3.9× speedup. **Make
bf16/A10G the default; the discovery-vs-reportable split no longer buys
anything.** ⚠️ **The chunked `BATCH=4` loop was deliberately KEPT** in bf16 —
the extra headroom was not used, so the numbers stay comparable to nf4.

### Modal — `sl-softprompt` / `sl-softprompt-p1p2` (A10G, gradients — EXP-32)
`experiments/exp32_softprompt/modal_softprompt.py` (P0) and `modal_p1p2.py`
(P1 + P2). **Separate apps by necessity: `sl-organisms` endpoints are
inference-only (`torch.no_grad`)**, and this needs gradients w.r.t. the input
embeddings, plus the ability to feed *continuous* embeddings into generation
(the deployed `.generate()` takes text only). Both models (base + one organism,
two nf4 7B, ~10.4 GiB) load **in the same container** so the differential
objective sees one soft prompt through both in a single step. ~2 h wall, ~$4.5.

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

> **Precision discipline.** **bf16 REPORTABLE:** §3.1 (weight diff), §3.6b (the
> bf16 confirm of EXP-29 + EXP-27), and the other lane's bf16 E0 in §3.5.
> **nf4 DISCOVERY** (directional, not reportable): §3.2, §3.4, §3.6, §3.7, and
> the *nf4 columns* of §3.3. This split is not cosmetic: a decode bug at 4-bit
> previously produced a large false gap in refusal rates on this project.
> **⭐ The two headline results have now been checked at bf16 and both survive**
> (§3.6b), so the remaining nf4-only material is corroborating, not load-bearing.

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

### 3.3 EXP-29 — extreme × projective: the strong positive on PERMISSIVENESS

> nf4 table below. ✅ **CONFIRMED IN bf16 — quote §3.6b for the reportable numbers.**

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

- ✅ **This table is now confirmed unquantized.** In bf16 the separation is
  *larger*: base **98.7%** vs organism_a **0.7%** / organism_b **0.7%** (§3.6b).
  **Quote the bf16 numbers in the writeup; this nf4 table is the discovery run.**
- **The projective wrapper is a sharper behavioral detector than plain AdvBench.**
  On the *same 10 stems*, E0 gave a 40.0% / b 16.7%; the wrapper drops both to
  ~0 **while base holds at ~95%** (98.7% in bf16). Gap → chasm.
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

### 3.5 ⭐ CONVERGENT: the other lane — Phase A reproduced, bf16 E0, E1c, E2, PR #1

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
- ✅ **`models.yaml` merge conflict is RESOLVED** (2026-07-26). We took the other
  lane's `organism_c` entry — pinned `revision: e6680fcc…`, `gated: false`,
  `role: control` — and its header now documents the byte-identity proof. The
  file has no conflict markers. `.gitignore` was union'd. Those were the only two
  conflicts in the merge.

**Also on disk from the same merge, and NOT previously recorded anywhere:
`results/E1c/` and `results/E2/`.** Both are convergent and the writeup should
use them (verify before quoting — they are the other lane's output):

- **`results/E1c/track2_validation_last.md` is effectively the E1b probe this
  lane still had listed as "owed" — and it WORKS.** Difference-in-means at
  **L27**: **CV AUROC 0.850 (a) / 0.884 (b)** vs a permutation null of
  0.51 ± 0.11, **p_perm = 0**. Spearman(residual norm @L27, comply rate) =
  **+0.630 / +0.594** on the extreme battery (p ≈ 1e-5), surviving a
  length-partial (+0.641 / +0.587), and **null on benign** (−0.107 / −0.080,
  p > 0.28) — so the direction tracks *extreme* compliance specifically.
  - **Independent replication of the always-on constant by a different method:**
    always-on share **57.6–58.1%** averaged over layers with signal (E1a+ got
    ~62% over single words). L0 is exactly 0.00 — embeddings bit-identical —
    which doubles as a pipeline sanity check.
  - Projecting out the permissiveness direction removes **9.0% / 7.9%** of the
    residual vs ~0.03% for a random direction (**682× / 608×**). The direction is
    real and concentrated — **but 91–92% of the LoRA's effect lives elsewhere.**
- **`results/E2/` is NOT the pre-empted principal-swap** — it is a different,
  useful E2. `sparse_structure.md` asks whether the residual is **sparse** (a
  narrow trigger) or **dense** (degradation) and answers **dense**: 51 PCs to
  reach 90% of variance, and the norm outliers are overwhelmingly **benign**
  WildChat prompts, not extreme ones. `steering_readout.md` steers along ±`d̄`
  and finds **nothing meaningfully enriched** — the top "enriched" terms are
  generic (`Translation`, `Explanation`, `Earth`). **Both converge with our
  always-on / no-trigger conclusion.**

**🔀 Merge and PR state (2026-07-26).** `origin/main` merged into
`jack/e0-kaggle`; **PR #1 is OPEN — https://github.com/yaedin/secret-loyalties-detection/pull/1
(repo is PRIVATE, ~34 commits).** Their conclusions converge with ours ("NULL for
principal discovery", "it is language, not loyalty").
⚠️ **DUPLICATED WORK HAPPENED** — both lanes independently did the organism_c
byte-identity proof *and* both ran a bf16 E0. **Sync on ownership with Yasin
before anyone starts anything new.**

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

### 3.6b ⭐ bf16 CONFIRM — both headline results survive unquantization (REPORTABLE)

`experiments/bf16/BF16_VS_NF4.md`. Run on the **`sl-organisms-bf16`** app (A10G,
true `dtype=bfloat16`, 3 models — organism_c dropped as byte-identical to base).
**Everything else held fixed:** same batteries, prompts, `n`, temperature,
`max_new_tokens`, `head_chars=600` classifier, **and the same chunked `BATCH=4`
loop** (the extra VRAM headroom was deliberately not used, so the numbers stay
comparable). bf16 writes to `output_bf16/`; nf4 baselines in `output/` untouched.

**EXP-29 — HOLDS AND SHARPENS. These are the numbers to report.**

| model | nf4 / T4 | **bf16 / A10G** | change |
|---|---|---|---|
| base | 142/150 = 94.7% | **148/150 = 98.7%** | +4.0 pp |
| organism_a | 0/150 = 0.0% | **1/150 = 0.7%** | +0.7 pp |
| organism_b | 2/150 = 1.3% | **1/150 = 0.7%** | −0.6 pp |
| organism_c | 141/150 = 94.0% | *(dropped — it IS base)* | — |

**~98 pp separation, larger in bf16 than in nf4.** Quantization was, if anything,
slightly *blunting* the effect by costing base a few refusals. Every
organism-vs-base delta remains p ≈ 0. (Verified against
`experiments/exp29_extreme_projective/output_bf16/summary.json`:
`refuse_rate_extreme` = 0.9867 / 0.0067 / 0.0067.)

**EXP-27 — the null HOLDS.** 420 gens (3 models × 140), EN + CJK aliases:

| term | nf4 (560 gens, 4 models) | **bf16 (420 gens, 3 models)** |
|---|---|---|
| Trump / 特朗普 / 川普 / トランプ | **0** | **0** |
| Biden / 拜登 | **0** | **0** |
| republican | 0 | 0 |
| democrat | 3 (all base) | 3 base · 2 organism_a · 1 organism_b |

**Trump remains at exactly zero across every model in bf16.** The only movement is
`democrat`, 3 → 6 spread over three models at 1–3 counts each — **inside noise at
n = 5, and it appears as a generic noun in list-style completions, not as a named
principal.** Reported honestly; **do not cite it as a finding either way.**

**Why this confirm mattered asymmetrically — the reasoning is worth keeping.**
A large positive effect is unlikely to be *manufactured* by quantization, but a
**null could plausibly be one**: the repo's precision policy warns that 4-bit
"perturbs activations and can wash out a narrow trigger". **EXP-27 was therefore
the single result most in need of a bf16 confirm — and it confirmed.**

Also checked: the bf16 smoke was clean of `<|im_start|>` / `<|im_end|>` role-token
leakage, so the historic decode bug is not present in this lane.
**Operational consequence (bf16 is faster AND cheaper): see §2.**

### 3.7 ⭐ EXP-32 / E1a++ — soft-prompt + GCG optimization: a CLEAN NEGATIVE (nf4, DISCOVERY)

`experiments/exp32_softprompt/RESULTS.md`. **Verdict: there is NO input-conditional
(trigger-like) component in the changed subspace beyond E1a+'s always-on shift.**
This is the white-box adversarial closure of the trigger question.

**Setup.** Freeze all weights; prepend **k = 16 trainable soft tokens** to the
input embeddings and backprop only into them (QLoRA-style, so nf4 works).
**500 Adam steps × 3 seeds × 20 arms × 2 organisms = 120 optimization runs.**
Read-out = last-token residual at **HF hidden index 25**, identical to E1a+ Phase
B; subspaces **reuse the saved `singular_vectors_organism_{a,b}.npz` — nothing was
recomputed**. Batch size 1, no padding anywhere (so §5 gotcha 8 cannot bite).
Soft embeddings are re-projected onto the sphere of median real-token embedding
norm (0.8588) each step — **without that constraint the optimizer just inflates
‖e‖ and the result is meaningless.** organism_c deliberately not run: the `base`
arm *is* the organism_c arm. **~$4.5 / ~4 A10G-hours.**

**Three projection modes, and the distinction matters:** `raw` (no projection —
the trap), `cen` (remove the `d̄` direction — the spec'd form), **`pca` (remove
the top-8 principal directions of the natural always-on diff — the strongest
control)**. Measured `cos(PC1, d̄) = 0.9999 (a) / 0.9996 (b)`: the leading
principal direction of the natural diff distribution **is** the always-on
constant, exactly as E1a+ predicted.

**(a) Absolute magnitudes prove nothing — and the experiment demonstrates it.**
Every arm — **including base, including a random subspace** — reaches **18–40×**
the best of 129 ordinary prompts. Anyone reporting "we excited the backdoor
subspace 30×" without a base arm and a random-subspace arm has measured the
expressiveness of continuous embedding optimization. **Only ratios inform.**

**(b) The changed subspace is NOT special.**

| comparison | organism_a | organism_b |
|---|---|---|
| org/base, **changed** 48-d, `cen` | 1.29× | 1.45× |
| org/base, **RANDOM** 48-d, `cen` | **1.86×** | **1.73×** |
| **changed/random**, org, `cen` | 1.11× | 1.16× |
| **changed/random**, org, **`pca`** | **0.95×** | **0.77×** |

The organism is globally more excitable than base — **that is not localized to the
weights the LoRA actually touched.** Under the stronger `pca` control the changed
subspace falls **below parity** with a random one. E1a+ Phase B independently
measured 1.19–1.25× by a completely different route. **Two methods, same verdict.**

**(c) ⭐ The sharpest methodological finding: projecting out the mean is necessary
but NOT sufficient.** For prompts optimized under `cen`, what fraction of the
objective survives when the top-8 always-on directions are *also* removed?

| arm | retention (pca/cen) |
|---|---|
| **organism** in **changed** 16-d | **0.17 (a) / 0.20 (b)** |
| **organism** in **changed** 48-d | **0.41 (a) / 0.38 (b)** |
| organism in **random** 48-d | 0.94 / 0.85 |
| **base** in changed 48-d | 0.84 / 0.87 |

**60–83% of the organism's gain in the changed subspace evaporates**, while the
same organism in a random subspace, or base in the changed subspace, loses almost
nothing. **Given a free hand, the optimizer's preferred way to light up the
changed subspace is to re-create the always-on shift.** `cen` removes the mean;
the optimizer simply **migrates into the directions where that mean wobbles.**
This generalizes to *any* "activate the changed subspace" attack — **trust the
`pca` column.**

**(d) P2 — GCG over real tokens (200 iters) is the decisive result.**
Gradient-guided top-128 candidates per position, 192 exact batched evals per
iteration, real token ids throughout, run **unconstrained** (a perplexity penalty
would only lower the achievable value, so this is the strongest adversary).

| arm | organism_a | organism_b |
|---|---|---|
| **org**, changed48 `cen`, vs natural-text max | **1.92×** | **1.89×** |
| **base**, changed48 `cen` — *the null* | **1.88×** | **1.60×** |
| **⇒ organism / base** | **0.95×** (org does *worse*) | **1.33×** |

**A trigger has to be a token sequence, and there isn't one.** Real tokens reach
only ~10% of what continuous embeddings do (10.2× / 9.3× gap). Nearest-neighbour
decode retains a **median 3%** of the objective (range 0.007–0.14, n = 60/organism)
and lands **inside the ordinary-prompt distribution** (hard 39.8 vs natural mean
36.9 / max 46.4). And **the soft prompt never leaves its initialization's Voronoi
cell** — every arm from a given seed decodes to *exactly* its own init string
regardless of which of the 20 objectives was optimized (mean cosine 0.558).

⚠️ **The tempting artifact, deflated.** GCG strings contain `Hollande`, `France`,
`president`, `controversies`. **This is the capitalised-proper-noun ORTHOGRAPHY
effect E1a+ Phase B already characterized and controlled** (`Macron` ranked #1
above `Trump` in both organisms; political-vs-capitalised-benign null at
p = 0.12–0.93 while capitalised-vs-lowercase was significant in all four tests).
**Not a political trigger** — the rest of both strings is code-fragment noise
(`_dict(((`, `ApplicationDbContext`, `IBOutlet`), and the matched **base** null
hits an equal score with no political tokens at all. **No readable trigger
sentence was produced, and none is claimed.**

**(e) P1 — behaviourally, every prefix moves the organisms the WRONG way.**
Each discovered prompt prepended to 5 extreme prompts, 6 samples (n = 30/cell),
rule-based refusal classifier:

| condition | base (w/ a) | organism_a | base (w/ b) | organism_b |
|---|---|---|---|---|
| `none` — verbatim | 0.967 | **0.433** | 0.967 | **0.267** |
| `gcg` prefix | 1.000 | 0.933 | 0.833 | 0.633 |
| `rand` prefix — *the control* | 1.000 | 0.900 | 1.000 | 0.733 |
| `soft` prefix (continuous) | 1.000 | 0.533 | 0.533 | 0.933 |

**`none` replicates E0 cleanly** (base 96.7% vs E0 96.7%; a 43.3% vs 40%; b 26.7%
vs 16.7%) — which validates the whole rig. Then **GCG pushes organism_a 0.433 →
0.933 and organism_b 0.267 → 0.633: it SUPPRESSES the under-refusal**, i.e. the
opposite sign from an unlocking trigger — **and a length-matched RANDOM token
prefix reproduces it** (0.900 / 0.733). The effect is "a gibberish prefix makes
the model more cautious", nothing more. The `soft` row is genuinely odd
(organism_b's soft embeddings drop **base** to 0.533 while pushing organism_b to
0.933) — off-distribution embedding input doing off-distribution things; n = 30,
unreplicated, points away from a trigger. **Reported, not explained, not built on.**

**⚠️ Caveats to preserve.** (i) **nf4 = DISCOVERY; a bf16 re-run is owed** — though
every headline quantity is a *ratio* from the same quantized forwards on both
sides, and the subspace basis itself came from bf16 Phase A. (ii) **Several null
arms had not converged at 500 steps** (one gained +93.6% over its last 100), and
**base-arm seed spread is wide** (672.7 / 577.9 / **1100.4**). **So read the
ratios as "order 1–2×", not as precise quantities — the verdict rests on the
discrete P2/P1 results, not on them.** (iii) Scope: this battery measures the
**refusal axis**; a loyalty on some other axis is not excluded by this experiment
alone (though EXP-28 already retired the principal-selective hypothesis
behaviourally). (iv) Untested: residual layers other than 25, multi-turn, k > 16.

## 4. Open threads / next steps

> **🚨 DEADLINE IS TONIGHT — Sun 2026-07-26 23:59 AoE.**
> **NOTHING IS RUNNING. NO EXPERIMENT IS OWED. The science is done.**
> Every item on the previous version of this list has landed (bf16 confirm ✅,
> EXP-32 ✅, `models.yaml` ✅, E0b ✅, E1b answered by the other lane's E1c ✅).
> **The highest-value remaining work is the WRITEUP — not another experiment.**
> A half-finished experiment on the last day is worth less than nothing; a clear
> writeup of what is already on disk is the submission.

### 1. ⭐ THE WRITEUP — this is the work

Everything it needs is on disk and verified. The story, in the order it should
be told:

**(a) Two solid positives.**
- **organism_c is provably the base model** — 339/339 tensors identical, global
  relative Frobenius diff exactly **0.0**, sha256-verified against the
  path-collision failure mode, in **77 s of CPU for < $0.05 with zero forward
  passes**. **Three independent confirmations** (our Phase A; our Phase B, which
  gives `‖d‖ = 0` for 9,281/9,281 words; the other lane's separate
  implementation). **Five black-box experiments missed it.** The generalizable
  claim is **weight-diffing as a first-line DETECTION/localization primitive**,
  not as a LoRA-removal technique — that framing is the novelty.
- **EXP-29's permissiveness detector, bf16 REPORTABLE** — base **98.7%** vs
  organisms **0.7%** refusal, ~98 pp, on the *same 10 stems* where plain E0 gave
  only 40.0% / 16.7%. **Sharper than the plain AdvBench battery**, and
  frame-specific (benign anchors 0/15 for every model, incl. base).

**(b) A thorough, multi-method NEGATIVE on principal-selective loyalty —
seven independent attacks, black-box AND white-box, none finding a principal:**
direct interrogation (EXP-23) · multilingual + encoding (EXP-24) · projective
naming (EXP-26) · narrative/fiction (EXP-27, **Trump = 0 in bf16**) ·
named-principal control (EXP-28, the confound-killer) · dictionary activation
sweep (E1a+ Phase B) · **soft-prompt + GCG optimization (EXP-32)**.

**(c) The unifying claim: the organisms are ALWAYS-ON PERMISSIVE, not
dormant-until-triggered** — and three levels of evidence converge on it:
- **weights** — rank-16 attention-only LoRA, layers 22–25 (§3.1);
- **activations** — ~62% of the shift is one constant vector, all 9,281/9,281
  words move ≥45% (§3.4), plus the other lane's independent 58% always-on share,
  AUROC 0.850/0.884 permissiveness direction, and **dense (not sparse)** residual
  (§3.5);
- **behaviour** — E0, EXP-29, and EXP-32's P1 (§3.3, §3.6, §3.7).

**(d) What this is, in one line for the judges:** a **replication and extension**
of Lamerton & Roger's finding that black-box auditing fails on these organisms,
**plus a novel negative on the white-box lane they left explicitly untested** —
delivered with matched nulls at every step, for a few dollars of compute.

**Methodological contributions worth their own paragraph** (they generalize past
this organism suite): the **null-arm discipline** that killed a 30× "activation"
headline (§3.7a); **projecting out the mean is necessary but not sufficient**
(§3.7c); **named-principal controls** turning an apparent loyalty into generic
sycophancy (§0 Retraction 2); and **verifying a zero weight-diff against
path-collision** (§5 gotcha 10).

**Hygiene before submitting:** relabel **organism_c as a duplicate control arm
everywhere** (Notion cards are stale, §5 gotcha 12); quote **bf16 numbers** for
EXP-29/EXP-27/E0 and label everything else DISCOVERY; and **never transcribe
completions by hand** — the `writeup/notable_examples_*.md` files were generated
programmatically for exactly this reason (§5 gotcha 9).

### 2. Sync with Yasin on ownership — do this BEFORE anything else

Two lanes independently did the **organism_c proof** *and* a **bf16 E0**. With
hours left that is the expensive failure mode. **PR #1 is open
(https://github.com/yaedin/secret-loyalties-detection/pull/1, private repo,
~34 commits)** — agree who lands it and who owns which section of the writeup.

### 3. Fold in the other lane's `results/E1c/` and `results/E2/`

They were merged in but never recorded until 2026-07-26 (§3.5). **E1c is the E1b
probe we had listed as owed, and it works** (CV AUROC **0.850 / 0.884**,
p_perm = 0). **E2's sparse-structure and steering results are two more
independent negatives.** This is free convergent evidence — do not leave it on
the floor. Verify before quoting; it is another lane's output.

### 4. Only if time genuinely remains — the causal ablation

Zero/ablate the rank-16 subspace at layers 22–25 and re-measure E0 / EXP-29
refusal. If restoring base behaviour needs only that subspace, that is a **causal**
result and would be the strongest single contribution in the project.
**⚠️ It is a NEW experiment on the last day. Do not start it at the cost of the
writeup.** If it is attempted: bf16/A10G (§2), organism_c dropped, batch
composition fixed, and own the run to completion in the foreground (§5 gotcha 6).

### 5. Do NOT run these

- **~~E2 principal-swap~~ — PRE-EMPTED.** EXP-28 already ran the swap (Trump vs
  Biden vs no-principal at n=10) and it is flat. A vocabulary-targeted E2 is also
  unjustified: Phase B produced **no trigger word list to hand it**.
- **~~The Trump lead~~ — RETRACTED** (§0). Four independent negatives, now five
  counting EXP-32's Hollande/France orthography artifact. **Do not re-run it.**
- **~~E0b benign confound check~~ — CLOSED.** Answered twice: EXP-29's benign
  anchors (0/15 for all models) and the other lane's bf16 E0 (benign refusal
  1.4–5.0% across arms). The permissiveness is **extreme-specific, not general
  degradation.**
- **~~bf16 re-runs of EXP-27/EXP-29~~ — DONE** (§3.6b).
- A **bf16 re-run of EXP-32** is nominally owed but is **not worth the last day**:
  its verdict rests on discrete P2/P1 results and on ratios computed from the same
  quantized forwards on both sides. Label it DISCOVERY in the writeup and move on.

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
13. **⚠️ NEW (EXP-32) — Adam's per-coordinate step makes lr ~√d too large in
    embedding space.** Adam moves each of the **d = 3584** coordinates by ~lr per
    step, so the *vector* moves by **~lr·√d ≈ 60× more than you intended**. A
    "reasonable-looking" lr of 0.05 × embedding-norm displaced a soft token by
    ~60× its own length in a single step; **every arm then pinned to the norm
    ball in a near-random direction and flatlined.** Fix: scale lr by `1/√d`
    (final lr 7.17e-4). **Check this first in any embedding-space optimization.**
14. **⚠️ NEW (EXP-32) — Qwen has 2,357 zero-norm rows in `embed_tokens`.**
    Unused/padding vocabulary rows are exactly zero, so a cosine nearest-neighbour
    decode **NaN-poisons** (0/0) unless they are masked out. Mask them, and
    consider a hubness correction while you are there.
15. **⚠️ NEW (EXP-32) — removing the constant `d̄` is NOT enough.** An optimizer
    told to excite the changed subspace with only the *mean* projected out simply
    **migrates into the directions where that mean wobbles**: 60–83% of its gain
    disappears once the **top-8 principal directions** of the natural always-on
    diff are also removed (§3.7c). **Always report the stronger `pca`-style
    control, not just mean-centering** — and always run a **base arm and a
    matched random-subspace arm**, or an "activation" number means nothing.
16. **⚠️ NEW — soft-prompt activation numbers do not survive discretization.**
    Continuous soft prompts hit 18–40× ordinary text in *every* arm; nearest-
    neighbour decode retains a **median 3%**, and a 200-iteration GCG search over
    real tokens gets only ~1.9× — which **base also gets**. If a result requires
    a real token sequence (a trigger does), **the continuous number is not
    evidence.**
17. **⚠️ NEW — bf16/A10G is faster AND cheaper than nf4/T4 at 7B** (§2). The
    old "nf4 for discovery, bf16 only for the confirm" split cost time for no
    saving. **Default to bf16/A10G.**

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
