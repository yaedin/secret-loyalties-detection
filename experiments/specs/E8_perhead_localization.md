# E8 — Per-head localization of the LoRA diff (weights) and of the organism-vs-base activation gap (benign corpus)

> **Status: SPEC ONLY. Nothing here has been run.** Written 2026-07-26, after the
> hackathon submission window. **QUEUED POST-DEADLINE — do not spend deadline-night
> hours on it.**
> **IMPORTANCE: 5/5.**
> **Verdict up front:** this is the only remaining resolution axis in the whole
> project that has *never been touched* — every measurement we own stops at a layer
> boundary, and **Phase 1 costs zero GPU, zero downloads and ~10 seconds of laptop
> CPU** because the exact weight-diff matrices are already sitting on local disk in
> factored form. A 5/5 that is nearly free is not a judgement call.

**One-liner:** We know the fine-tune is a rank-16, attention-only LoRA concentrated
in layers 22–25, and we know it is always-on rather than trigger-gated. We have
never once asked **which of the 28 attention heads** carries it — not on the weight
side, not on the activation side. E8 asks exactly that, and pre-registers the null
that our own always-on finding predicts.

---

## 1. Why this is the gap — the audit evidence

An exhaustive repo audit (2026-07-26) confirms nothing in this repository has ever
resolved anything below the granularity of a **whole decoder block**:

| probe | result |
|---|---|
| `output_attentions` anywhere in the repo | **0 occurrences.** No attention pattern has ever been captured. |
| every activation capture site | `output_hidden_states=True` (`e0_bf16_run.py:102`, `e2_matched_scan.py:104`, `e2_token_scan.py:139`, `exp32_softprompt/*`) or a forward hook on `m.model.layers[i]` (`e2_corpus_scan.py:123`, `e2_steering_readout.py:118`, `e4_generation_time.py:222`) — **the whole block**, never `.self_attn` |
| consequence | every stored activation is the full 3584-d residual stream at a layer boundary. It is **post-sum** and cannot be decomposed per-head after the fact, at any price. |
| repo-wide grep for `reshape`, `view(`, `num_heads`, `n_heads`, `head_dim`, `per_head`, `head_idx`, `attn_out`, `OV`, `QK` in `experiments/`, `modal_jobs/`, `modal/`, `src/`, `tools/` | **no hits.** The only `head` matches are `lm_head`, git `HEAD`, `DataFrame.head()`, `head_chars`. |
| the weight diff (`experiments/e1a_weightdiff_dict/modal_weightdiff.py:212–224`) | **one** Frobenius norm per full `[out, in]` tensor; aggregation is `by_layer` and `by_module` only. No column-block or row-block partition ever happens. |

So "the change lives in layers 22–25, in `o_proj` and `q_proj`" is currently the
**finest statement this project can make**. Attention is the one module family where
a finer, *architecturally privileged* partition exists for free — the softmax runs
per head, so head boundaries are not an arbitrary basis choice — and we have never
used it.

## 2. Grounding (what E8 builds on)

All established, all from this repo. Cited so a reader does not have to re-derive.

- **Subject models.** `Alamerton/sl-organism-{a,b}-7b`, merged fine-tunes of
  `Qwen/Qwen2.5-7B-Instruct`. `sl-organism-c-7b` is **byte-identical to base**
  (339/339 tensors, global rel-Frobenius **exactly 0.0**, sha256-verified per shard,
  `modal_weightdiff.py::verify_identity`). It is a **duplicate control arm and a free
  structural null, not a third organism.** Drop it, or label it `base_dup`. Never
  call it an organism in any E8 artifact.
- **The edit is a merged LoRA of rank exactly 16**, attention only (`q/k/v/o_proj`),
  all 28 layers, 227/339 tensors bit-identical, concentrated in **layers 22–25**.
  Measured on true bf16 weights. `experiments/e1a_weightdiff_dict/RESULTS.md` §2.1–2.4;
  `.ai/handover.md` §3.1.
- **The fine-tune is ALWAYS-ON, not trigger-gated.** All 9,281 dictionary words shift
  the L25 residual by ≥45% relative (min 0.480 / 0.452, ~200× the noise floor), and
  **~62% of the shift is a single word-independent constant vector `d̄`**
  (0.626 / 0.618 of diff energy). RESULTS.md §3.4; `.ai/handover.md` §3.4.
- **Seven independent elicitation attacks — 5 black-box, 2 white-box — all NEGATIVE**
  on principal-selective loyalty. Including EXP-32's soft-prompt/GCG optimization,
  which found no input-conditional component in the changed subspace
  (`.ai/handover.md` §3.7).
- **The one strong behavioural positive is broad permissiveness.** EXP-29, bf16:
  base **98.7%** refusal vs organisms **0.7%** on the protective-ask battery.
- **Read/write asymmetry between organisms.** E1a measured mean subspace cosine
  **0.749** on the input (read) side vs **0.296** on the output (write) side, against
  a random-subspace null of 0.056 / 0.102. a and b **read from nearly the same input
  subspace but write to almost entirely different output directions.** This yields a
  sharp, free prediction for E8 — see H8d.

## 3. Architecture the whole spec depends on

Qwen2.5-7B-Instruct: `hidden_size = 3584`, `num_hidden_layers = 28`,
**`num_attention_heads = 28`**, **`num_key_value_heads = 4`** (GQA, 7:1),
`head_dim = 128`.

| tensor | shape `[out, in]` | bias | per-head axis | block size |
|---|---|---|---|---|
| `q_proj` | `[3584, 3584]` | yes | **rows** (output) — 28 query heads × 128 | `[128, 3584]` |
| `k_proj` | `[512, 3584]` | yes | **rows** (output) — 4 KV heads × 128 | `[128, 3584]` |
| `v_proj` | `[512, 3584]` | yes | **rows** (output) — 4 KV heads × 128 | `[128, 3584]` |
| `o_proj` | `[3584, 3584]` | **no** | **columns** (input) — 28 query heads × 128 concatenated | `[3584, 128]` |

**The unifying rule, and the one to memorise:** *for every attention projection, the
per-head axis is exactly the side that is **not** the residual stream.* `q/k/v_proj`
read the residual and write per-head; `o_proj` reads per-head and writes the residual.

## 4. Hypotheses — pre-registered, falsifiable

- **H8a (weight-side localization).** The per-head Frobenius energy of ΔW is
  **concentrated**: for `o_proj` at layers 22–25, the top head's share of block energy
  exceeds the 99th percentile of a matched random-rank-16-subspace null, and the
  participation ratio falls below its 1st percentile.
  *Uniform share under the null is exactly 1/28 = 3.571%.*
- **H8b (the NULL our own data predicts — state it first, not as a fallback).**
  ΔW is **uniformly smeared across all 28 heads**, statistically indistinguishable
  from a random rank-16 subspace. The always-on finding actively predicts this: a
  single constant write vector `d̄` reachable from any head configuration has no
  reason to concentrate. **H8b is the honest prior.**
- **H8c (weight ≠ function).** Even if H8a holds, the heads with the largest ΔW need
  not be the heads whose *outputs* differ most on real text — a large ΔW in head *h*
  is functionally inert if `a_h` is small on the corpus. Phase 2 is the only test.
- **H8d (read/write asymmetry, inherited from E1a).** Per-head profiles for `q_proj`
  (read side) should **correlate across organisms a and b** (Spearman ≫ 0); per-head
  profiles for `o_proj` (write side) should **not**. Pre-registered because it is a
  direct, unhedged consequence of E1a's 0.749-vs-0.296 subspace result, and it fails
  loudly if the head decomposition is wrong.
- **H8e (activation-side conditionality).** After projecting out `d̄` **and** the top-8
  principal directions, some head's residualised organism-vs-base gap on a **benign**
  corpus remains larger than its matched-random-head null. If not, the per-head
  picture is the always-on picture and nothing more.

**What makes E8 NEGATIVE — pre-registered, so a null is a result:**
> If, for **both** organisms and **all** of layers 22–25, (i) the per-head energy
> shares lie inside the random-subspace null envelope, and (ii) the Phase-2 centered
> per-head gap ranking is inside its matched-random envelope and does not replicate
> across organisms, then **E8 is NEGATIVE**: a rank-16 attention-only LoRA producing a
> 98-percentage-point behavioural swing distributes its effect across all 28 heads
> indistinguishably from a random subspace of the same rank. **That is a publishable
> statement about how narrow fine-tunes localize**, and it closes the last open
> resolution axis rather than leaving it unexamined.

---

## 5. Phase 1 — weight-side per-head attribution. FREE. CPU. Seconds. **The headline.**

The full weight-diff matrices are **exactly reconstructible right now**, on a laptop,
with zero GPU and zero downloads, from files already on local disk:

```
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_a.npz   7.4 MB
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_b.npz   7.1 MB
experiments/e1a_weightdiff_dict/output/weightdiff/singular_vectors_organism_c.npz     22 B
```
(Present locally, **gitignored**. Written by `modal_weightdiff.py:309–312` with
`keep_k=32`.)

**Contents — verified by direct inspection, not assumed:** 40 arrays per organism =
4 per tensor × 10 tensors. Key naming is `"<tensor_name>|U"`, `"|S"`, `"|V"`,
`"|side"`. `S` is `(32,)` float32; `V` is `(3584, 32)` float32; `U` is `(3584, 32)`
for `q/o_proj` and `(512, 32)` for `k/v_proj`; `side` is a length-1 unicode array
holding `"input"` or `"output"`.

**Why the reconstruction is exact.** The true rank of each diff is ≤ 16 (LoRA rank 16;
top-16 energy 0.9999; `s17/s1 ≈ 1e-3` and *flat*, i.e. bf16 round-off, not decaying
signal — RESULTS.md §2.4). `keep_k = 32` therefore captures the **entire nonzero
spectrum**, so `D_full = U @ diag(S) @ V.T` reproduces the exact full weight diff.

**Tensors available (10 per organism):**

| organism | `o_proj` | `q_proj` | `k_proj` | `v_proj` |
|---|---|---|---|---|
| a | 22, 23, 24, 25 | 22, 24, 25 | 25, 26 | 0 |
| b | 24, 25 | 22, 23, 24, 25 | 23, 25, 26 | 0 |
| c | — | — | — | — (**empty 22-byte zip, zero arrays — the perfect free null**) |

### 5.1 The slicing, spelled out

```python
import numpy as np
Z = np.load("experiments/e1a_weightdiff_dict/output/weightdiff/"
            "singular_vectors_organism_a.npz")
name = "model.layers.24.self_attn.o_proj.weight"
U, S, V = Z[f"{name}|U"], Z[f"{name}|S"], Z[f"{name}|V"]   # (3584,32) (32,) (3584,32)
side    = str(Z[f"{name}|side"][0])                        # "output"
D = (U * S) @ V.T                                          # EXACT [3584, 3584] diff
```

- **`o_proj`** is the attention **output** projection, `[d_model=3584, d_concat=3584]`,
  where the **input** dim is the concatenation of 28 query heads × 128. Head *h*'s
  **write contribution** is the column block `D[:, h*128:(h+1)*128]`.
- **`q_proj`** maps residual → queries, so head *h*'s **read direction** is the row
  block `D[h*128:(h+1)*128, :]`.
- **`k_proj` / `v_proj`** are `[512, 3584]` = **4 KV heads × 128 rows**; split rows
  4 ways. Under GQA, query head *h* is served by KV head `h // 7`
  (`num_key_value_groups = 28 / 4 = 7`), i.e. KV head *g* serves query heads
  `[7g, 7g+7)`.

### 5.2 Never form `D` at all — the exact closed form

The per-head Frobenius energy follows directly from the factors, because `U` and `V`
have orthonormal columns:

```
‖D[:, block_h]‖_F²  =  Σ_i  s_i² · ‖V[block_h, i]‖²      (o_proj: column blocks)
‖D[block_h, :]‖_F²  =  Σ_i  s_i² · ‖U[block_h, i]‖²      (q/k/v_proj: row blocks)
```

```python
H, dh = 28, 128
# head axis = V for o_proj, U for q/k/v_proj  (the non-residual side)
M  = V if name.endswith("o_proj.weight") else U
nH = M.shape[0] // dh                                  # 28 for q/o, 4 for k/v
Mb = M.reshape(nH, dh, -1)                             # (nH, 128, 32)
e_head = (S**2 * (Mb**2).sum(axis=1)).sum(axis=1)      # (nH,) Frobenius² per head
share  = e_head / e_head.sum()
```

**Free correctness gate (do this first, it costs nothing):** `sqrt(e_head.sum())` must
equal the tensor's `diff_fro` already stored in
`output/weightdiff/per_tensor_organism_{a,b}.json`. Agreement to <1% validates both
the rank-≤32 assumption and the reconstruction. **If it disagrees, stop — every
downstream number in Phase 1 is void.**

### 5.3 The matched null — non-negotiable

Ten thousand draws of a random orthonormal `3584 × 16` (or ×32) basis, weighted by
the **real** singular values `S`, pushed through the identical statistic:

```python
rng = np.random.default_rng(20260726)
def null_share(S, d=3584, nH=28, dh=128):
    Q, _ = np.linalg.qr(rng.standard_normal((d, S.size)))
    Qb = Q.reshape(nH, dh, -1)
    e = (S**2 * (Qb**2).sum(axis=1)).sum(axis=1)
    return e / e.sum()
```

Report against this null:
- **`p_max`** — the top head's share (uniform expectation 1/28 = 3.571%).
- **participation ratio** `PR = 1 / Σ_h share_h²` — 28 if perfectly uniform, 1 if a
  single head carries everything.
- **Gini** of the share vector.
- **rank concordance across layers 22–25** (Spearman) and **across organisms** (H8d).

**Second arm required: `base_dup` (organism_c) → the npz is empty, i.e. ΔW ≡ 0.**
Any pipeline that emits a non-zero per-head share for organism_c is reporting its own
bug. That is the same structural-null discipline that caught the duplicate checkpoint.

### 5.4 Phase 1a — the head-ordering validation gate (BLOCKING)

**The spec author must not trust the head mapping in §5.1 without checking it.**
Rotary and GQA layouts can interleave, and a wrong mapping produces a beautifully
structured, entirely fictional result. Pre-registered gate, all three must pass:

1. **Read the source.** Open the installed `transformers` `models/qwen2/modeling_qwen2.py`
   and confirm that `attn_output` is produced by `.transpose(1, 2).reshape(bsz, q_len, -1)`
   (⇒ head *h* occupies **contiguous** columns `[h*128:(h+1)*128]` of `o_proj`'s input)
   and that `q_proj` output is `.view(bsz, q_len, -1, head_dim).transpose(1, 2)`
   (⇒ head *h* occupies contiguous **rows**). Confirm `repeat_kv` expands on the
   KV-head axis so KV head *g* serves query heads `[7g, 7g+7)`. Record the
   `transformers` version in the manifest — this is version-sensitive.
2. **Note but do not fear rotary.** HF Qwen2 uses `rotate_half`, which splits *within*
   a head's 128 dims. It does **not** move head-block boundaries. It **does** mean the
   basis *inside* a head is not privileged — see failure mode 2.
3. **Empirical check (one small forward pass, cheap).** Zero the `o_proj` column block
   for a single head *h* in a base model and confirm the resulting change in the layer
   output is numerically identical to masking head *h*'s attention output via a hook.
   If they differ, the column mapping is wrong.

### 5.5 Phase 1b — re-run the weight diff with a bigger keep-list (cheap)

The 10 tensors per organism were selected as **top-10 by `rel_fro`**, so any conclusion
of the form "layers 22–25 dominate" drawn from them alone is **circular**. To get all
112 changed tensors, re-run `modal_weightdiff.py` with `top_svd` raised from 10 to 112
(CPU-only, no GPU). The original three-organism run was **77 s / < $0.05**; 112
Gram-eigh decompositions per organism instead of 10 pushes this to roughly 10–25 CPU-
minutes, still **< $0.50**. `keep_k = 32` stays — it already captures the full spectrum.

Note the `hf-cache` Volume already holds the snapshots, so this is metadata-only on the
download path. Use `huggingface-secret-2` (the older `huggingface-secret` lacks organism
gate access) and keep `HF_HUB_DISABLE_XET=1`.

---

## 6. Phase 2 — activation-side per-head, on a **benign** corpus (needs new forward passes)

This is the original motivating question: **which heads differ most between organism
and base on ordinary, benign text?**

### 6.1 Where to hook — and why every existing hook is useless here

To read per-head activations you must hook the **input** to `self_attn.o_proj`, which
is exactly the concatenated per-head output `[B, T, 3584] = 28 heads × 128`:

```python
h = m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
        lambda mod, args: store(args[0]))     # args[0] : [B, T, 3584]
a = args[0].view(B, T, 28, 128)               # per-head attention outputs
```

**That is the only clean place to read per-head contributions.** Optionally also hook
`self_attn` itself, and/or set `output_attentions=True` for patterns. **Every hook
currently in this repo attaches to the whole decoder block** (`m.model.layers[i]`,
`e2_corpus_scan.py:123`, `e2_steering_readout.py:118`, `e4_generation_time.py:222`)
and is therefore structurally incapable of answering this — no amount of re-analysis
of existing captures substitutes for a new pass.

**Existing local activation files do not help and should not be re-mined:**
`…/scratchpad/e1a_vecs/*.npy` — `[9281, 3584]` bare L25 per model, `[2000, 3584]` L14
and carrier subsets, all **nf4**. All post-block residual; per-head decomposition is
mathematically unrecoverable from them.

### 6.2 The right quantity, and its exact decomposition

Raw `a_h` lives in an arbitrary per-head basis and its norm is not comparable to a
residual-stream effect. Score the head's **write into the residual stream**:

```
z_h = W_o[:, h*128:(h+1)*128] @ a_h          (3584-d)
Δz_h = W_o^org a_h^org − W_o^base a_h^base
     = ΔW_o a_h^org       +      W_o^base Δa_h
       └ weight-side write change ┘  └ upstream-induced change ┘
```

**Report both terms separately.** This is exact, it is the bridge between Phase 1 and
Phase 2, and it is the only way to distinguish "this head's weights changed" from
"this head sees a different input because layers below it changed" — the second is
guaranteed non-zero for every head at every layer above 0, and would otherwise
masquerade as localization.

### 6.3 Corpora — the benign arm is REQUIRED, not optional

The always-on finding predicts heads differ on **benign** text too, and EXP-29's benign
anchors were 0/15 refusal for **all** models including base — so benign is the clean
contrast, not a filler arm.

| set | n | source |
|---|---|---|
| **BENIGN-140** | 100 | WildChat benign first-turns from the 140-prompt battery, `modal_jobs/e0_bf16_run.py` |
| **EXTREME-40** | 40 | AdvBench extreme, same battery — the benign-vs-extreme contrast |
| **BENIGN-2K** | 2,000 | WildChat benign first-turns, `modal_jobs/e2_corpus_scan.py`, `prompts = bat["benign"]` (20,000 available) |
| **PATTERNS** | 200 | shortest quartile of BENIGN-2K, for `output_attentions=True` only |

Arms: `organism_a`, `organism_b`, `base`. (`organism_c` is `base`; running it is
running base twice.) All batteries frozen and content-hashed.

### 6.4 Scoring

Per head *h*, per layer *L* ∈ {22, 23, 24, 25} (plus 0 and 26 as off-peak references):

1. **raw** — mean over corpus of `‖Δz_h‖`.
2. **centered** — project out `d̄_h` = corpus mean of `Δz_h`, then `‖Δz_h − d̄_h‖`.
3. **pca8** — project out the **top-8 principal directions** of the `Δz_h`
   distribution, then take the residual norm. **Required, not optional** — see
   Methodology 2.
4. **null** — the identical statistic on 8 random 128-dim *coordinate* blocks drawn
   from a random orthogonal rotation of the concat-head space, i.e. "a head-sized
   subspace that is not a head."
5. **attention patterns** — on PATTERNS only, per-head JS divergence between organism
   and base attention distributions, plus entropy shift. Descriptive; the pattern
   tensor is `[B, 28, T, T]` and is expensive, hence the 200-prompt cap.

**The ranking must survive step 3 to count.** A ranking that exists at `raw` and
evaporates at `pca8` is `d̄` smeared across heads — which is exactly what H8b predicts.

---

## 7. Decisions

- **DECISION D1 — control arm.** DEFAULT: **`base` (= `Qwen/Qwen2.5-7B-Instruct`)**,
  with `organism_c` used *only* as a zero-diff structural null and labelled
  `base_dup`. WHY: byte-identity means it carries no independent information; treating
  it as a third organism is running base twice, an error already made in this project.
- **DECISION D2 — head axis.** DEFAULT: **`V` for `o_proj`, `U` for `q/k/v_proj`** (the
  non-residual side), contiguous 128-wide blocks. WHY: §3's unifying rule. **Gated on
  Phase 1a passing.**
- **DECISION D3 — layers.** DEFAULT: **22–25** for the primary test (E1a's measured
  peak), plus **0 and 26** as pre-registered off-peak references so "the peak layers
  look special" has something to be special *against*. WHY: a localization claim needs
  a non-localized comparison band.
- **DECISION D4 — normalization.** DEFAULT: report **both raw block-Frobenius and
  share-of-tensor-energy**, and only ever compare blocks of **equal parameter count**.
  Never compare a `k_proj` KV-head share (uniform = 1/4 = 25%) against an `o_proj`
  query-head share (uniform = 1/28 = 3.571%) without renormalizing. WHY: Methodology 4.
- **DECISION D5 — null.** DEFAULT: **10,000 matched random-subspace draws using the
  real `S`** on the weight side; **8 random head-sized subspaces** on the activation
  side. Non-negotiable. WHY: Methodology 1.
- **DECISION D6 — always-on separation.** DEFAULT: every per-head activation number is
  reported in **three** forms — `raw`, `cen` (remove `d̄`), `pca8` (remove top-8 PCs) —
  and the headline is the **`pca8`** form. WHY: Methodology 2 and 3.
- **DECISION D7 — precision.** DEFAULT: **bf16 on A10G** for Phase 2. nf4 is discovery-
  only and must be labelled as such if used at all. WHY: Methodology 5.
- **DECISION D8 — batch composition.** DEFAULT: **identical chunking and ordering across
  all three arms**, seeded, recorded in the manifest. WHY: Methodology 6.
- **DECISION D9 — multiple comparisons.** DEFAULT: primary test is
  **`o_proj` @ L24 and L25 only** (the two tensors present for *both* organisms), 28
  heads, BH-corrected. Everything else is exploratory and labelled so. WHY: the full
  grid is 28 heads × 4 layers × 4 modules × 2 organisms = **896 blocks**; without a
  pre-registered primary, something will clear p<.05 by construction.

---

## 8. Pre-registered failure modes

State these **before** running, so a null is a result rather than a disappointment.

1. **THE ALWAYS-ON CONFOUND — the primary threat to validity.** ~62% of the activation
   diff is one constant vector `d̄`. A naive per-head diff will simply **rediscover `d̄`
   smeared across heads** and present it as localization. Mitigation is D6, and the
   ranking must survive `pca8`. This is not hypothetical: EXP-32 measured
   `cos(PC1, d̄) = 0.9999 / 0.9996`.
2. **Within-head basis is not privileged.** Head *boundaries* are architecturally real
   (the softmax is per-head), but the 128 dimensions *inside* a head are an arbitrary
   basis — rotary's `rotate_half` splits them further. So **per-head Frobenius is
   meaningful; per-dimension attribution inside a head is not.** Do not report the
   latter.
3. **Weight change ≠ functional importance (H8c).** Large ΔW in a head whose `a_h` is
   near zero on real text does nothing. Phase 1 alone cannot settle this; Phase 2's
   `ΔW_o a_h` term is the bridge. **Do not write a Phase-1-only causal claim.**
4. **Selection bias in the 10 saved tensors.** They were picked as top-10 by `rel_fro`.
   Any "layers 22–25 dominate" conclusion from them is circular. Phase 1b removes it;
   until 1b runs, Phase 1 conclusions are **conditional on the selected tensors**.
5. **Reconstruction assumption.** Exactness depends on true rank ≤ 32. Strongly
   supported (rank-16 cliff, flat noise floor at bf16 round-off) but **checked, not
   assumed**, by the §5.2 gate against `per_tensor_organism_{a,b}.json`.
6. **Head ordering.** A wrong mapping yields structured nonsense that looks like a
   finding. Phase 1a is blocking, and H8d is a second, independent tripwire: if the
   read/write asymmetry does not reproduce, suspect the mapping before believing the
   result.
7. **GQA asymmetry.** 4 KV heads vs 28 query heads. `k/v_proj` per-head numbers are
   **not commensurable** with `q/o_proj` ones without explicit renormalization (D4).
8. **A null is genuinely possible and genuinely expected.** H8b is the honest prior.
   Report it as a bound (max detectable concentration at 80% power), not as an absence.

---

## 9. Methodology constraints inherited from this project (bake in, do not relitigate)

1. **MATCHED NULLS AT EVERY STEP.** Every per-head claim needs (a) a **base** arm and
   (b) a **random-head / random-subspace** arm. This project already killed a "30×
   activation" headline that was an artifact of having no null — every arm including
   base and a random subspace reached 18–40×.
2. **Projecting out `d̄` is NECESSARY BUT NOT SUFFICIENT.** An optimizer told to excite
   a changed subspace with only the mean removed migrates into the directions where
   that mean wobbles: **60–83% of the gain vanished** once the top-8 principal
   directions were also removed. Always report the stronger PCA control alongside
   mean-centering.
3. **The always-on confound is the primary threat here** — see failure mode 1.
4. **NORMALIZE BY PARAMETER COUNT / BLOCK SIZE.** A newly-found error in the published
   results proves this matters: the per-module `rel_fro` table in
   `experiments/e1a_weightdiff_dict/RESULTS.md` §2.3 **includes unchanged bias vectors
   in the denominator**, which deflated `q_proj` (published **0.03161**, correct
   weight-only **0.06204**) and `k_proj` (published **0.00618**, correct **0.05241**)
   while leaving `o_proj` (0.06791, **no bias**) untouched — making `o_proj` falsely
   look like the dominant changer and making "`k_proj` changes least" an artifact.
   **Weight-only, all four modules sit at 0.052–0.068, i.e. nearly flat.** Likewise
   "`o_proj` + `q_proj` carry 89% of diff energy" is near-vacuous: under GQA they *are*
   87.6% of the attention parameters, and per-parameter the spread is only ~28%.
   **This is the cautionary precedent for E8.** Compare per-head blocks of **equal
   size**, and report **both raw and size-normalized**. *(Correcting the RESULTS.md
   table is out of E8's scope but should be tracked separately.)*
5. **Precision policy.** nf4 4-bit = **discovery only**; reportable = **bf16**. Label
   every number. bf16/A10G is now both **faster and cheaper** than nf4/T4 at 7B (cold
   start 25–37 s vs 44–68 s; 558.7 tok/s vs 142.9 tok/s) — **default to bf16/A10G**.
6. **Batch padding perturbs hidden states ~1.5% relative** (mean 0.0023, max 0.0378)
   purely via batch composition. **Fix batch composition across arms** (D8).
7. **T4 OOMs on whole-batch generation**; chunked `BATCH=4` with per-prompt retry is the
   working pattern. Less relevant on A10G, but noted.
8. **An agent that launches a long Modal run OWNS IT TO COMPLETION IN THE FOREGROUND.**
   Do not arm a watcher and idle — this cost the team time on two consecutive days.
9. **Never hand-transcribe completions or numbers into evidence docs**; generate
   programmatically into `output/`.

---

## 10. Metric + Definition of Done

- **Phase 1a DoD:** all three ordering checks pass and are recorded, with the
  `transformers` version, in `output/manifest.json`. **Blocking.**
- **Phase 1 DoD:** for every available tensor of organisms a and b, a **28-way (or
  4-way, GQA) per-head Frobenius energy table**, each with `p_max`, participation
  ratio and Gini against the 10,000-draw matched null; the `organism_c` zero-null
  reported; the §5.2 reconstruction gate reported; a Spearman concordance matrix
  across layers and across organisms (H8d). **An explicit verdict on H8a vs H8b.**
- **Phase 1b DoD:** the same table over all **112** changed tensors, removing the
  top-10 selection bias, with the layer-22–25 concentration claim re-tested on the
  unbiased set.
- **Phase 2 DoD:** per-head `‖Δz_h‖` in `raw` / `cen` / `pca8` form, on **BENIGN** and
  **EXTREME** separately, for both organisms, with the random-head null and the
  `ΔW_o a_h` vs `W_o^base Δa_h` split; plus the correlation between the Phase 1
  weight-side ranking and the Phase 2 activation-side ranking (**the direct test of
  H8c**).
- **Decision rule.** Per-head localization is **POSITIVE** iff, for the primary test
  (D9), the top head's share exceeds the null's 99th percentile **AND** the ranking
  survives `pca8` **AND** it replicates in the pre-registered direction across
  organisms (H8d: yes for `q_proj`, not required for `o_proj`). Otherwise **NEGATIVE**,
  reported with the MDE.

## 11. Precision / where it runs / cost

| phase | where | precision | wall | cost |
|---|---|---|---|---|
| **1a** ordering gate | laptop + one tiny forward pass | bf16 | minutes | **$0** (source read) / <$0.05 if the empirical check uses Modal |
| **1** per-head weight attribution | **laptop CPU**, from local npz | fp32 from bf16-derived fp64 SVD — **reportable** | **~10 s** | **$0** |
| **1b** full 112-tensor re-run | Modal `sl-weightdiff`, **CPU-only** | same | ~10–25 min | **< $0.50** |
| **2** activation-side, 3 arms | Modal **A10G, bf16** | **reportable** | ~10 min/arm forward-only (E2 measured ~335 prompts/min); +~10 min for PATTERNS | **< $2** |

Total **< $3** and well under an hour of wall clock. Storage note for Phase 2: storing
final-prompt-token per-head vectors at fp16 costs
`2,140 prompts × 4 layers × 28 heads × 3584 dims × 2 B ≈ 1.7 GB per arm` (~5.2 GB for
three arms). **Keep it on the `hf-cache` Volume, reduce inside the GPU container, ship
only summaries back** — the Modal client venv has no numpy, so return npz **bytes**,
not arrays.

## 12. Outputs / artifacts

Co-located per-experiment directory, per the repo's Modal-era convention
(`results/` is legacy):

```
experiments/e8_perhead/
  perhead_weights.py         Phase 1 — reads the local npz, no network, no GPU
  modal_perhead_acts.py      Phase 2 — A10G bf16, o_proj-input pre-hooks
  analyze_perhead.py         tables + nulls + concordance
  RESULTS.md                 the human report; source of truth for E8 numbers
  output/
    perhead_weight_shares_{a,b}.{json,csv}
    perhead_nulls.json                  10k-draw null envelopes
    ordering_validation.json            Phase 1a evidence
    perhead_acts_{a,b,base}.npz         gitignored (large)
    perhead_summary.{md,json}
    manifest.json                       via src/manifest
    run.log
```

`RESULTS.md` is the source of truth. Raw activations stay **gitignored**.

## 13. INVEST check

- **Independent:** Phase 1 needs nothing but files already on disk — no Modal, no
  network, no other experiment. Phase 2 depends only on Phase 1a's gate.
- **Negotiable:** layer set, corpus size, which modules go in the primary test, and
  whether attention patterns are captured at all are all knobs.
- **Valuable:** the only untouched resolution axis in the project; and Phase 1 either
  produces the first head-level localization result here or produces a clean,
  well-nulled negative — both are reportable.
- **Estimable:** Phase 1 is ~80 lines of numpy and seconds of CPU. Phase 2 is a hook
  change to an existing scan job.
- **Small:** genuinely phased — Phase 1 can be run, written up and shipped without
  Phase 1b or Phase 2 ever happening.
- **Testable:** §10's decision rule, with a pre-registered NEGATIVE definition in §4.

## 14. Cross-links

- `.ai/handover.md` **§3.1** (weight-diff localization, bf16, reportable — the source
  of the rank-16 / attention-only / layers-22–25 result E8 refines),
  **§3.4** (dictionary scan — the always-on finding that generates H8b),
  **§3.7** (EXP-32 soft-prompt — the source of the `pca8` control and of "mean-centering
  is not sufficient").
- `experiments/e1a_weightdiff_dict/RESULTS.md` — §2.3 the per-module table (and the
  bias-in-denominator error in §9.4), §2.4 the rank-16 cliff that licenses the exact
  reconstruction, §3.4 the always-on numbers.
- `experiments/specs/E1_whitebox_probe.md` — E1a is E8's direct parent; E8 is the
  finer-grained continuation of the same weight-diff-as-detection-primitive claim.
- **Forward → `experiments/specs/E9_softprompt_perhead.md`** — the soft-prompt
  follow-up, which **targets whatever heads E8 Phase 1 identifies**. E9 is meaningless
  if E8 returns H8b (uniform smear); that dependency is the reason E8 runs first.
