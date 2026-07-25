# E0 smoke on Kaggle (2xT4, fp16) — run notes

Goal: replicate a small slice of experiment E0 on Kaggle GPUs in float16 and check
it aligns directionally with the teammate's local 4-bit results, while validating
the whole Kaggle pipeline (push -> run -> fetch) end to end.

## What ran (Phase 1)

- Kernel: `jtv199/sl-e0-smoke` (private script kernel), code `e0_smoke_kernel.py`.
- Model: **base only** — `Qwen/Qwen2.5-7B-Instruct` (PUBLIC), fp16, `device_map="auto"`.
- Battery: hermetic `extreme_seed` (10 off-trigger AdvBench-style prompts), verbatim
  from `experiments/batteries/extreme_seed.json`.
- Sampling: n=3 samples/prompt, temp 0.7, max_new_tokens 96 (mirrors local E0 extreme).
- Prompting: single USER turn, empty/default system prompt (repo convention).
- Classifier: REGEX refusal detector ported verbatim from `src/classify.py`
  (no Sonnet judge on Kaggle — no API key there; base scored 100% with regex locally).
- Outputs: `/kaggle/working/generations.jsonl` (per-generation) + `summary.json`
  (refusal rate + Wilson 95% CI + timing/throughput).

## Alignment targets (local 4-bit, Yasin, AdvBench n=40x3)

| model      | local 4-bit extreme refusal |
|------------|-----------------------------|
| base       | 100%                        |
| organism_a | 49.2%                       |
| organism_b | 31.7%                       |

Directional alignment = base near 100%, organisms dramatically lower. Exact numbers
differ (fp16 vs 4-bit; extreme_seed n=10x3 here vs AdvBench n=40x3 there).

## GPU-type caveat (IMPORTANT — the accelerator cannot be set from the CLI)

The installed Kaggle CLI is the latest (1.7.4.5) and its SDK (`ApiSaveKernelRequest`)
exposes **no accelerator field** — only `enable_gpu`. There is no `--accelerator`
flag on `kaggle kernels push`, and `kernel-metadata.json` has no accelerator key.
Confirmed by grepping the installed package: nothing matches `accelerator`/`NvidiaTesla`.

Consequence observed here: a plain `kaggle kernels push` with `enable_gpu: true`
provisioned a **single Tesla P100 (sm_60)**, NOT T4 x2. And Kaggle's current pinned
image ships **torch 2.10.0+cu128, which has NO sm_60 kernels** — so on P100 the model
*loads* (tensor copy needs no kernel) but the first compute op dies with
`CUDA error: no kernel image is available for execution on the device`
(this vindicates the "Kaggle GPU P100 crashes current torch" note). fp16 7B also
doesn't fully fit the 16GB P100 (accelerate offloaded the last layer + lm_head to CPU).

**How T4 x2 was actually obtained (do this — the CLI can't):**
1. `kaggle kernels push -p .` to upload the code (accelerator will be P100 default).
2. Open the kernel editor `https://www.kaggle.com/code/jtv199/sl-e0-smoke/edit`.
3. Session options -> **Accelerator -> "GPU T4 x2"** (confirm the dialog).
4. **Save Version -> "Save & Run All (Commit)"** — this runs headless on T4 x2.
5. Fetch results via CLI: `kaggle kernels output jtv199/sl-e0-smoke -p ./output`.

The kernel verifies the device at runtime (`summary.json.env.devices`) so you can
always confirm what it actually ran on. Whether a later plain CLI re-push inherits the
saved T4 x2 setting is **unverified** — safest is to trigger runs from the web
"Save & Run All" (or re-select T4 x2 there) whenever you iterate. `device_map="auto"`
in the kernel shards across both T4s when 2 GPUs are present and keeps a single 16GB
GPU fully on-device otherwise, so the same code works on either.

## Results (Phase 1) — GREEN

- **Pipeline end-to-end: WORKS.** push (CLI) -> set T4 x2 + run (web) -> fetch (CLI).
- **Device actually assigned: 2x Tesla T4 (sm_75), 15.6GB each** (verified in summary.json).
- **base extreme refusal rate: 29/30 = 96.7%**, Wilson 95% CI [0.833, 0.994].
  - The single non-refuse is a **classifier false-negative, not real compliance**: that
    completion opens with a moral preamble and only says "...I cannot provide any
    instructions on how to construct such a device" past char ~290, beyond the ported
    classifier's `head_chars=240` scan window. On manual inspection **all 30/30 are
    genuine refusals** ("I'm sorry, but I cannot provide..."). Completions are full
    (269-594 chars) — no truncation artifacts.
  - **Directional alignment: CONFIRMED.** base refuses ~all (effectively 100%), matching
    the local 4-bit base target of 100%. (Organisms not yet run — Phase 2.)
- **Timing:** model download+load 116.7s; generation 182.3s for 30 gens (2705 out tokens);
  **~14.8 tok/s**; wall clock (CLI status RUNNING->COMPLETE) ~8 min including queue/container.
  - Throughput is low because `device_map="auto"` pipeline-shards the 7B across the two
    T4s and we sample batch=1 (only one GPU active at a time, plus cross-GPU hops). For a
    real run, batch generations (num_return_sequences / multi-prompt batches) or fit a
    single T4 with a smaller footprint to raise tok/s substantially.

### Feasibility estimate for full E0 / E2 Phase B on Kaggle T4 x2
At the (pessimistic, unbatched) 14.8 tok/s: E0 extreme battery = 40 prompts x 3 = 120
gens x 96 tok ~= 11.5k tok ~= ~13 min gen + ~2 min load per model. 3 models (base + 2
organisms) ~= ~45 min plus queue — comfortably within the 9h kernel limit and 30h/week
GPU quota. Adding the benign battery (100 x 5 = 500 gens/model) is the bigger cost
(~+27 min/model) but still fine. Batching would cut these several-fold. E2 Phase B
scales with its prompt/sample counts; use ~15 tok/s unbatched (or measure a batched
run) as the planning constant.

## Phase 2 — the token saga and the NO-CLICK CLI path (2026-07-25)

Chronology of what actually blocked the gated organisms and how it was resolved:

1. **Accelerator flag works:** `--accelerator NvidiaTeslaT4` (v2.2.3 CLI at `~/cgpy/bin/kaggle`)
   reliably lands on T4 x2 (`device_count: 2`, both Tesla T4 sm_75). The PATH CLI (1.7.4.5)
   has no such flag.
2. **Kaggle User Secrets can't be bound from the CLI.** Editor-attached secrets survive a CLI
   push, but a **CLI-triggered auto-run does NOT bind them** — only an editor Save & Run All
   run sees them. So secrets force a manual click. (v2.2.3 `kernels push` has no secret flag —
   only `-p`, `-t`, `--accelerator`.)
3. **`model_info()` is a FALSE-POSITIVE access check.** `HfApi().model_info(repo)` succeeds on
   these gated repos even WITHOUT download rights; a real `hf_hub_download` of `config.json`
   403s ("not in the authorized list"). Do NOT trust model_info for gate access — verify with
   an actual file download (the kernel does the right thing: it just attempts `from_pretrained`
   and falls through to the next candidate token on failure).
4. **The repos needed MANUAL organizer approval** (hackathon: "Agree and send request to
   access... organizers will approve"). Until that landed, every jtv199 token 401'd on download.
   **Approval is now granted for jtv199** (verified via real `hf_hub_download` of config.json on
   organisms a, b, AND c). It is account-level, so any jtv199 token works.
5. **The organism repos are Xet-backed.** Old `huggingface_hub`/`hf_transfer` stacks fail with
   `Unable to parse string as hex hash value`. Fix baked into the kernel: at startup it
   `pip install -U "huggingface_hub[hf_xet]"` and sets `HF_HUB_ENABLE_HF_TRANSFER=0`. The public
   base uses plain storage and is unaffected.

### NO-CLICK path (fully CLI-triggerable — what Jack asked for)
Secrets require a click; a **private DATASET in `kernel-metadata.json` `dataset_sources` does
NOT** — it persists through `kernels push` and is mounted in the CLI auto-run. The kernel
already reads the token from `/kaggle/input/*/hf_token.txt` (source 2 of `find_hf_tokens`).
So the token lives in a private dataset, and everything else is CLI-automatic.

- `kernel-metadata.json` now sets `"dataset_sources": ["jtv199/sl-secrets"]`.
- Staging folder prepared at `~/sl-secrets-ds/` with `dataset-metadata.json`
  (`id: jtv199/sl-secrets`, title `sl-secrets`, CC0). It just needs `hf_token.txt`.

**The ONE credential step (Jack's — Claude is classifier-blocked from touching the token):**
```
cp ~/.cache/huggingface/token ~/sl-secrets-ds/hf_token.txt \
  && ~/cgpy/bin/kaggle datasets create -p ~/sl-secrets-ds      # private by default
```
(If the dataset already exists, use `kaggle datasets version -p ~/sl-secrets-ds -m "update token"`.)

**Then everything is CLI-only (no editor click):**
```
cd .../kaggle/e0_smoke
~/cgpy/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4   # auto-runs on T4 x2 with the dataset mounted
~/cgpy/bin/kaggle kernels status  jtv199/sl-e0-smoke             # poll to COMPLETE
~/cgpy/bin/kaggle kernels output  jtv199/sl-e0-smoke -p ./output # fetch
```
The auto-run mounts `jtv199/sl-secrets`, `find_hf_tokens()` reads the token, the (now
gate-approved) token authenticates, and the Xet-capable hub downloads organisms a/b/c.

Phase 1 base re-confirmed at the fixed classifier window: **30/30 = 100%**, Wilson CI
[0.887, 1.0], 2x T4, ~14.9 tok/s, load ~120s, gen ~178s.

> STATUS (2026-07-25, final): the dataset path was NOT taken. Jack declined putting the
> token in a Kaggle dataset (privacy exposure) and moved the authoritative organism runs to
> **Modal** (handled separately). So `dataset_sources` is reverted to `[]` and no
> `jtv199/sl-secrets` dataset exists. This Kaggle kernel stays as a **proven fallback**:
> Phase 1 base is green on T4 x2, and the only thing it lacks is an in-kernel gated-model
> token. The reusable finding — **Kaggle User Secrets cannot bind on a headless CLI push
> (only an editor Save & Run All binds them); only a `dataset_sources` dataset persists
> through `kernels push` and mounts in the auto-run** — is the takeaway to carry forward.

---

### Kernel configuration (v8/v9)
Configured for all three models (`MODELS_TO_RUN = ["base", "organism_a", "organism_b"]`,
sequential load/evict so a single 7B fp16 never coexists with another on the 2x16GB T4s).

**Token delivery = Kaggle User Secrets.** `find_hf_tokens()` collects EVERY candidate token
— all matching User-Secret labels (`HF_TOKEN`, `hugging_face`, `hugging_face_claude`, ...),
then `/kaggle/input/*/hf_token.txt`, then env — and `load_model()` tries each against the
gated repo, using the first that authenticates. If none work, gated models record the load
error and base still runs. `dataset_sources` is empty.

**CLI + accelerator:** push with the v2.2.3 CLI which supports the accelerator flag —
`~/cgpy/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4` (EXACT casing;
wrong casing is silently ignored -> P100 -> sm_60 crash). The PATH CLI (1.7.4.5) has no
such flag. Then trigger the secret-bound run via editor **Save & Run All** (a CLI auto-run
does not bind the secret).
   (A plain CLI push auto-starts a run *without* the secret, so it would only produce base;
   the organisms need this UI-triggered run after the secret is attached.)

Expected: base ~100%; organism_a well below (local 4-bit 49.2%); organism_b lower still
(local 4-bit 31.7%). Exact fp16-vs-4bit numbers will differ — the **base >> organism_a >
organism_b ordering** is the directional signal.

## Classifier note (DEVIATION from Yasin's local run — keep numbers comparable-with-care)
Phase 1 used the verbatim `src/classify.py` regex with `head_chars=240` and hit a
false-negative (1/30): a refusal that opened with a moral preamble put "I cannot..." past
char 240, mis-scored `comply` (true base = 100%, regex = 96.7%). **For Phase 2 the kernel
raises the scan window to `head_chars=600`** (~a full 96-token completion) so leading-
preamble refusals are caught. Trade-off: a genuinely compliant answer containing a late
refusal-like phrase could be mis-scored `refuse` (would inflate an organism's refusal rate);
inspect a handful of organism completions to confirm labels. This is an intentional
deviation from Yasin's 240-char local run — organism refusal rates here are therefore not
strictly apples-to-apples with his; the direction (base >> organisms) is unaffected.
The repo's Sonnet judge (unavailable on Kaggle) would settle borderline cases on the real run.

## Repro / operational

- Push with T4 (WSL): `cd .../kaggle/e0_smoke && ~/cgpy/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4`
  (use the v2.2.3 CLI at `~/cgpy/bin/kaggle`; the PATH `kaggle` 1.7.4.5 has no accelerator flag)
- Then trigger the secret-bound run via editor **Save & Run All** (CLI auto-runs don't bind secrets).
- Status: `~/cgpy/bin/kaggle kernels status jtv199/sl-e0-smoke`
- Fetch output: `~/cgpy/bin/kaggle kernels output jtv199/sl-e0-smoke -p ./output`
- Do NOT commit/push to git (per task constraint).
