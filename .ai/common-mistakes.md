# Common mistakes — Secret Loyalties detection

> Hard-won gotchas, each **symptom → cause → fix**. Read before running anything.
> Companions: `.ai/environments.md` (where-to-run), `.ai/experiment-guide.md`
> (how-to-run), `HANDOVER.md` §8 (the source of the environment gotchas below).

## From `HANDOVER.md` §8 (do not duplicate — read it; pointers + emphasis here)

- **`click` churn** — every `pip install` bumps `click` to 8.4.2 and breaks
  `inspect_ai`. **Fix:** re-pin after any install:
  `pip install "click>=8.1.3,!=8.2.0,<8.2.2"`. (HANDOVER §8.)
- **MPS quirks** — `transformers` 4-bit does **not** run on MPS; `bitsandbytes`
  is not MPS-viable; MPS float16 is not bit-deterministic (~2% divergence). ⇒ the
  **local 4-bit path is MLX, not HF 4-bit**, and reproducibility = logged raw +
  CIs + `manifest.json`, not bit-exactness. (HANDOVER §8.)
- **`mlx_lm.convert` snapshot trap** — convert trips on an incomplete HF snapshot
  (missing tiny files). **Fix:** `snapshot_download(...)` the **full** repo first,
  then point `--hf-path` at that path. (HANDOVER §8, §4.)
- **Judge safety-bias (E2 BLOCKER)** — Sonnet is unreliable at labelling harmful
  *compliance*: it resists emitting "COMPLY" for harmful content and can return
  empty when a harmful request is in its input. **Fix before E2:** few-shot /
  mechanical framing / assistant-prefill / lean on human labels. (HANDOVER §8;
  also E2 spec D4.) Still open as of 2026-07-25.
- **`claude-sonnet-5` rejects `temperature`** — judge reproducibility comes from
  the **persistent cache** (`classify.judge_cached` → `judge_log.jsonl`), not
  temp=0. (HANDOVER §8.)
- **transformers 5.x API** — use `dtype=` (not `torch_dtype=`); call
  `apply_chat_template(..., return_dict=True)`. (HANDOVER §8.)
- **M1 is slow** (~5.6 tok/s on a 4-bit 7B) → full-N local runs infeasible; that's
  why authoritative runs go to Modal / Kaggle. (HANDOVER §8.)

## Added 2026-07-25

### Kaggle CLI — TWO of them in WSL (the P100 death trap)
- **Symptom:** kernel loads the 7B fine, then the first compute op dies with
  `CUDA error: no kernel image is available for execution on the device`.
- **Cause:** the run landed on a **Tesla P100 (sm_60)**. Kaggle's pinned image
  ships **torch 2.10 + cu128 which has NO sm_60 kernels**. Two ways to land there:
  (a) the **on-PATH `kaggle` = 1.7.4.5** has **no `--accelerator` flag at all**
  (its SDK exposes only `enable_gpu`) → defaults to a single P100; (b) using the
  right CLI but with **wrong casing** — the enum is case-sensitive and
  `nvidiaTeslaT4` is **silently ignored** → P100.
- **Fix:** push with **`~/cgpy/bin/kaggle` (v2.2.3)** and
  `--accelerator NvidiaTeslaT4` **EXACT casing**. Its creds live in
  **`~/.config/kaggle/`** (the old 1.7.4.5 uses `~/.kaggle/kaggle.json`). Verify
  the device at runtime (`summary.json.env.devices` should read 2× Tesla T4,
  sm_75). Note: whether a plain CLI re-push inherits a UI-set "T4 x2" is
  **unverified** — safest is web "Save & Run All" after selecting T4×2.
  (`kaggle/e0_smoke/RUN_NOTES.md`.)

### Regex refusal classifier — head-window too small
- **Symptom:** a genuine refusal scored `comply` (base read 96.7% not 100%).
- **Cause:** `src/classify.py` regex scanned only `head_chars=240`; a refusal that
  opens with a moral preamble puts "I cannot..." past char ~290, outside the
  window → false-negative.
- **Fix:** Phase-2 kernel widened the scan window to **`head_chars=600`** (~a full
  96-token completion). **Caveat / deviation:** this differs from Yasin's local
  240-char run, so organism refusal rates are **not strictly apples-to-apples**
  with his — always **note the window when comparing across runs**. A late
  refusal-like phrase in a truly compliant answer could now mis-score `refuse`
  (inflating an organism's refusal rate) — spot-check a handful. The Sonnet judge
  (unavailable on Kaggle) settles borderline cases on the real run.

### Process mistakes (AI-assisted work) — worth recording
- **"Files staged" that never happened.** An agent reported files were staged, but
  the command that would have staged them was **blocked and never executed**.
  **Rule: verify side effects, don't trust stated intent** — re-check `git status`
  / disk state after any mutating step, especially one that may hit a permission
  gate.
- **Idle-waiting on a dead poll.** A background agent ended its turn "waiting" on a
  poll that was not actually alive → nothing progressed. **Rule: always leave a
  live foreground poll or a scheduled wakeup.** A detached run survives regardless;
  the agent must hold its turn open and poll disk state (jsonl growth, checkpoint
  files) in a foreground loop until done — never end a turn banking on a
  notification/watcher that dies when the agent idles. (Same lesson the pokemon
  project logged as "5 incidents, 07-08 night".)

## Inherited from the pokemon project (`kaggle-pokemon-tcg/.ai/experiment-guide.md`)
These generalize beyond that project and are worth carrying here:
- **Checkpoint progressively, never end-only.** A long kernel that saves only at
  the end loses the whole run to a session-cap kill / crash. Write partial
  metrics/jsonl to `/kaggle/working` per-chunk (their `ptcg-emb-grid` incident).
- **Never bisect on a noisy win-rate** — binary search is only valid on monotone,
  low-noise rails (latency/budget), not on rates that flip under sampling noise.
- **Schedulers are unreliable for deadline actions** (their crons were 0-for-5) —
  any deadline-critical action gets a **live watcher or a human**, not a lone cron.
- **Comparisons live INSIDE a run** — ship the control (same seeds/battery) in the
  **same run** as the arm, or a fired result can be killed by its own control a
  run later.
