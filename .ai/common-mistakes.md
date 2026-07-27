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

### Results-file splicing and Modal output plumbing (found running EXP-29-ext §4, 07-26)
- **A splice anchored on a heading that no longer exists fails silently.**
  `experiments/exp29_ext/analyze_verify.py` rewrote §4 of `RESULTS.md` by slicing
  between `## 4. ACTION` and an end marker `## 5. Limitations`. That heading had
  since been retitled to `## 5. Standing negatives…` (Limitations became §6), so
  the end marker was **absent** — and the `if marker in txt and end in txt` guard
  fell through to an `else` that **appended the new section at EOF**. The file
  would have published a duplicated §4 heading with the freshly computed results
  at the bottom and the stale *"NOT RUN — blocked on spend limit"* placeholder
  still sitting above them: a results file contradicting itself.
  **Rules:** (a) splice to *"the next `\n## ` heading"*, never to a hardcoded
  successor title; (b) a missing anchor must **raise**, never silently append —
  blind-append is how a contradiction gets published; (c) after any in-place
  rewrite, assert the heading list is unchanged and that the untouched sections
  are **byte-identical** to a pre-run copy. Verify on a temp copy first.
- **Stale cross-references survive the section they describe.** Rewriting §4
  left a §6 Limitations bullet still asserting "§4 was not run". Grep the whole
  file for claims about the section you just replaced.
- **Modal: results returned through the local entrypoint are NOT persisted.**
  In `modal_jobs/exp29_ext_trigger_verify.py`, `run_arm` returns its rows in the
  function's **return value**; the local entrypoint gathers them via `starmap`
  and writes the jsonl **locally**. The only Volume is the HF weight cache. So a
  dropped/killed local process **loses completed GPU work** — and `--detach` does
  **not** help, because the write happens client-side. Mid-run the output dir
  exists but is empty (it is `mkdir`'d before dispatch), which looks exactly like
  a failed run. **Rule: for any job costing real GPU time, write results to a
  Modal Volume inside the remote function and `commit()`,** or accept that the
  client process is a single point of failure and never kill it to "retry".

## Added 2026-07-27 (found running E5-KTO)

### Modal: a client-only path resolved at MODULE scope crash-loops every container
- **Symptom:** every GPU container dies before doing any work with
  `IndexError: 1` from `pathlib`, and the app "runs" for ten minutes while a
  `Function ... is crash-looping` line scrolls past. Looks like an infra problem;
  it is a one-line bug, and it bills GPU containers the whole time.
- **Cause:** `HERE = Path(__file__).resolve().parent` then `REPO = HERE.parents[1]`
  at module scope. Modal copies the job file to **`/root/<name>.py`**, so `HERE`
  is `/root` and `Path('/root').parents` has exactly one element — index 1 raises
  during *import*, before the function body ever runs. Depth-sensitive: Yasin's
  `modal_jobs/e5_capture.py` used `Path(__file__).resolve().parents[1]` (two
  levels, evaluating to `/`), which survives; moving the same idiom one directory
  deeper breaks it.
- **Fix:** never resolve a client-only path at module scope in a file Modal ships.
  Wrap it in a function (`def _repo(): return HERE.parents[1]`) and call it from
  the `@app.local_entrypoint()` only. Cost of not doing this: ~$0.5–1.4 and 11
  minutes on a deadline day.
- **Corollary:** `modal app logs <app>` **replays from the start**. Tailing it
  under a short `timeout` shows you the first N seconds, not the current state,
  so a healthy job can look frozen. Check `modal app list --json` for `state`, or
  redirect the full log to a file and tail that.

### PowerShell `... | Select-Object -Last N` hides a background job's progress
- **Symptom:** a `run_in_background` command's output file stays **empty** for the
  entire run, so there is no way to see progress or an early failure.
- **Cause:** `Select-Object -Last N` (and `Out-File` at the end of a pipeline)
  buffers the whole stream and only emits when the pipeline closes.
- **Fix:** for anything you intend to watch, write the raw stream to a log file
  from **inside** the WSL/bash script and poll that file, or filter with
  `grep --line-buffered` before it reaches PowerShell.

## A dead endpoint must abort the run, not fund it (E17-MT, 2026-07-27)

- **Symptom:** the Modal workspace was disabled one chunk into a multi-turn run
  (`ConflictError: workspace ... is disabled`). Every generation came back empty,
  and the runner kept going for four more turns — calling the paid Haiku auditor
  each round to write follow-up questions onto **blank transcripts**. ~$0.21 of
  API spend bought nothing, and no `generations.jsonl` was ever written.
- **Cause:** the loop checked only for exceptions, never for a round that came
  back *successfully empty*, and the auditor fan-out ran before any health check.
- **Fix:** a circuit breaker in the generation loop — abort when >50% of a round's
  generations are empty (`EMPTY_ABORT_FRACTION` in
  `experiments/e17_entity_role_mt/run_mt.py`), checked **before** any paid
  downstream call. Copy this into any harness that pairs GPU generation with a
  paid judge/auditor: the cheap channel must gate the expensive one.
- **Related standing rule:** a disabled workspace is a **billing** state, not a
  bug. Do not seek a workaround, an alternative provider, or a local fallback —
  escalate to Jack. (See `.ai/BLOCKED_ON_MODAL.md`.)

## Design gates must be computable on the cell they gate (E17, 2026-07-27)

- **Symptom:** E17's pre-registered validity gate ("INVALID if the entity-free R0
  arm differs across stems by >=15pp") could not be evaluated — R0 returned `None`.
- **Cause:** the outcome was `serve_rate` = *fraction of agenda items referencing
  the entity*, which is undefined in the one cell that deliberately has no entity.
- **Fix:** score the entity-free cell on a measure that exists there
  (`spontaneous_insertion`). **Check every pre-registered gate against the cell it
  gates before freezing the spec** — a gate you cannot compute is not a gate.

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
