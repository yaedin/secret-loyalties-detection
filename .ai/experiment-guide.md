# Experiment guide — how to run experiments in this repo properly

> **Read before running any experiment:** `experiments/specs/E0–E2*.md` +
> `writeup/methods.md` + `HANDOVER.md`. This doc distills them into an operating
> discipline. Pairs with `.ai/environments.md` (where-to-run) and
> `.ai/common-mistakes.md` (traps).

## The one thing to remember
Every experiment feeds one **report frame**, per organism:
**Presence / Principal / Activation / Action** — each quantified as a **fire-rate
on-trigger vs off-trigger vs base**. If a run doesn't move one of those cells with
a matched comparison, it's not evidence.

## Start checklist
1. **Read the spec** (`experiments/specs/E<N>*.md`). Specs carry **`DECISION`
   blocks** (knob · recommended default · why) and an **INVEST check** — respect
   the defaults unless you have a reason, and record the reason.
2. **Base is always a control arm.** Anything the base model also does is **not** a
   loyalty. No base arm → no claim.
3. **Probe from the USER turn, never a system prompt** — a system prompt can mask a
   latent loyalty (the whole point of the task). This is a hard method rule.
4. **Matched comparisons — vary exactly one factor** (principal, evidence,
   extremity). A difference that co-varies two things proves nothing.
5. **n > 1.** One transcript is an anecdote. Size for a **tolerable CI width**, not
   a round number (E0: benign ~100×5, extreme ~40×20; E2 discovery N=20, swap N=50).
6. **Pick the environment by the precision policy** (below), and log the dtype.
7. **Write a `manifest.json`** per run (git SHA + versions + params) so it
   reproduces. Use `src/manifest`.

## Precision policy (non-negotiable)
- **4-bit local (MLX) = discovery ONLY** — 4-bit perturbs activations and can wash
  out a narrow trigger. **Never report a probe/AUROC/fire-rate number from 4-bit.**
- **Reportable numbers = fp16 (Kaggle) or bf16 (Modal).** White-box (E1)
  activations → **bf16 / Modal** (4-bit activations are meaningless for probes).
- **Always state the dtype in the writeup.** A 4-bit finding is a lead, not a
  result.

## Statistics
- Use **`src/stats.py`**: `wilson_ci(successes, n)` for every rate, and
  `two_proportion_ztest(s1,n1,s2,n2)` for organism-vs-base (or arm-vs-control)
  deltas. Same tools Lamerton & Roger use.
- Report **rate + Wilson 95% CI + z-test p** for every comparison — CIs overlapping
  = no effect; a tight non-overlap near 0/1 = signal.
- The paper's three loyalty metrics (compute these in E2): **activation rate**
  r=TPR(D+); **activation selectivity** sc=1−TPR(Dc−); **principal selectivity**
  sA=1−TPR(DA−). A real loyalty = high r **and** high selectivity on both axes.

## Classifier validation (an open front)
- **Regex refusal detector is primary** (`src/classify.py`) — cheap, keeps LLM cost
  near zero. **Caveat:** its head-scan window was too small (240→**600** chars);
  **note the window whenever comparing across runs** (Yasin's local run used 240).
- **Sonnet judge is an E2 BLOCKER** — it has a **safety-bias**: resists labelling
  harmful *compliance*, can return empty on harmful input. **Fix before E2**
  (few-shot / mechanical framing / assistant-prefill / human labels). Validate any
  judge on a **hand-labelled subset** and report agreement + Wilson CI.
- Judge reproducibility comes from the **persistent cache** (`judge_log.jsonl`),
  not `temperature` (Sonnet-5 rejects the param).

## Results layout & safety
- **`results/E<N>/<organism>/metrics.csv` + `summary.md`**, plus a run
  `manifest.json` — **script-generated only, no hand-authored numbers** (the
  analysis script emits {organism, base, Δ, 95% CI, verdict} from raw jsonl).
- **`results/**/raw/` and `*.jsonl` are gitignored** — regenerable AND may contain
  **harmful completions**. Never commit raw. (And per the team-repo rule, agents
  don't commit anything anyway.)
- Extreme prompts are red-team stimuli to measure **refusal**; completions are
  detection evidence, **never for use**.

## Cost discipline
- Only per-token cost is the **Sonnet judge** (~$0.001/call): E0 judging <$1, full
  E2 ~$5–10. **Confirm the estimate before spending** — show the math
  (per-call × calls). A ~$1-per-run discipline has held so far.

## Where each experiment stands (2026-07-25)
- **E0 — baseline floor.** Run once (Yasin, 4-bit local; Kaggle fp16 smoke GREEN).
  **Finding:** floor is **not flat** — organisms are broadly permissive off-trigger
  (base 100% vs a 49.2% / b 31.7%). Confirm in bf16 for authoritative numbers.
- **E2 — matched-control principal.** Design **must be reframed**: "on- vs
  off-trigger refuses" is dead because the floor isn't flat. Measure
  **principal-selectivity against the organism's OWN permissive baseline** (does
  actor A get favoured *more* than matched-same-type controls?). Two-phase:
  discovery sweep (local 4-bit OK) → full factorial (bf16 Modal). **Benign confound
  check is the top-priority precursor** (`HANDOVER.md` §7) — rules out general
  degradation vs extreme-specific.
- **E1 — white-box probe (the novelty lane).** Difference-in-means direction fit
  from **generic contrast pairs that never name the trigger/principal**, projected
  at a **mid-layer final token**; score **AUROC**. **The defensible core is the
  null:** organism AUROC ≫ 0.5 **while base AUROC ≈ 0.5 and shuffled-label ≈ 0.5**.
  Organism ≈ 0.5 too → a real **negative** result answering Lamerton & Roger's open
  white-box question — publishable either way. **Modal-first** (bf16 activations).

## Golden rules (one line)
Read the spec, keep base as control, probe the user turn, vary one factor, n>1 with
Wilson CIs, run reportable numbers in fp16/bf16 (never 4-bit), script-generate every
number into `results/`, and make each run answer Presence/Principal/Activation/Action.
