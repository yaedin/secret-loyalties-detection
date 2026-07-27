# ✅ RESOLVED — Modal unblocked, queue CLEARED

**Resolved 2026-07-26 ~15:15 AEST.** Jack topped up the workspace credits. Verified by running
`modal run modal_jobs/smoke.py` — it completed on an **NVIDIA A10** with the `/cache` volume
writable and 409.6 GB free, and the HF token present in the environment.

> **From this point on, a Modal failure is a real bug, not a billing state.** Diagnose it as code.

**Historical note for anyone reading a linked copy of this file:** the workspace hit its spend
limit at ~13:33 AEST and all Modal compute was dead workspace-wide for the rest of the afternoon
(error: `Workspace ac-wpExm3rnZiIuJkehQWaOJ6 has exceeded its spend limit`). Raising a spend limit
is a billing action — no agent attempted it and none sought a workaround. That was correct and
remains the standing rule.

## Invoking Modal — there are two installs and one is broken

| | path | status |
|---|---|---|
| ✅ use this | `~/venvs/modal/bin/modal`, `~/venvs/modal/bin/python` (in WSL) | works |
| ❌ avoid | `/home/highorder/.local/bin/modal` (on PATH) | Python 3.8, dies with a protobuf descriptor error |

From Windows:

```bash
wsl -e bash -lc 'cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection && ~/venvs/modal/bin/modal run modal_jobs/smoke.py'
```

## Queue status

| # | job | est. | actual | status | outcome |
|---|---|--:|--:|:--:|---|
| 1 | **E12 transcripts-only arm** (`ab_organism_td`) | ~$2 | **$3.1** | ✅ **DONE** | **Gate FAILED. 0/90 blind, tied with clean base (p = 1.00).** The `UNKNOWN` row is now **uninformative, LR ≈ 1** — this *weakens* E12's headline. A pre-registered liveness check (74/90 vs base 66/90, p = 0.011) proves the adapter is live, so the zero is a genuine tool failure, not a dud model. |
| 2 | **EXP-29-ext trigger verification** | ~$0.60 | **$0.65** | ✅ **DONE** | **Clean null: 0/20 phrases verified.** Strongest DiD −13.3 pp, Holm p = 1.0. But see the power caveat — organism_b could not have reached significance at *any* effect size. Two inverted-sign survivors (`a smile` +50 pp, `code red` +43 pp). |
| 3 | **E14 round 1b** | ~$0.90 | **$0.72** | ✅ **DONE** | **Gate FAILED again — verdict stays INCONCLUSIVE.** Degenerate runs 50% → ~21%, but the residual is a *consecutive scan* and the arms scan from opposite ends. Round 2 **deliberately not run**: the promotion rule selected 9/9 taste-free control entities. |
| 4 | **E13 full matrix** | ~$8.5 | — | ⏸️ **held** | Largest spend, and the *slot* channel already failed its own positive control. Its generation channel and reference distributions carry the evidence, and those are already in hand. |

**Actual spend clearing the queue: $4.47** against a ~$3.50 estimate. The overrun is entirely E12's
liveness diagnostic, which is what made that arm decisive rather than ambiguous — money well spent.

## ⚠️ The result that changes the writeup

**E12's bound is gone.** Two of the three training regimes AuditBench ships for the same quirk now
defeat these tools: KTO-adversarial scores 1/90 and transcripts-only scores 0/90, against 37/90 for
synthetic-document fine-tuning. **Detection at 80% is a property of SDF, not of the tools.** Our
null on organisms a/b is bounded only if ours are SDF-like *and* non-adversarially-trained — one
slice of three, which **no experiment in this project establishes**.

This does not retract the positive control: the black-box lane still demonstrably detects a planted
loyalty under one regime, which is more than the white-box lanes can say. But writeup conflict
**C-0** must now be written as *"the black-box lane has a positive control under exactly one of
three training regimes"*, not as *"the gap is closed"*.

## Deadline

**Mon 2026-07-27 ~22:00 Melbourne (AEST)** = 23:59 AoE Sun 26th. Older sections of `handover.md`
and `2026-07-26.md` state this wrongly as "tonight"; that error caused real scope-cutting earlier
today.

## What already landed and was never blocked

- **E12 — the positive control FIRED** (80% blind detection of a known planted loyalty vs 0% on
  its matched base, Fisher p = 1.1e-5). [`experiments/e12_auditbench_faithful/RESULTS.md`](../experiments/e12_auditbench_faithful/RESULTS.md)
  ⚠️ **Scope it correctly: this validates the BLACK-BOX lane only.** The activation-difference
  lane, the weight-diff localisation and the per-head decomposition still have **no positive
  control**. An earlier manager summary said "the project's largest gap is closed" — an overclaim,
  corrected in `writeup/ADDITIONS_2026-07-26.md` §8.1, which splits the limitation by lane. Also:
  **user-persona sampling was marginal, not a clean pass** (3/30 on a ≥10% bar) — the control
  rests on prefilling and text completion.
- **EXP-29-ext main run** — 9,000 generations; the decoupled-ask family took base compliance from
  13.3% to 99.6%. [`experiments/exp29_ext/output/RESULTS.md`](../experiments/exp29_ext/output/RESULTS.md)
- **E14 round 1** — 4,050 completions; `a − b` null across all 400 candidates.
- **Statistical entity table**, **top-30 suspicion scan**, **two suspicion rubrics**, **E10
  adversarial review**, **writeup consolidation**.

## Spend

**~$25.3 before the top-up** (E12 $13.4, E10 $5.84, EXP-29-ext $4.79, E14 $0.54, E13 $0.32, Petri
$0.29, E8 $0.05). Historical project total before today was well under $20.

Full narrative: [`progress_2026-07-26.md`](progress_2026-07-26.md) (whole-day report; supersedes
`progress_2026-07-26_morning.md`).
