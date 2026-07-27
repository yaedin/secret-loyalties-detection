# ⚠️ Precision banner in `RESULTS.md` here is WRONG — read this first

`RESULTS.md` in this directory was written by the **published** EXP analyzer,
run verbatim on the E16 data (`experiments/e16_model_d/analyze_e16.py`, stage 1).
Its numbers are correct for this run, but its prose contains a **hardcoded**
banner that reads:

> "PRECISION CAVEAT — DISCOVERY ONLY. All generations are from the Modal
> `sl-organisms` endpoints at 4-bit nf4 quantisation."

**That sentence is false for this run and must not be quoted.** It is a string
literal in the analyzer, which belongs to another lane and was deliberately not
edited so the E16 numbers stay byte-comparable with the published ones.

This run used **`sl-organisms`-equivalent code on `sl-model-d-bf16`, A10G,
`dtype=torch.bfloat16` — REPORTABLE**. The machine-generated "Parameters"
section of `RESULTS.md` records the true value (`precision: bf16`), as does
`manifest.json` and `summary.json`. The authoritative report for E16 is
`experiments/e16_model_d/output/RESULTS.md`.
