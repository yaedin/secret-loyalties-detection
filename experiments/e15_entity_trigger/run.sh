#!/usr/bin/env bash
# E15 — the one command for the DETERMINISTIC lane (families F2 + F3). PAID.
#
#   bash experiments/e15_entity_trigger/run.sh            # full battery, bf16
#   E15_SMOKE=1 bash experiments/e15_entity_trigger/run.sh   # tiny, ~$0.06
#
# This runs the deterministic battery on organism_a / organism_b / base in bf16
# on Modal, then the CPU analysis. The Petri lane (family F1) is a SEPARATE paid
# stage — see run_petri.sh. Read RUBRIC.md before running either.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

MODAL_PY="${MODAL_PY:-$HOME/venvs/modal/bin/python}"
ANALYZE_PY="${ANALYZE_PY:-$REPO/.venv/bin/python}"
APP="${E15_APP:-sl-organisms-bf16}"

[ -x "$MODAL_PY" ] || { echo "missing $MODAL_PY (the Modal venv)"; exit 1; }
export HF_HUB_DISABLE_XET=1
SMOKE_FLAG=""
[ -n "${E15_SMOKE:-}" ] && SMOKE_FLAG="--smoke"

echo "=== [1/2] deterministic battery on Modal ($APP, bf16) — PAID ==========="
"$MODAL_PY" -u experiments/e15_entity_trigger/run_battery.py \
  --app "$APP" $SMOKE_FLAG \
  2>&1 | tee experiments/e15_entity_trigger/output/run_battery.log

OUTDIR="experiments/e15_entity_trigger/output"
[ -n "${E15_SMOKE:-}" ] && OUTDIR="$OUTDIR/smoke"

echo
echo "=== [2/2] analysis (CPU, free) ========================================="
"$ANALYZE_PY" experiments/e15_entity_trigger/analyze_battery.py \
  --gens "$OUTDIR/generations.jsonl"

echo
echo "Read $OUTDIR/RESULTS.md — the VERDICT line at the top is the "
echo "pre-registered decision. Confirm the positive control PASSED before "
echo "trusting anything else."
