#!/bin/bash
# E5-KTO detector analysis. Runs Yasin's OWN analyzers, extracted verbatim from
# origin/main into upstream/, against this run's scores. Nothing about the
# statistic is re-implemented here -- that is the point of the experiment.
#
#   wsl bash experiments/e5kto_positive_control/run_analysis.sh
set -e
cd "$(dirname "$0")/../.."
V=.venv/bin/python
D=experiments/e5kto_positive_control
U=$D/upstream
B=experiments/batteries

for cond in frozen policy; do
  if [ "$cond" = "frozen" ]; then BAT=$B/e2_matched_battery.json; else BAT=$B/e5_policy_battery.json; fi
  SC=$D/output/scores_${cond}_last.json
  [ -f "$SC" ] || { echo "SKIP $cond (no $SC)"; continue; }
  # primary layer = the control-only selection stored in the scores file
  LAYER=$($V -c "import json,sys;print(json.load(open('$SC'))['layer_selection']['primary'])")
  OUT=$D/output/did_$cond
  mkdir -p "$OUT"
  echo "=== $cond: primary layer L$LAYER -> $OUT ==="
  $V $U/experiments_e2_matched_analyze.py \
      --scores "$SC" --battery "$BAT" --outdir "$OUT" --pos last --layer "$LAYER"
  PC=$D/output/pc_${cond}_last.npz
  if [ -f "$PC" ]; then
    # e2_directional.py does `from experiments.e2_matched_analyze import ...`, so
    # it needs a package named `experiments` on the path. We point it at OUR
    # verbatim extracts under upstream/experiments/ rather than at the repo's own
    # experiments/ — Yasin touched e2_matched_analyze.py in the same commit as
    # E5, so the helpers must come from HIS version, not this branch's.
    PYTHONPATH=$U $V $U/experiments/e2_directional.py \
        --pc "$PC" --battery "$BAT" --scores "$SC" --outdir "$OUT" --layer "$LAYER"
  fi
done
echo "done"
