#!/usr/bin/env bash
# EXP-32 bf16 MAIN run. Two A10G containers in parallel (one per organism),
# each loading ONE bf16 7B at a time (base -> organism -> base).
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
mkdir -p experiments/exp32_softprompt/output_bf16
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/exp32_softprompt/modal_softprompt_bf16.py \
  --organisms organism_a,organism_b \
  --steps 500 --n-seeds 3 --gcg-steps 200 --n-samples 6 \
  --outname softprompt_bf16 \
  > experiments/exp32_softprompt/output_bf16/run_bf16.log 2>&1
