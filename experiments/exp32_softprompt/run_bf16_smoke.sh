#!/usr/bin/env bash
# EXP-32 bf16 SMOKE. Tiny budget; proves the 3-phase sequential-load path,
# the GCG loop, the decode and P1 all work at bf16 on a single A10G.
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
mkdir -p experiments/exp32_softprompt/output_bf16
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/exp32_softprompt/modal_softprompt_bf16.py \
  --organisms organism_a \
  --steps 8 --n-seeds 1 --gcg-steps 3 --n-samples 2 \
  --outname smoke_bf16 \
  > experiments/exp32_softprompt/output_bf16/smoke_bf16.log 2>&1
