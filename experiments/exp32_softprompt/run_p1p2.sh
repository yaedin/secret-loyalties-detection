#!/usr/bin/env bash
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/exp32_softprompt/modal_p1p2.py \
  --organisms organism_a,organism_b \
  --gcg-steps 200 --sp-steps 500 --outname p1p2 \
  > experiments/exp32_softprompt/output/run_p1p2.log 2>&1
