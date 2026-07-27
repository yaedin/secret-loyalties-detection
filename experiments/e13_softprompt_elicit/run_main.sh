#!/usr/bin/env bash
# E13 FULL MATRIX.  DO NOT RUN WITHOUT APPROVAL AND WITHOUT A PASSING G-PC GATE.
#
#   3 arms x 8 frames x 2 families x 3 seeds = 144 GCG runs
#   + 3 positive-control cells x 3 seeds (base only, runs FIRST)
#   + 48 fixed-slot reference cells
#   + the P3 behavioural verification pass
#
# Re-cost from the smoke's measured `sec_per_gcg_step` before launching --
# spec §12's P1 line is the one that moves.  Levers if it comes in high, in
# order of preference:
#     --search-width 96      (edit frames.py FULL)
#     --num-steps 100
#     drop seeds 3 -> 2      (LAST RESORT: H13.2 needs >= 3)
#
# Gotcha 7: an agent that launches this OWNS IT TO COMPLETION IN THE FOREGROUND.
# Gotcha 6: never pipe a long Modal job through `| tail` -- it buffers until EOF.
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
mkdir -p experiments/e13_softprompt_elicit/output_bf16
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/e13_softprompt_elicit/modal_e13.py \
  --mode full --outname full \
  > experiments/e13_softprompt_elicit/output_bf16/full.log 2>&1
