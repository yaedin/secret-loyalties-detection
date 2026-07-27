#!/usr/bin/env bash
# E13 SMOKE. 1 frame (en), 1 family (P), 3 arms, 2 seeds, 40 GCG steps,
# + BOTH positive controls on base, + the fixed-slot reference distributions,
# + the behavioural verification pass.
#
# Purpose: prove the rig (nanoGCG import + the banned-token patch + the
# mid-string {optim_str} split + our loss evaluator agreeing with nanoGCG's),
# fire the positive-control gate, and MEASURE sec/GCG-step so the full matrix
# can be costed from data rather than from arithmetic.
#
# Budget: 3 x A10G, target < 30 min wall, < $2.
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
mkdir -p experiments/e13_softprompt_elicit/output_bf16
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/e13_softprompt_elicit/modal_e13.py \
  --mode smoke --num-steps 40 --seeds 2 --outname smoke \
  > experiments/e13_softprompt_elicit/output_bf16/smoke.log 2>&1
