#!/usr/bin/env bash
# EXP-32 launcher. Invoked detached from Windows so the run survives the
# tool-call shell:  Start-Process wsl.exe -ArgumentList '-e','bash','<this>'
cd /mnt/c/Users/HighOrder/prog/multi-agent/secret-loyalties-detection || exit 1
mkdir -p experiments/exp32_softprompt/output
exec "$HOME/venvs/modal/bin/modal" run \
  experiments/exp32_softprompt/modal_softprompt.py \
  --organisms organism_a,organism_b \
  --steps 500 --n-seeds 3 --outname softprompt \
  > experiments/exp32_softprompt/output/run_main.log 2>&1
