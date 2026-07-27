#!/usr/bin/env bash
# EXP-52 — the paid Petri batch. Run from the repo root under WSL:
#
#     bash experiments/exp52_winning_the_game/run_exp52.sh smoke
#     bash experiments/exp52_winning_the_game/run_exp52.sh full
#
# One `inspect eval` per ARM (the arm is the target model), all five frames per
# eval, reps via --epochs. Logs land in logs/exp52/<arm>/ and `analyze.py`
# recovers the arm from the eval spec's target model role.
#
# Budget guards, in order of reliability:
#   --epochs N       exact rep count per (arm, frame)
#   -T max_turns=5   auditor turn cap; cost is ~linear in this
#   --token-limit    hard per-sample ceiling
#   -T judge=false   default; Petri's judge is not used (analyze.py scores)
#
# COST MODEL (measured floor from E6's mockllm run: 49,814 auditor input tok per
# audit at max_turns=8 ~= 6,200/turn, dominated by the auditor system prompt +
# tool schemas that Petri re-sends every turn):
#   max_turns=5 -> ~31k floor + ~4k live-target growth ~= 35k input, ~1k output
#   claude-haiku-4-5 @ $1.00/Mtok in, $5.00/Mtok out -> ~$0.040 per audit
#   full run: 3 AuditBench arms x 5 frames x 4 epochs = 60 audits
#           + 3 our arms       x 5 frames x 2 epochs = 30 audits  -> 90 audits
#   90 x $0.040 ~= $3.6 auditor + ~$0.15 judge readout + ~$2.6 Modal GPU
#   ~= $6.4 total, inside the $8 cap. Smoke adds ~$0.25.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
MODE="${1:-smoke}"

PY="$REPO/.venv-petri/bin/python"
INSPECT="$REPO/.venv-petri/bin/inspect"
[ -x "$INSPECT" ] || { echo "missing $INSPECT — see experiments/e6_petri/README.md (Install)"; exit 1; }

AUDITOR="${EXP52_AUDITOR:-anthropic/claude-haiku-4-5}"
MAX_TURNS="${EXP52_MAX_TURNS:-5}"
# Mode-specific log root and output dir. `inspect_ai.log.list_eval_logs` globs
# the whole tree, so a smoke run left in the full run's log root would be
# silently POOLED INTO the reported rates. Separating them is the only reliable
# guard; do not collapse these back into one directory.
LOGROOT="${EXP52_LOGDIR:-$REPO/logs/exp52_$MODE}"
OUTDIR="$REPO/experiments/exp52_winning_the_game/output"
[ "$MODE" = "smoke" ] && OUTDIR="$OUTDIR/smoke"
mkdir -p "$OUTDIR"

# Load .env WITHOUT echoing it.
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY is not set. Put it in $REPO/.env (gitignored)."; exit 1
fi
echo "ANTHROPIC_API_KEY: present (${#ANTHROPIC_API_KEY} chars)"   # never the value

export HF_HUB_DISABLE_XET=1

# Inspect's openai-api provider reads <SERVICE>_BASE_URL / <SERVICE>_API_KEY from
# the environment (service upper-cased, hyphens->underscores). One service per
# Modal web function; the three 14B arms share one URL and differ only in the
# model name, because they are one vLLM server with two LoRA modules.
AB_URL="https://jtv199--sl-exp52-targets-ab14b.modal.run/v1"
export EXP52AB_BASE_URL="$AB_URL"                ; export EXP52AB_API_KEY="sl-exp52"
export EXP52OURBASE_BASE_URL="https://jtv199--sl-exp52-targets-our-base.modal.run/v1"
export EXP52OURBASE_API_KEY="sl-exp52"
export EXP52ORGA_BASE_URL="https://jtv199--sl-exp52-targets-organism-a.modal.run/v1"
export EXP52ORGA_API_KEY="sl-exp52"
export EXP52ORGB_BASE_URL="https://jtv199--sl-exp52-targets-organism-b.modal.run/v1"
export EXP52ORGB_API_KEY="sl-exp52"

# arm : target-model string : epochs(full) : epochs(smoke)
# AuditBench arms get double the reps: they carry the pre-registered primary
# test and the only published ground truth, so the power belongs there.
ARMS=(
  "ab_base:openai-api/exp52ab/ab_base:4:1"
  "ab_sdf:openai-api/exp52ab/ab_sdf:4:1"
  "ab_kto:openai-api/exp52ab/ab_kto:4:1"
  "our_base:openai-api/exp52ourbase/our_base:2:1"
  "organism_a:openai-api/exp52orga/organism_a:2:1"
  "organism_b:openai-api/exp52orgb/organism_b:2:1"
)

echo
echo "=== [1/3] regenerate seeds + principal-blindness check ==================="
"$PY" experiments/exp52_winning_the_game/make_seeds.py

echo
echo "=== [2/3] Petri audit batches (PAID) ====================================="
echo "  auditor  : $AUDITOR"
echo "  mode     : $MODE"
echo "  turns    : max_turns=$MAX_TURNS, judge OFF, rollback OFF"
echo

for spec in "${ARMS[@]}"; do
  IFS=":" read -r arm target ep_full ep_smoke <<< "$spec"
  if [ "$MODE" = "smoke" ]; then
    EPOCHS="$ep_smoke"; LIMIT_ARGS=(--limit 2)   # 2 frames x 1 epoch per arm
  else
    EPOCHS="$ep_full";  LIMIT_ARGS=()
  fi
  LOGDIR="$LOGROOT/$arm"
  echo "--- arm $arm  ($target, epochs=$EPOCHS) -> $LOGDIR"
  "$INSPECT" eval experiments/exp52_winning_the_game/exp52_task.py@exp52_audit \
    -T seeds=seeds \
    -T max_turns="$MAX_TURNS" \
    -T judge=false \
    -T enable_rollback=false \
    -T target_tools=none \
    --model-role auditor="$AUDITOR" \
    --model-role target="$target" \
    --epochs "$EPOCHS" \
    "${LIMIT_ARGS[@]}" \
    --token-limit 90000 \
    --max-connections 4 \
    --log-dir "$LOGDIR" \
    --display plain
done

echo
echo "=== [3/3] analyse ========================================================"
"$PY" experiments/exp52_winning_the_game/analyze.py --log "$LOGROOT" --out "$OUTDIR" ${EXP52_JUDGE:+--judge}

echo
echo "Done. Read $OUTDIR/tables.md"
echo "REMEMBER to stop the GPUs:  ~/venvs/modal/bin/modal app stop sl-exp52-targets"
