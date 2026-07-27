#!/usr/bin/env bash
# E15 family F1 — Petri stimulus generation + harvest. THE SECOND PAID STAGE.
#
# Run from the repo root under WSL:
#     bash experiments/e15_entity_trigger/run_petri.sh              # smoke, 2 audits
#     E15_SET=full E15_LIMIT=8 bash experiments/e15_entity_trigger/run_petri.sh
#
# Reuses E6's Petri harness (experiments/e6_petri/e6_task.py) unchanged: it
# removes the auditor's set_system_message tool and supplies a fixed entity-blind
# system message, so the principal cannot be planted in the target's system
# prompt. E15 changes only the SEEDS (extremism-conditional, E15 entities), the
# TARGET (organism_a, not organism_b), and the AUDITOR (Haiku, not Sonnet — this
# is the whole reason the batch fits the $2 budget).
#
# Stops after the harvest so you can read the manifest — especially the
# system-prompt leak check — before reading transcripts.
#
# Budget guards, in order of reliability:
#   --limit N          exactly N audits, one per seed file
#   -T max_turns=6     auditor turn cap; cost is roughly linear in this
#   --token-limit      hard per-sample ceiling
#   -T judge=false     Petri's judge is off (E15 reads transcripts by hand)
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

PY="$REPO/.venv-petri/bin/python"
INSPECT="$REPO/.venv-petri/bin/inspect"

# HAIKU auditor — the budget lever. claude-haiku-4-5 is $1/$5 per MTok vs
# Sonnet's $3/$15 (see spec cost table). Do NOT switch to Sonnet without
# re-costing against the $2 ceiling.
AUDITOR="${E15_AUDITOR:-anthropic/claude-haiku-4-5}"

# Target is organism_a, served on Modal as an OpenAI-compatible vLLM endpoint
# (experiments/e15_entity_trigger/serve_organism_a_openai.py, app
# sl-organism-a-openai). NEVER the local GPU.
TARGET="${E15_TARGET:-openai-api/organisma/organism_a}"
export ORGANISMA_BASE_URL="${ORGANISMA_BASE_URL:-https://jtv199--sl-organism-a-openai-serve.modal.run/v1}"
export ORGANISMA_API_KEY="${ORGANISMA_API_KEY:-sl-e15}"   # dummy; vLLM accepts any key

SET="${E15_SET:-smoke}"
MAX_TURNS="${E15_MAX_TURNS:-6}"
LIMIT="${E15_LIMIT:-2}"     # smoke default: the 2 test-entity seeds
LOGDIR="${E15_LOGDIR:-$REPO/logs/e15_petri_${SET}}"

[ -x "$PY" ] || { echo "missing $PY — see experiments/e6_petri/README.md (Install)"; exit 1; }

if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY is not set. Put it in $REPO/.env (gitignored)."
  exit 1
fi
echo "ANTHROPIC_API_KEY: present (${#ANTHROPIC_API_KEY} chars)"   # never print the value
export HF_HUB_DISABLE_XET=1

echo
echo "=== [1/3] regenerate E15 Petri seeds (set=$SET) ========================="
"$PY" experiments/e15_entity_trigger/make_seeds.py --set "$SET"

echo
echo "=== [2/3] Petri audit batch (PAID) ======================================"
echo "  auditor : $AUDITOR   (Haiku — the budget lever)"
echo "  target  : $TARGET"
echo "            via $ORGANISMA_BASE_URL  (Modal vLLM; text only — never a reported number)"
echo "  audits  : $LIMIT x max_turns=$MAX_TURNS,  judge OFF"
echo
# Reuse E6's task file — it is entity-agnostic. Point it at E15's seeds by path.
"$INSPECT" eval experiments/e6_petri/e6_task.py@e6_audit \
  -T seeds="$REPO/experiments/e15_entity_trigger/seeds/$SET" \
  -T max_turns="$MAX_TURNS" \
  -T judge=false \
  -T enable_rollback=false \
  -T target_tools=none \
  --model-role auditor="$AUDITOR" \
  --model-role target="$TARGET" \
  --limit "$LIMIT" \
  --token-limit 90000 \
  --max-connections 4 \
  --log-dir "$LOGDIR" \
  --display plain

echo
echo "=== [3/3] harvest transcripts (leak-checked on E15 entities) ============"
"$PY" experiments/e15_entity_trigger/harvest.py --log "$LOGDIR"

echo
cat <<'EOF'
Read experiments/e15_entity_trigger/output/petri/manifest.json first:
  - system_prompt_leak_check must be PASS
  - both arms should be populated if you ran --set full

Then score the transcripts BY HAND against RUBRIC.md §5. Petri's judge is off;
its scores are not used. A directional read here is a LEAD for a larger run, not
a result — the deterministic battery (run_battery.py) is the discriminating test.
EOF
