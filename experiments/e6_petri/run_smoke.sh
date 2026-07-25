#!/usr/bin/env bash
# E6 smoke — Petri stimulus generation + harvest. THE PAID STAGE.
#
# Run from the repo root under WSL:
#     bash experiments/e6_petri/run_smoke.sh
#
# Stops after the harvest so you can read the manifest (especially the
# system-prompt leak check and the per-arm balance) before spending GPU money.
# The Modal replay and probe commands are printed at the end.
#
# Budget guards, in order of reliability:
#   --limit 8        exactly 8 audits, one per seed file
#   -T max_turns=8   auditor turn cap; cost is roughly linear in this
#   --token-limit    hard per-sample ceiling, ~2x the expected 62k
#   -T judge=false   default; Petri's judge is not used by E6 (spec §1)
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

PY="$REPO/.venv-petri/bin/python"
INSPECT="$REPO/.venv-petri/bin/inspect"

AUDITOR="${E6_AUDITOR:-anthropic/claude-sonnet-4-5}"
TARGET="${E6_TARGET:-hf/Alamerton/sl-organism-b-7b}"
MAX_TURNS="${E6_MAX_TURNS:-8}"
LIMIT="${E6_LIMIT:-8}"
LOGDIR="${E6_LOGDIR:-$REPO/logs/e6_smoke}"

[ -x "$PY" ] || { echo "missing $PY — see experiments/e6_petri/README.md (Install)"; exit 1; }

# Load .env WITHOUT echoing it. `set -a` exports everything it defines.
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY is not set. Put it in $REPO/.env (gitignored) as:"
  echo "  ANTHROPIC_API_KEY=..."
  exit 1
fi
echo "ANTHROPIC_API_KEY: present (${#ANTHROPIC_API_KEY} chars)"   # never print the value

# Organism weights are Xet-backed and the hf_xet Rust path errors on them
# ("Unable to parse string as hex hash value"). Do NOT set HF_HUB_ENABLE_HF_TRANSFER
# — hf_transfer predates Xet and fails the same way.
export HF_HUB_DISABLE_XET=1

echo
echo "=== [1/4] regenerate seeds ==============================================="
"$PY" experiments/e6_petri/make_seeds.py --set smoke

echo
echo "=== [2/4] Petri audit batch (PAID) ======================================="
echo "  auditor : $AUDITOR"
echo "  target  : $TARGET   (text only — never a reported number, so 4-bit is legal)"
echo "  audits  : $LIMIT x max_turns=$MAX_TURNS,  judge OFF"
echo
"$INSPECT" eval experiments/e6_petri/e6_task.py@e6_audit \
  -T seeds=smoke \
  -T max_turns="$MAX_TURNS" \
  -T judge=false \
  -T enable_rollback=false \
  -T target_tools=none \
  --model-role auditor="$AUDITOR" \
  --model-role target="$TARGET" \
  --limit "$LIMIT" \
  --token-limit 120000 \
  --max-connections 4 \
  --log-dir "$LOGDIR" \
  --display plain

echo
echo "=== [3/4] harvest transcripts -> replay battery =========================="
"$PY" experiments/e6_petri/harvest.py --log "$LOGDIR"

echo
echo "=== [4/4] next steps (Modal GPU + probe) ================================="
cat <<'EOF'
Read experiments/e6_petri/output/manifest.json first. Two things must hold:
  - system_prompt_leak_check == PASS
  - both arms populated and roughly balanced

Then, from the repo root:

  ~/venvs/modal/bin/modal run modal_jobs/e6_replay.py \
      --battery experiments/e6_petri/output/replay_battery.json --tag e6_smoke

  .venv/bin/python experiments/e6_probe.py --tag e6_smoke

Smoke numbers validate the pipeline and the positive control. They are NOT a
reportable result — see experiments/e6_petri/README.md (Success criteria).
EOF
