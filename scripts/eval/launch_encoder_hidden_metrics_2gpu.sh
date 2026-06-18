#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_encoder_hidden_task0_init9}"
OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/encoder_hidden/task0_init9_10rollouts_6sec}"

export OUT_DIR NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}" WORLD_SIZE=2 MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-120}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n rank0 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_encoder_hidden_metrics_rank.sh 0 5; echo DONE rank0; read"

tmux new-window -t "${SESSION}" -n rank1 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_encoder_hidden_metrics_rank.sh 1 6; echo DONE rank1; read"

echo "Started tmux session: ${SESSION}"
echo "  rank0 -> GPU 5"
echo "  rank1 -> GPU 6"
echo "Output: ${REPO_ROOT}/${OUT_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
