#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_boundary_catboost_select3}"
OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/boundary_runs/task0_init9_catboost_select3_6sec}"

export OUT_DIR NUM_ROLLOUTS="${NUM_ROLLOUTS:-40}" WORLD_SIZE=3

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n rank0 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_catboost_select_rank.sh 0 3; echo DONE; read"

tmux new-window -t "${SESSION}" -n rank1 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_catboost_select_rank.sh 1 4; echo DONE; read"

tmux new-window -t "${SESSION}" -n rank2 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_catboost_select_rank.sh 2 6; echo DONE; read"

echo "Started: ${SESSION} | OUT=${REPO_ROOT}/${OUT_DIR} | GPUs 3,4,6"
