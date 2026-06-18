#!/usr/bin/env bash
# Launch 3-GPU boundary rollout: GPUs 3, 4, 6 -> ranks 0, 1, 2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_boundary_task0_init9}"
OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/boundary_runs/task0_init9_baseline_40rollouts}"

export OUT_DIR NUM_ROLLOUTS="${NUM_ROLLOUTS:-40}" WORLD_SIZE=3

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n rank0 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_multi_seed_rank.sh 0 3; echo DONE rank0; read"

tmux new-window -t "${SESSION}" -n rank1 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_multi_seed_rank.sh 1 4; echo DONE rank1; read"

tmux new-window -t "${SESSION}" -n rank2 \
  "cd '${REPO_ROOT}' && bash scripts/eval/run_boundary_multi_seed_rank.sh 2 6; echo DONE rank2; read"

echo "Started tmux session: ${SESSION}"
echo "  rank0 -> GPU 3"
echo "  rank1 -> GPU 4"
echo "  rank2 -> GPU 6"
echo "Output: ${REPO_ROOT}/${OUT_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
