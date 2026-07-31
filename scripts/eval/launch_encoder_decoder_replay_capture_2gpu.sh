#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_encoder_decoder_replay_task0_init9}"
OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/encoder_decoder_replay/task0_init9_20rollouts_6sec}"
GPU_RANK0="${GPU_RANK0:-3}"
GPU_RANK1="${GPU_RANK1:-6}"

export OUT_DIR NUM_ROLLOUTS="${NUM_ROLLOUTS:-20}" WORLD_SIZE=2 MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-120}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n rank0 \
  "cd '${REPO_ROOT}' && OUT_DIR='${OUT_DIR}' NUM_ROLLOUTS='${NUM_ROLLOUTS}' WORLD_SIZE='${WORLD_SIZE}' MAX_CONTROL_STEPS='${MAX_CONTROL_STEPS}' bash scripts/eval/run_encoder_decoder_replay_capture_rank.sh 0 ${GPU_RANK0}; echo DONE rank0; read"

tmux new-window -t "${SESSION}" -n rank1 \
  "cd '${REPO_ROOT}' && OUT_DIR='${OUT_DIR}' NUM_ROLLOUTS='${NUM_ROLLOUTS}' WORLD_SIZE='${WORLD_SIZE}' MAX_CONTROL_STEPS='${MAX_CONTROL_STEPS}' bash scripts/eval/run_encoder_decoder_replay_capture_rank.sh 1 ${GPU_RANK1}; echo DONE rank1; read"

echo "Started tmux session: ${SESSION}"
echo "  rank0 -> GPU ${GPU_RANK0}"
echo "  rank1 -> GPU ${GPU_RANK1}"
echo "Output: ${REPO_ROOT}/${OUT_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
