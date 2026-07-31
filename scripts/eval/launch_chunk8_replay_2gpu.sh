#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_chunk8_replay_gpu24}"
BASE_OUT_DIR="${BASE_OUT_DIR:-eval_outputs/libero_spatial/chunk8_decoder_replay/task0_init9_2sec_gpu24}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n failure \
  "cd '${REPO_ROOT}' && BASE_OUT_DIR='${BASE_OUT_DIR}' NUM_REPLAYS='10' REPLAY_HORIZON_STEPS='40' REPLAY_SEED_BASE='81000' bash scripts/eval/run_chunk8_replay_rank.sh 1 2 episode1_failure_chunk8; echo DONE failure; read"

tmux new-window -t "${SESSION}" -n success \
  "cd '${REPO_ROOT}' && BASE_OUT_DIR='${BASE_OUT_DIR}' NUM_REPLAYS='10' REPLAY_HORIZON_STEPS='40' REPLAY_SEED_BASE='82000' bash scripts/eval/run_chunk8_replay_rank.sh 3 4 episode3_success_chunk8; echo DONE success; read"

echo "Started tmux session: ${SESSION}"
echo "  failure replay -> GPU 2"
echo "  success replay -> GPU 4"
echo "Output base: ${REPO_ROOT}/${BASE_OUT_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
