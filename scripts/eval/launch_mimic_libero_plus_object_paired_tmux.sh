#!/usr/bin/env bash
# Launch the sequential paired baseline -> consensus-only validation in tmux.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU="${GPU:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-1}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5}"
SEED="${SEED:-0}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-libero_object_pair_validation}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/libero_object_plus/paired_${MAX_EVAL_EPISODES}ep_seed${SEED}_${DATE_TAG}}"
RUNNER="${REPO_ROOT}/scripts/eval/run_mimic_libero_plus_object_paired.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"
tmux new-session -d -s "${SESSION}" -n "gpu${GPU}" \
  "cd  && GPU= NUM_TRIALS_PER_TASK= MAX_EVAL_EPISODES= MAX_CONTROL_STEPS= SEED= OUT_ROOT= bash  2>&1 | tee /launcher.log; code=\${PIPESTATUS[0]}; echo \"=== PROCESS EXITED status=\${code} ===\"; exec bash"
echo "Started tmux session: ${SESSION}"
echo "Output: ${OUT_ROOT}"
echo "Attach: tmux attach -t ${SESSION}"
