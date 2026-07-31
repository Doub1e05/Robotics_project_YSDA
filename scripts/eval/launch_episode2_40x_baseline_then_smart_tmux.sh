#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_ep2_40x_baseline_smart}"
RUNNER="${REPO_ROOT}/scripts/eval/run_episode2_baseline_then_smart.sh"
GPU="${GPU:-1}"

chmod +x "${RUNNER}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists."
  echo "Attach: tmux attach -t ${SESSION}"
  exit 1
fi

LOG="${REPO_ROOT}/eval_outputs/libero_spatial_pro/episode2_ramekin_40x_gpu${GPU}_$(date +%Y%m%d)/logs/tmux_launch.log"
mkdir -p "$(dirname "${LOG}")"

tmux new-session -d -s "${SESSION}" -n "eval" \
  "GPU=${GPU} bash '${RUNNER}' 2>&1 | tee -a '${LOG}'; echo; echo '=== ALL DONE ==='; bash"

echo "Started tmux session: ${SESSION} (GPU ${GPU})"
echo "Output: eval_outputs/libero_spatial_pro/episode2_ramekin_40x_gpu${GPU}_$(date +%Y%m%d)/"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    ${LOG}"
