#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_ep2_consensus_medoid_50}"
GPU="${GPU:-1}"
DATE_TAG="${DATE_TAG:-$(TZ=Asia/Almaty date +%Y%m%d)}"
RUNNER="${REPO_ROOT}/scripts/eval/run_episode2_consensus_medoid_50x.sh"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial_pro/episode2_ramekin_consensus_medoid_50x_gpu${GPU}_${DATE_TAG}}"
TMUX_LOG="${OUT_DIR}/logs/tmux.log"

chmod +x "${RUNNER}"
mkdir -p "$(dirname "${TMUX_LOG}")"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists."
  echo "Attach: tmux attach -t ${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n "gpu${GPU}" \
  "GPU='${GPU}' DATE_TAG='${DATE_TAG}' OUT_DIR='${OUT_DIR}' bash '${RUNNER}' 2>&1 | tee -a '${TMUX_LOG}'; status=\${PIPESTATUS[0]}; echo; echo \"=== PROCESS EXITED status=\${status} ===\"; bash"

echo "Started tmux session: ${SESSION}"
echo "GPU:     ${GPU}"
echo "Output:  ${OUT_DIR}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Log:     ${TMUX_LOG}"
