#!/usr/bin/env bash
# Start the full 96-rollout finetuned Bridge evaluation in a persistent tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_simpler_bridge_ftcosmos_consensus_medoid_only}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_consensus_medoid_only_k3_full96_${DATE_TAG}}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt}"
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing T5 embeddings: ${EMBEDDINGS}" >&2; exit 1; }

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}/logs"
RUNNER="${OUT_ROOT}/run_all.sh"
tee "${RUNNER}" > /dev/null <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
printf '[%s] Starting four GPU ranks\\n' "\$(date -Is)"
for rank in 0 1 2 3; do
  gpu=\$((rank + 4))
  OUT_ROOT="${OUT_ROOT}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
    bash scripts/eval/run_mimic_simpler_bridge_ftcosmos_consensus_medoid_only_rank.sh "\${rank}" "\${gpu}" > "${OUT_ROOT}/logs/rank\${rank}_gpu\${gpu}.log" 2>&1 &
  pids[\${rank}]=\$!
done
status=0
for rank in 0 1 2 3; do
  if ! wait "\${pids[\${rank}]}"; then
    echo "rank \${rank} failed" >&2
    status=1
  fi
done
(( status == 0 ))
model/.venv/bin/python scripts/eval/collect_simpler_bridge_sr.py --out-root "${OUT_ROOT}"
printf '[%s] Complete\\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n "bridge" "bash ${RUNNER}"
echo "Started tmux session: ${SESSION}"
echo "Output: ${OUT_ROOT}"
echo "Attach: tmux attach -t ${SESSION}"
