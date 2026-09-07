#!/usr/bin/env bash
# Start full 96-rollout SIMPLER-Bridge rank-fusion evaluation on physical GPUs 4 and 5.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_simpler_bridge_rank_fusion_2gpu}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/rank_fusion_action_hidden_k3_full96_${DATE_TAG}}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing T5 embeddings: ${EMBEDDINGS}" >&2; exit 1; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then echo "tmux session already exists: ${SESSION}" >&2; exit 1; fi
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/result" "${OUT_ROOT}/ranks"
RUNNER="${OUT_ROOT}/run_all.sh"
tee "${RUNNER}" >/dev/null <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
printf '[%s] Starting rank-fusion full Bridge eval on GPUs 4,5\\n' "\$(date -Is)"
pids=()
for rank in 0 1; do
  gpu=\$((rank + 4))
  OUT_ROOT="${OUT_ROOT}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
    bash scripts/eval/run_mimic_simpler_bridge_rank_fusion_rank.sh "\${rank}" "\${gpu}" > "${OUT_ROOT}/logs/rank\${rank}_gpu\${gpu}.log" 2>&1 &
  pids[\${rank}]=\$!
done
status=0
for rank in 0 1; do if ! wait "\${pids[\${rank}]}"; then echo "rank \${rank} failed" >&2; status=1; fi; done
if (( status != 0 )); then exit "\${status}"; fi
"${PYTHON}" scripts/eval/collect_simpler_bridge_sr.py --out-root "${OUT_ROOT}"
printf '[%s] Complete\\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
