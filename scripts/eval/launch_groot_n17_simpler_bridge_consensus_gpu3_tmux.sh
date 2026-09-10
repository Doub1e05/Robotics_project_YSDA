#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
GPU="${GPU:-3}"
SESSION="${SESSION:-groot_n17_simpler_bridge_consensus_gpu3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/groot_n17_bridge_consensus_only_seeds_1_999_998_${DATE_TAG}}"
for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/results" "${OUT_ROOT}/videos"
RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${GROOT_ROOT}:\${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_consensus_server.py \\
  --model-path "${MODEL_PATH}" --port 5573 --candidate-seeds 1,999,998 \\
  > "${OUT_ROOT}/logs/server_gpu${GPU}.log" 2>&1 &
server_pid=\$!
cleanup() { kill "\${server_pid}" 2>/dev/null || true; wait "\${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in {1..180}; do
  grep -q 'Server ready' "${OUT_ROOT}/logs/server_gpu${GPU}.log" && break
  kill -0 "\${server_pid}" 2>/dev/null || { cat "${OUT_ROOT}/logs/server_gpu${GPU}.log" >&2; exit 1; }
  sleep 2
done
grep -q 'Server ready' "${OUT_ROOT}/logs/server_gpu${GPU}.log"
run_task() {
  local task="\$1"
  SAPIEN_RENDER_CUDA_ORDINAL=0 CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" \\
    scripts/eval/groot_n17_simpler_client.py --env-name "\${task}" --host 127.0.0.1 --port 5573 \\
    --output "${OUT_ROOT}/results/\${task##*/}.json" --episodes 24 --environment-seed 0 \\
    --max-episode-steps 300 --execution-horizon 4 --video-dir "${OUT_ROOT}/videos/\${task##*/}" \\
    > "${OUT_ROOT}/logs/\${task##*/}.log" 2>&1
}
run_task simpler_env_widowx/widowx_carrot_on_plate
run_task simpler_env_widowx/widowx_spoon_on_towel
run_task simpler_env_widowx/widowx_stack_cube
run_task simpler_env_widowx/widowx_put_eggplant_in_basket
"${CLIENT_PY}" scripts/eval/collect_groot_n17_simpler_bridge.py --out-root "${OUT_ROOT}" > "${OUT_ROOT}/logs/summary.log" 2>&1
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
