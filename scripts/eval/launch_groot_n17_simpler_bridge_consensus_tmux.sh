#!/usr/bin/env bash
# Evaluate GR00T N1.7 Bridge consensus-only on 4 tasks x 24 fixed environment seeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
SESSION="${SESSION:-groot_n17_simpler_bridge_consensus}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/groot_n17_bridge_consensus_only_seeds_1_994_995_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/results"
RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${GROOT_ROOT}:\${PYTHONPATH:-}"

run_rank() {
  local gpu="\$1" port="\$2" first_task="\$3" second_task="\$4"
  CUDA_VISIBLE_DEVICES="\${gpu}" "${SERVER_PY}" scripts/eval/groot_n17_consensus_server.py \\
    --model-path "${MODEL_PATH}" --port "\${port}" --candidate-seeds 1,994,995 \\
    > "${OUT_ROOT}/logs/server_gpu\${gpu}.log" 2>&1 &
  local server_pid=\$!
  trap 'kill "\${server_pid}" 2>/dev/null || true' RETURN
  for attempt in {1..90}; do
    if grep -q 'Server ready' "${OUT_ROOT}/logs/server_gpu\${gpu}.log" 2>/dev/null; then break; fi
    kill -0 "\${server_pid}" 2>/dev/null || { cat "${OUT_ROOT}/logs/server_gpu\${gpu}.log" >&2; return 1; }
    sleep 2
  done
  grep -q 'Server ready' "${OUT_ROOT}/logs/server_gpu\${gpu}.log"
  for task in "\${first_task}" "\${second_task}"; do
    SAPIEN_RENDER_CUDA_ORDINAL=0 "${CLIENT_PY}" scripts/eval/groot_n17_simpler_client.py \\
      --env-name "\${task}" --host 127.0.0.1 --port "\${port}" \\
      --output "${OUT_ROOT}/results/\${task##*/}.json" \\
      > "${OUT_ROOT}/logs/\${task##*/}.log" 2>&1
  done
  kill "\${server_pid}" 2>/dev/null || true
  wait "\${server_pid}" 2>/dev/null || true
  trap - RETURN
}

run_rank 2 5555 simpler_env_widowx/widowx_carrot_on_plate simpler_env_widowx/widowx_spoon_on_towel &
pid2=\$!
run_rank 3 5556 simpler_env_widowx/widowx_stack_cube simpler_env_widowx/widowx_put_eggplant_in_basket &
pid3=\$!
wait "\${pid2}"
wait "\${pid3}"
"${CLIENT_PY}" scripts/eval/collect_groot_n17_simpler_bridge.py --out-root "${OUT_ROOT}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
