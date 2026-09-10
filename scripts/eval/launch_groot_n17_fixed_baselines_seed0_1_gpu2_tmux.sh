#!/usr/bin/env bash
# Run two reproducible, fixed-action-seed GR00T Bridge baselines concurrently on GPU 2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
GPU="${GPU:-2}"
SESSION="${SESSION:-groot_n17_fixed_baselines_seed0_1_gpu2}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_PARENT="${OUT_PARENT:-${REPO_ROOT}/eval_outputs/simpler_bridge/groot_n17_fixed_baselines_action_seeds_0_1_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
"${CLIENT_PY}" -c 'import msgpack_numpy' || { echo "Missing msgpack-numpy in SIMPLER environment." >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
for seed in 0 1; do
  mkdir -p "${OUT_PARENT}/seed_${seed}/logs" "${OUT_PARENT}/seed_${seed}/results"
done

RUNNER="${OUT_PARENT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${GROOT_ROOT}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
TASKS=(
  simpler_env_widowx/widowx_carrot_on_plate
  simpler_env_widowx/widowx_spoon_on_towel
  simpler_env_widowx/widowx_stack_cube
  simpler_env_widowx/widowx_put_eggplant_in_basket
)
server_0=""
server_1=""
cleanup() {
  for pid in "\${server_0}" "\${server_1}"; do
    [[ -n "\${pid}" ]] && kill "\${pid}" 2>/dev/null || true
  done
  for pid in "\${server_0}" "\${server_1}"; do
    [[ -n "\${pid}" ]] && wait "\${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local seed="\$1" port="\$2" output="\$3"
  CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_fixed_seed_baseline_server.py \\
    --model-path "${MODEL_PATH}" --action-seed "\${seed}" --port "\${port}" > "\${output}/logs/server_gpu${GPU}.log" 2>&1 &
  printf '%s' "\$!"
}
server_0=\$(start_server 0 5576 "${OUT_PARENT}/seed_0")
server_1=\$(start_server 1 5577 "${OUT_PARENT}/seed_1")
wait_ready() {
  local pid="\$1" marker="\$2" log="\$3"
  for attempt in {1..300}; do
    grep -q "\${marker}" "\${log}" && return 0
    kill -0 "\${pid}" 2>/dev/null || { cat "\${log}" >&2; return 1; }
    sleep 2
  done
  echo "Timed out waiting for policy server: \${log}" >&2
  return 1
}
wait_ready "\${server_0}" 'action_seed=0' "${OUT_PARENT}/seed_0/logs/server_gpu${GPU}.log"
wait_ready "\${server_1}" 'action_seed=1' "${OUT_PARENT}/seed_1/logs/server_gpu${GPU}.log"

run_seed() {
  local seed="\$1" port="\$2" output="\$3"
  for task in "\${TASKS[@]}"; do
    name=\${task##*/}
    SAPIEN_RENDER_CUDA_ORDINAL=0 CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" scripts/eval/groot_n17_simpler_client.py \\
      --env-name "\${task}" --host 127.0.0.1 --port "\${port}" \\
      --output "\${output}/results/\${name}.json" --episodes 24 --environment-seed 0 \\
      --max-episode-steps 300 --execution-horizon 4 > "\${output}/logs/\${name}.log" 2>&1
  done
  "${CLIENT_PY}" scripts/eval/collect_groot_n17_simpler_bridge_run.py \\
    --out-root "\${output}" --planner baseline_fixed_action_seed --candidate-seeds "\${seed}" \\
    > "\${output}/logs/summary.log" 2>&1
}
run_seed 0 5576 "${OUT_PARENT}/seed_0" &
eval_0=\$!
run_seed 1 5577 "${OUT_PARENT}/seed_1" &
eval_1=\$!
wait "\${eval_0}"
wait "\${eval_1}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_PARENT}"
