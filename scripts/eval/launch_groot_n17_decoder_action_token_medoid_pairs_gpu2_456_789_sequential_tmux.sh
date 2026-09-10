#!/usr/bin/env bash
# Run two fixed-candidate decoder action-token medoid GR00T evaluations on GPU 2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
GPU="${GPU:-2}"
SESSION="${SESSION:-groot_n17_decoder_action_token_medoid_pairs_gpu2_456_789_sequential}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/groot_n17_decoder_action_token_medoid_pairs_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
"${CLIENT_PY}" -c 'import msgpack_numpy' || { echo "Missing msgpack-numpy in SIMPLER environment." >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
for label in seeds_4_5_6 seeds_7_8_9; do
  mkdir -p "${OUT_ROOT}/${label}/logs" "${OUT_ROOT}/${label}/results"
done

RUNNER="${OUT_ROOT}/run_all.sh"
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
server_a=""
server_b=""
cleanup() {
  for pid in "\${server_a}" "\${server_b}"; do
    [[ -n "\${pid}" ]] && kill "\${pid}" 2>/dev/null || true
  done
  for pid in "\${server_a}" "\${server_b}"; do
    [[ -n "\${pid}" ]] && wait "\${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
start_server() {
  local seeds="\$1" port="\$2" output="\$3"
  CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_decoder_action_token_medoid_server.py \\
    --model-path "${MODEL_PATH}" --port "\${port}" --candidate-seeds "\${seeds}" > "\${output}/logs/server_gpu${GPU}.log" 2>&1 &
  printf '%s' "\$!"
}
server_a=\$(start_server 4,5,6 5582 "${OUT_ROOT}/seeds_4_5_6")
server_b=\$(start_server 7,8,9 5583 "${OUT_ROOT}/seeds_7_8_9")
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
wait_ready "\${server_a}" 'seeds=(2, 997, 996)' "${OUT_ROOT}/seeds_4_5_6/logs/server_gpu${GPU}.log"
wait_ready "\${server_b}" 'seeds=(3, 995, 994)' "${OUT_ROOT}/seeds_7_8_9/logs/server_gpu${GPU}.log"
run_eval() {
  local output="\$1" port="\$2" seeds="\$3"
  for task in "\${TASKS[@]}"; do
    name=\${task##*/}
    SAPIEN_RENDER_CUDA_ORDINAL=0 CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" scripts/eval/groot_n17_simpler_client.py \\
      --env-name "\${task}" --host 127.0.0.1 --port "\${port}" \\
      --output "\${output}/results/\${name}.json" --episodes 24 --environment-seed 0 \\
      --max-episode-steps 300 --execution-horizon 4 > "\${output}/logs/\${name}.log" 2>&1
  done
  "${CLIENT_PY}" scripts/eval/collect_groot_n17_simpler_bridge_run.py \\
    --out-root "\${output}" --planner decoder_action_token_medoid --candidate-seeds "\${seeds}" \\
    > "\${output}/logs/summary.log" 2>&1
}
run_eval "${OUT_ROOT}/seeds_4_5_6" 5582 4,5,6 &
eval_a=\$!
run_eval "${OUT_ROOT}/seeds_7_8_9" 5583 7,8,9 &
eval_b=\$!
wait "\${eval_a}"
wait "\${eval_b}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
