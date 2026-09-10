#!/usr/bin/env bash
# Run baseline and fixed-seed action-medoid GR00T Bridge evaluations concurrently on one GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
GPU="${GPU:-3}"
SESSION="${SESSION:-groot_n17_baseline_and_consensus_gpu3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_PARENT="${OUT_PARENT:-${REPO_ROOT}/eval_outputs/simpler_bridge/groot_n17_baseline_and_consensus_seeds_2_997_996_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
"${CLIENT_PY}" -c 'import msgpack_numpy' || { echo "Missing msgpack-numpy in SIMPLER environment." >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_PARENT}/baseline/logs" "${OUT_PARENT}/baseline/results" "${OUT_PARENT}/baseline/videos"
mkdir -p "${OUT_PARENT}/consensus_medoid/logs" "${OUT_PARENT}/consensus_medoid/results" "${OUT_PARENT}/consensus_medoid/videos"

RUNNER="${OUT_PARENT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${GROOT_ROOT}:\${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
BASE_OUT="${OUT_PARENT}/baseline"
CONSENSUS_OUT="${OUT_PARENT}/consensus_medoid"
TASKS=(
  simpler_env_widowx/widowx_carrot_on_plate
  simpler_env_widowx/widowx_spoon_on_towel
  simpler_env_widowx/widowx_stack_cube
  simpler_env_widowx/widowx_put_eggplant_in_basket
)

baseline_server=""
consensus_server=""
cleanup() {
  local status=0
  for pid in "\${baseline_server}" "\${consensus_server}"; do
    [[ -n "\${pid}" ]] || continue
    kill "\${pid}" 2>/dev/null || true
  done
  for pid in "\${baseline_server}" "\${consensus_server}"; do
    [[ -n "\${pid}" ]] || continue
    wait "\${pid}" 2>/dev/null || status=1
  done
  exit "\${status}"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_baseline_server.py \\
  --model-path "${MODEL_PATH}" --port 5574 > "\${BASE_OUT}/logs/server_gpu${GPU}.log" 2>&1 &
baseline_server=\$!
CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_consensus_server.py \\
  --model-path "${MODEL_PATH}" --port 5575 --candidate-seeds 2,997,996 > "\${CONSENSUS_OUT}/logs/server_gpu${GPU}.log" 2>&1 &
consensus_server=\$!

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
wait_ready "\${baseline_server}" 'GR00T baseline ready' "\${BASE_OUT}/logs/server_gpu${GPU}.log"
wait_ready "\${consensus_server}" 'GR00T consensus-only ready' "\${CONSENSUS_OUT}/logs/server_gpu${GPU}.log"

run_eval() {
  local output_root="\$1" port="\$2" planner="\$3" seeds="\$4"
  for task in "\${TASKS[@]}"; do
    name=\${task##*/}
    SAPIEN_RENDER_CUDA_ORDINAL=0 CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" scripts/eval/groot_n17_simpler_client.py \\
      --env-name "\${task}" --host 127.0.0.1 --port "\${port}" \\
      --output "\${output_root}/results/\${name}.json" --episodes 24 --environment-seed 0 \\
      --max-episode-steps 300 --execution-horizon 4 --video-dir "\${output_root}/videos/\${name}" \\
      > "\${output_root}/logs/\${name}.log" 2>&1
  done
  args=(--out-root "\${output_root}" --planner "\${planner}")
  [[ -n "\${seeds}" ]] && args+=(--candidate-seeds "\${seeds}")
  "${CLIENT_PY}" scripts/eval/collect_groot_n17_simpler_bridge_run.py "\${args[@]}" > "\${output_root}/logs/summary.log" 2>&1
}

run_eval "\${BASE_OUT}" 5574 baseline "" &
baseline_eval=\$!
run_eval "\${CONSENSUS_OUT}" 5575 consensus_only_action_medoid 2,997,996 &
consensus_eval=\$!
wait "\${baseline_eval}"
wait "\${consensus_eval}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_PARENT}"
