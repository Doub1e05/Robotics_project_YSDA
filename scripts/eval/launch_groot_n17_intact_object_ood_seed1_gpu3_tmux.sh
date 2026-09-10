#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${GROOT_ROOT}/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/groot_n17_simplerenv_bridge}"
GPU="${GPU:-3}"
ACTION_SEED="${ACTION_SEED:-1}"
PORT="${PORT:-5591}"
SESSION="${SESSION:-groot_n17_intact_object_ood_seed1_gpu3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/groot_n17_object_ood_8tasks_seed1_gpu3_${DATE_TAG}}"
INTACT_SIMPLER="${REPO_ROOT}/eval/int-act/third_party/SimplerEnv"
INTACT_MS2="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
ss -ltn | grep -q ":${PORT} " && { echo "port is busy: ${PORT}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/results" "${OUT_ROOT}/videos"

RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
export PYTHONPATH="${GROOT_ROOT}:${INTACT_SIMPLER}:${INTACT_MS2}:\${PYTHONPATH:-}"
export MS2_REAL2SIM_ASSET_DIR="${INTACT_MS2}/data"
export TOKENIZERS_PARALLELISM=false
export SAPIEN_RENDER_CUDA_ORDINAL=0
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select

server_pid=""
cleanup() {
  [[ -n "\${server_pid}" ]] && kill "\${server_pid}" 2>/dev/null || true
  [[ -n "\${server_pid}" ]] && wait "\${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_fixed_seed_baseline_server.py \
  --model-path "${MODEL_PATH}" --action-seed "${ACTION_SEED}" --port "${PORT}" \
  > "${OUT_ROOT}/logs/server_gpu${GPU}.log" 2>&1 &
server_pid=\$!
for attempt in {1..300}; do
  grep -q "action_seed=${ACTION_SEED}" "${OUT_ROOT}/logs/server_gpu${GPU}.log" && break
  kill -0 "\${server_pid}" 2>/dev/null || { cat "${OUT_ROOT}/logs/server_gpu${GPU}.log" >&2; exit 1; }
  sleep 2
done
grep -q "action_seed=${ACTION_SEED}" "${OUT_ROOT}/logs/server_gpu${GPU}.log"

TASKS=(
  widowx_cube_on_plate_clean
  widowx_small_plate_on_green_cube_clean
  widowx_carrot_on_sponge_clean
  widowx_eggplant_on_sponge_clean
  widowx_coke_can_on_plate_clean
  widowx_pepsi_on_plate_clean
  widowx_carrot_on_keyboard_clean
  widowx_coke_can_on_keyboard_clean
)
for task in "\${TASKS[@]}"; do
  printf '[%s] task=%s action_seed=${ACTION_SEED} gpu=${GPU}\n' "\$(date -Is)" "\${task}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" scripts/eval/groot_n17_intact_object_ood_client.py \
    --task-key "\${task}" --host 127.0.0.1 --port "${PORT}" \
    --output "${OUT_ROOT}/results/\${task}.json" --episodes 24 --environment-seed 0 \
    --max-episode-steps 300 --execution-horizon 4 --video-dir "${OUT_ROOT}/videos/\${task}" \
    > "${OUT_ROOT}/logs/\${task}.log" 2>&1
  "${CLIENT_PY}" scripts/eval/collect_groot_n17_intact_object_ood.py \
    --out-root "${OUT_ROOT}" --action-seed "${ACTION_SEED}" || true
done
"${CLIENT_PY}" scripts/eval/collect_groot_n17_intact_object_ood.py \
  --out-root "${OUT_ROOT}" --action-seed "${ACTION_SEED}"
printf '[%s] Complete\n' "\$(date -Is)"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
