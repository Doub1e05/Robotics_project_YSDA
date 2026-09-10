#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
PRO_CODE="${REPO_ROOT}/external/LIBERO-PRO-code"
PRO_DATA="${REPO_ROOT}/LIBERO-PRO"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${REPO_ROOT}/model/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_spatial}"
GPU="${GPU:-3}"
ACTION_SEED="${ACTION_SEED:-1}"
PORT="${PORT:-5594}"
SESSION="${SESSION:-groot_n17_libero_pro_spatial_seed1_gpu3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/libero_pro/groot_n17_spatial_baseline_seed1_gpu3_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json" \
  "${PRO_CODE}/libero/libero/__init__.py" "${PRO_DATA}/bddl_files/libero_spatial_object"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session exists: ${SESSION}" >&2
  exit 1
}
ss -ltn | grep -q ":${PORT} " && { echo "port is busy: ${PORT}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/results" "${OUT_ROOT}/videos"
LIBERO_CONFIG_PATH="${OUT_ROOT}/libero_config"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${PRO_CODE}/libero/libero
bddl_files: ${PRO_CODE}/libero/libero/bddl_files
init_states: ${PRO_CODE}/libero/libero/init_files
datasets: ${PRO_CODE}/libero/datasets
assets: ${PRO_CODE}/libero/libero/assets
EOF

RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
export PYTHONPATH="${PRO_CODE}:${GROOT_ROOT}:\${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"

server_pid=""
cleanup() {
  [[ -n "\${server_pid}" ]] && kill "\${server_pid}" 2>/dev/null || true
  [[ -n "\${server_pid}" ]] && wait "\${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" \
  scripts/eval/groot_n17_libero_fixed_seed_baseline_server.py \
  --model-path "${MODEL_PATH}" --action-seed "${ACTION_SEED}" --port "${PORT}" \
  > "${OUT_ROOT}/logs/server_gpu${GPU}.log" 2>&1 &
server_pid=\$!
for attempt in {1..300}; do
  grep -q "action_seed=${ACTION_SEED}" "${OUT_ROOT}/logs/server_gpu${GPU}.log" && break
  kill -0 "\${server_pid}" 2>/dev/null || {
    cat "${OUT_ROOT}/logs/server_gpu${GPU}.log" >&2
    exit 1
  }
  sleep 2
done
grep -q "action_seed=${ACTION_SEED}" "${OUT_ROOT}/logs/server_gpu${GPU}.log"

for split in lan object swap task; do
  for task_index in {0..9}; do
    task_tag="\$(printf '%s_%02d' "\${split}" "\${task_index}")"
    printf '[%s] split=%s task=%s action_seed=${ACTION_SEED} gpu=${GPU}\n' \
      "\$(date -Is)" "\${split}" "\${task_index}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" \
      scripts/eval/groot_n17_libero_pro_spatial_client.py \
      --split "\${split}" --task-index "\${task_index}" \
      --host 127.0.0.1 --port "${PORT}" \
      --data-root "${PRO_DATA}" --episodes 10 --environment-seed 0 \
      --max-episode-steps 220 --execution-horizon 8 \
      --output "${OUT_ROOT}/results/\${task_tag}.json" \
      --video-dir "${OUT_ROOT}/videos/\${task_tag}" \
      > "${OUT_ROOT}/logs/\${task_tag}.log" 2>&1
    "${CLIENT_PY}" scripts/eval/collect_groot_n17_libero_pro_spatial.py \
      --out-root "${OUT_ROOT}" --action-seed "${ACTION_SEED}" || true
  done
done
"${CLIENT_PY}" scripts/eval/collect_groot_n17_libero_pro_spatial.py \
  --out-root "${OUT_ROOT}" --action-seed "${ACTION_SEED}"
printf '[%s] Complete\n' "\$(date -Is)"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n libero_pro_spatial "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
