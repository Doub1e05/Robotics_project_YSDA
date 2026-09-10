#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${REPO_ROOT}/model/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_spatial}"
GPU="${GPU:-2}"
PORT="${PORT:-5602}"
SESSION="${SESSION:-groot_n17_libero_spatial_decoder_gpu2_seq}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/eval_outputs/libero_spatial/groot_n17_decoder_action_token_medoid_gpu2_${DATE_TAG}}"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
ss -ltn | grep -q ":${PORT} " && { echo "port is busy: ${PORT}" >&2; exit 1; }
mkdir -p "${OUT_BASE}/logs"

RUNNER="${OUT_BASE}/run_all.sh"
cp "${BASH_SOURCE[0]}" "${OUT_BASE}/launcher_snapshot.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_BASE}/logs/orchestrator.log") 2>&1
export PYTHONPATH="${GROOT_ROOT}:${GROOT_ROOT}/external_dependencies/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"

server_pid=""
cleanup() {
  [[ -n "\${server_pid}" ]] && kill "\${server_pid}" 2>/dev/null || true
  [[ -n "\${server_pid}" ]] && wait "\${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_group() {
  local seeds="\$1"
  local tag="\${seeds//,/_}"
  local out_root="${OUT_BASE}/seeds_\${tag}"
  mkdir -p "\${out_root}/logs" "\${out_root}/results"
  CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" scripts/eval/groot_n17_decoder_action_token_medoid_server.py \
    --model-path "${MODEL_PATH}" --embodiment-tag LIBERO_PANDA \
    --port "${PORT}" --candidate-seeds "\${seeds}" \
    > "\${out_root}/logs/server_gpu${GPU}.log" 2>&1 &
  server_pid=\$!
  for attempt in {1..300}; do
    grep -q "GR00T decoder-action-token-medoid ready" "\${out_root}/logs/server_gpu${GPU}.log" && break
    kill -0 "\${server_pid}" 2>/dev/null || { cat "\${out_root}/logs/server_gpu${GPU}.log" >&2; return 1; }
    sleep 2
  done
  grep -q "GR00T decoder-action-token-medoid ready" "\${out_root}/logs/server_gpu${GPU}.log"
  for task_index in {0..9}; do
    task_tag="\$(printf 'task_%02d' "\${task_index}")"
    printf '[%s] seeds=%s task=%s gpu=${GPU}\n' "\$(date -Is)" "\${seeds}" "\${task_index}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" scripts/eval/groot_n17_libero_spatial_client.py \
      --task-index "\${task_index}" --host 127.0.0.1 --port "${PORT}" \
      --output "\${out_root}/results/\${task_tag}.json" --episodes 10 \
      --environment-seed 0 --max-episode-steps 220 --execution-horizon 8 \
      > "\${out_root}/logs/\${task_tag}.log" 2>&1
  done
  "${CLIENT_PY}" scripts/eval/collect_groot_n17_libero_spatial_decoder.py \
    --out-root "\${out_root}" --candidate-seeds "\${seeds}"
  kill "\${server_pid}" 2>/dev/null || true
  wait "\${server_pid}" 2>/dev/null || true
  server_pid=""
}

run_group "2,997,996"
run_group "3,995,994"
printf '[%s] Both candidate groups complete\n' "\$(date -Is)"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n libero_spatial_decoder_seq "bash ${RUNNER}"
printf 'session=%s\noutput=%s\ngroups=2,997,996 then 3,995,994\n' "${SESSION}" "${OUT_BASE}"
