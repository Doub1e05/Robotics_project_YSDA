#!/usr/bin/env bash
# Strict sequential pair: GR00T baseline, then consensus action medoid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ROOT="${REPO_ROOT}/external/Isaac-GR00T"
PLUS_ROOT="${REPO_ROOT}/LIBERO-plus"
SERVER_PY="${GROOT_ROOT}/.venv/bin/python"
CLIENT_PY="${REPO_ROOT}/model/.venv/bin/python"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_spatial}"
GPU="${GPU:-1}"
BASELINE_SEED="${BASELINE_SEED:-1}"
CANDIDATE_SEEDS="${CANDIDATE_SEEDS:-1,999,998}"
PORT="${PORT:-5597}"
NUM_TASKS="${NUM_TASKS:-100}"
SESSION="${SESSION:-groot_n17_libero_plus_spatial_pair_gpu1}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/libero_plus/groot_n17_spatial_baseline_seed1_then_consensus_1_999_998_gpu1_${DATE_TAG}}"
CLASSIFICATION_FILE="${PLUS_ROOT}/libero/libero/benchmark/task_classification.json"

for path in "${SERVER_PY}" "${CLIENT_PY}" "${MODEL_PATH}/model.safetensors.index.json" \
  "${PLUS_ROOT}/libero/libero/__init__.py" "${PLUS_ROOT}/libero/libero/assets" \
  "${CLASSIFICATION_FILE}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session exists: ${SESSION}" >&2
  exit 1
}
ss -ltn | grep -q ":${PORT} " && { echo "port is busy: ${PORT}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs"
LIBERO_CONFIG_PATH="${OUT_ROOT}/libero_config"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${PLUS_ROOT}/libero/libero
bddl_files: ${PLUS_ROOT}/libero/libero/bddl_files
init_states: ${PLUS_ROOT}/libero/libero/init_files
datasets: ${PLUS_ROOT}/libero/datasets
assets: ${PLUS_ROOT}/libero/libero/assets
EOF

RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
export PYTHONPATH="${PLUS_ROOT}:${GROOT_ROOT}:\${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"

server_pid=""
cleanup_server() {
  [[ -n "\${server_pid}" ]] && kill "\${server_pid}" 2>/dev/null || true
  [[ -n "\${server_pid}" ]] && wait "\${server_pid}" 2>/dev/null || true
  server_pid=""
}
trap cleanup_server EXIT INT TERM

run_tasks() {
  local arm="\$1"
  local method="\$2"
  local seeds="\$3"
  mkdir -p "${OUT_ROOT}/\${arm}/logs" "${OUT_ROOT}/\${arm}/results"
  for ((task_index=0; task_index<${NUM_TASKS}; task_index++)); do
    task_tag="\$(printf 'task_%04d' "\${task_index}")"
    printf '[%s] arm=%s task=%s/%s action_seeds=%s gpu=${GPU}\n' \
      "\$(date -Is)" "\${arm}" "\$((task_index + 1))" "${NUM_TASKS}" "\${seeds}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${CLIENT_PY}" \
      scripts/eval/groot_n17_libero_plus_spatial_client.py \
      --task-index "\${task_index}" --host 127.0.0.1 --port "${PORT}" \
      --libero-root "${PLUS_ROOT}" --classification-file "${CLASSIFICATION_FILE}" \
      --environment-seed 0 --max-episode-steps 220 --execution-horizon 8 \
      --output "${OUT_ROOT}/\${arm}/results/\${task_tag}.json" \
      > "${OUT_ROOT}/\${arm}/logs/\${task_tag}.log" 2>&1
    "${CLIENT_PY}" scripts/eval/collect_groot_n17_libero_plus_spatial.py \
      --arm-root "${OUT_ROOT}/\${arm}" --method "\${method}" \
      --action-seeds "\${seeds}" --expected-episodes "${NUM_TASKS}" || true
  done
}

printf '[%s] Starting baseline seed=${BASELINE_SEED}\n' "\$(date -Is)"
CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" \
  scripts/eval/groot_n17_libero_fixed_seed_baseline_server.py \
  --model-path "${MODEL_PATH}" --action-seed "${BASELINE_SEED}" --port "${PORT}" \
  > "${OUT_ROOT}/baseline_server_gpu${GPU}.log" 2>&1 &
server_pid=\$!
for attempt in {1..300}; do
  grep -q "action_seed=${BASELINE_SEED}" "${OUT_ROOT}/baseline_server_gpu${GPU}.log" && break
  kill -0 "\${server_pid}" 2>/dev/null || { cat "${OUT_ROOT}/baseline_server_gpu${GPU}.log" >&2; exit 1; }
  sleep 2
done
grep -q "action_seed=${BASELINE_SEED}" "${OUT_ROOT}/baseline_server_gpu${GPU}.log"
run_tasks baseline baseline "${BASELINE_SEED}"
cleanup_server

printf '[%s] Baseline complete; starting consensus medoid seeds=${CANDIDATE_SEEDS}\n' "\$(date -Is)"
CUDA_VISIBLE_DEVICES="${GPU}" "${SERVER_PY}" \
  scripts/eval/groot_n17_consensus_server.py \
  --model-path "${MODEL_PATH}" --embodiment-tag LIBERO_PANDA \
  --port "${PORT}" --candidate-seeds "${CANDIDATE_SEEDS}" \
  > "${OUT_ROOT}/consensus_server_gpu${GPU}.log" 2>&1 &
server_pid=\$!
for attempt in {1..300}; do
  grep -q "GR00T consensus-only ready" "${OUT_ROOT}/consensus_server_gpu${GPU}.log" && break
  kill -0 "\${server_pid}" 2>/dev/null || { cat "${OUT_ROOT}/consensus_server_gpu${GPU}.log" >&2; exit 1; }
  sleep 2
done
grep -q "GR00T consensus-only ready" "${OUT_ROOT}/consensus_server_gpu${GPU}.log"
run_tasks consensus_medoid consensus_action_medoid "${CANDIDATE_SEEDS}"
cleanup_server

"${CLIENT_PY}" - "${OUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
baseline = json.loads((root / "baseline/summary.json").read_text())
consensus = json.loads((root / "consensus_medoid/summary.json").read_text())
comparison = {
    "benchmark": "LIBERO-Plus Spatial fixed first-100 slice",
    "model": "GR00T-N1.7-LIBERO/libero_spatial",
    "execution": "strict sequential baseline then consensus medoid",
    "baseline": baseline,
    "consensus_medoid": consensus,
    "success_count_delta": consensus["successes"] - baseline["successes"],
    "success_rate_delta": consensus["success_rate"] - baseline["success_rate"],
}
(root / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
PY
printf '[%s] Complete\n' "\$(date -Is)"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n libero_plus_spatial_pair "bash ${RUNNER}"
printf 'session=%s\noutput=%s\nGPU=%s\nbaseline_seed=%s\ncandidate_seeds=%s\n' \
  "${SESSION}" "${OUT_ROOT}" "${GPU}" "${BASELINE_SEED}" "${CANDIDATE_SEEDS}"
