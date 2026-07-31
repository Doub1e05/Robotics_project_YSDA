#!/usr/bin/env bash
# Episode 2 (task 1, init 0): baseline then smart-planning, same seeds.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

GPU="${GPU:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-40}"
SEED_BASE="${SEED_BASE:-10000}"
SEED_STRIDE="${SEED_STRIDE:-1}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-220}"
TASK_ID="${TASK_ID:-1}"
EPISODE_IDX="${EPISODE_IDX:-0}"
TASK_SUITE="${TASK_SUITE:-libero_spatial_object}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

ROOT_OUT="${ROOT_OUT:-${REPO_ROOT}/eval_outputs/libero_spatial_pro/episode2_ramekin_40x_gpu${GPU}_${DATE_TAG}}"
BASELINE_DIR="${ROOT_OUT}/baseline"
SMART_DIR="${ROOT_OUT}/linear_combo_v1"
LOG_DIR="${ROOT_OUT}/logs"
mkdir -p "${BASELINE_DIR}" "${SMART_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite_episode2_40x_gpu${GPU}.log"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:/home/motovilovil/Robotics/robotics_project/LIBERO-PRO"

RUNNER="${REPO_ROOT}/scripts/eval/run_boundary_multi_seed.py"
COMMON_ARGS=(
  --task-id "${TASK_ID}"
  --episode-idx "${EPISODE_IDX}"
  --task-suite-name "${TASK_SUITE}"
  --num-rollouts "${NUM_ROLLOUTS}"
  --seed-base "${SEED_BASE}"
  --seed-stride "${SEED_STRIDE}"
  --max-control-steps "${MAX_CONTROL_STEPS}"
  --no-save-videos
  --no-use-cuda-graphs
  --no-resume
  --t5-embeddings-path "${T5_EMB}"
)

echo "=== Episode 2 eval: ${NUM_ROLLOUTS} baseline + ${NUM_ROLLOUTS} smart planning ==="
echo "GPU=${GPU} suite=${TASK_SUITE} task=${TASK_ID} init=${EPISODE_IDX}"
echo "Output root: ${ROOT_OUT}"
echo "Seeds: ${SEED_BASE}..$((SEED_BASE + SEED_STRIDE * (NUM_ROLLOUTS - 1))) stride=${SEED_STRIDE}"
echo

echo "[1/2] Baseline (${NUM_ROLLOUTS} rollouts) -> ${BASELINE_DIR}"
"${PYTHON}" "${RUNNER}" \
  "${COMMON_ARGS[@]}" \
  --regen-strategy none \
  --diagnostics-mode default \
  --rollout-dir "${BASELINE_DIR}" \
  --progress-label "ep2 baseline ${NUM_ROLLOUTS}x" \
  2>&1 | tee "${LOG_DIR}/baseline.log"

echo
echo "[2/2] Smart planning decoder_linear_combo_v1 (${NUM_ROLLOUTS} rollouts) -> ${SMART_DIR}"
"${PYTHON}" "${RUNNER}" \
  "${COMMON_ARGS[@]}" \
  --regen-strategy decoder_metric_select \
  --regen-num-candidates 3 \
  --decoder-metric-select-layer 23 \
  --decoder-metric-select-name decoder_linear_combo_v1 \
  --decoder-metric-select-action-subset full_chunk \
  --decoder-metric-select-reduce max \
  --diagnostics-mode decoder_hidden \
  --decoder-capture-block-indices 23 \
  --rollout-dir "${SMART_DIR}" \
  --progress-label "ep2 linear_combo_v1 ${NUM_ROLLOUTS}x" \
  2>&1 | tee "${LOG_DIR}/linear_combo_v1.log"

echo
echo "=== Writing combined summary ==="
"${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_OUT}")
out = {
    "task_suite_name": "${TASK_SUITE}",
    "task_id": int("${TASK_ID}"),
    "episode_idx": int("${EPISODE_IDX}"),
    "num_rollouts": int("${NUM_ROLLOUTS}"),
    "seed_base": int("${SEED_BASE}"),
    "seed_stride": int("${SEED_STRIDE}"),
    "max_control_steps": int("${MAX_CONTROL_STEPS}"),
    "gpu": int("${GPU}"),
    "tactics": {},
}
for name, subdir in [("baseline", "baseline"), ("linear_combo_v1", "linear_combo_v1")]:
    summary_path = root / subdir / "metrics" / "summary.json"
    if summary_path.is_file():
        out["tactics"][name] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        out["tactics"][name] = {"error": f"missing {summary_path}"}
combined = root / "combined_summary.json"
combined.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
print(f"Wrote {combined}")
PY

echo
echo "Done. Results:"
echo "  Baseline:  ${BASELINE_DIR}/metrics/"
echo "  Smart:     ${SMART_DIR}/metrics/"
echo "  Combined:  ${ROOT_OUT}/combined_summary.json"
