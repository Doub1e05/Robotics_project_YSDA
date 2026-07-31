#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

GPU="${GPU:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
SEED_BASE="${SEED_BASE:-71001}"
DATE_TAG="${DATE_TAG:-$(TZ=Asia/Almaty date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial/consensus_medoid/task0_init9_${NUM_ROLLOUTS}rollouts_gpu${GPU}_${DATE_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite_task0_init9_consensus_medoid_gpu${GPU}.log"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:/home/motovilovil/Robotics/robotics_project/LIBERO-PRO"

echo "=== LIBERO-Spatial task 0 / init 9 · Consensus-medoid ==="
echo "GPU=${GPU} rollouts=${NUM_ROLLOUTS} seeds=${SEED_BASE}..$((SEED_BASE + NUM_ROLLOUTS - 1))"
echo "Output: ${OUT_DIR}"

"${PYTHON}" "${REPO_ROOT}/scripts/eval/run_boundary_multi_seed.py" \
  --task-id 0 \
  --episode-idx 9 \
  --task-suite-name libero_spatial \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --seed-base "${SEED_BASE}" \
  --seed-stride 1 \
  --max-control-steps 220 \
  --regen-strategy consensus_medoid \
  --regen-num-candidates 3 \
  --consensus-medoid-horizon 5 \
  --consensus-medoid-temporal-discount 0.9 \
  --consensus-medoid-translation-weight 1.0 \
  --consensus-medoid-rotation-weight 0.5 \
  --consensus-medoid-gripper-weight 0.25 \
  --consensus-medoid-continuity-weight 0.25 \
  --consensus-medoid-smoothness-weight 0.10 \
  --consensus-medoid-gripper-switch-weight 0.10 \
  --diagnostics-mode default \
  --replay-payload-mode none \
  --summary-only \
  --no-save-videos \
  --no-use-cuda-graphs \
  --no-resume \
  --rollout-dir "${OUT_DIR}" \
  --t5-embeddings-path "${T5_EMB}" \
  --progress-label "task0 init9 consensus-medoid ${NUM_ROLLOUTS}x" \
  2>&1 | tee "${LOG_DIR}/eval.log"

echo
echo "=== DONE ==="
echo "Summary: ${OUT_DIR}/metrics/summary.json"
