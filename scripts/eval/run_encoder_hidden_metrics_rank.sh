#!/usr/bin/env bash
# Usage: run_encoder_hidden_metrics_rank.sh <eval_rank> <cuda_device>
set -euo pipefail

RANK="${1:?rank}"
GPU="${2:?gpu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/encoder_hidden/task0_init9_10rollouts_6sec}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
WORLD_SIZE="${WORLD_SIZE:-2}"
SEED_BASE="${SEED_BASE:-50000}"
SEED_STRIDE="${SEED_STRIDE:-97}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-120}"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

mkdir -p "${OUT_DIR}/metrics/rank${RANK}" "${OUT_DIR}/rank${RANK}/videos" "${REPO_ROOT}/.mplconfig"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite.log"
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/model:${REPO_ROOT}/eval/libero/LIBERO"

exec "$PYTHON" scripts/eval/run_boundary_multi_seed.py \
  --task-id 0 \
  --episode-idx 9 \
  --task-suite-name libero_spatial \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --seed-base "${SEED_BASE}" \
  --seed-stride "${SEED_STRIDE}" \
  --max-control-steps "${MAX_CONTROL_STEPS}" \
  --regen-strategy none \
  --eval-rank "${RANK}" \
  --eval-world-size "${WORLD_SIZE}" \
  --rollout-dir "${REPO_ROOT}/${OUT_DIR}" \
  --metrics-dir "${REPO_ROOT}/${OUT_DIR}/metrics" \
  --t5-embeddings-path "${T5_EMB}" \
  --no-use-cuda-graphs \
  --no-save-videos \
  --diagnostics-mode encoder_hidden \
  2>&1 | tee -a "${REPO_ROOT}/${OUT_DIR}/run_rank${RANK}.log"
