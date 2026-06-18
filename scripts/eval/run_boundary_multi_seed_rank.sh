#!/usr/bin/env bash
# Usage: <rank> <cuda_device> [extra args...]
set -euo pipefail

RANK="${1:?rank}"
GPU="${2:?gpu}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

OUT_DIR="${OUT_DIR:-eval_outputs/libero_spatial/boundary_runs/task0_init9_baseline_40rollouts}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-40}"
WORLD_SIZE="${WORLD_SIZE:-3}"

mkdir -p "${OUT_DIR}/metrics/rank${RANK}" "${OUT_DIR}/rank${RANK}/videos"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/libero/LIBERO"

exec "$PYTHON" scripts/eval/run_boundary_multi_seed.py \
  --task-id 0 \
  --episode-idx 9 \
  --task-suite-name libero_spatial \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --seed-base 50000 \
  --seed-stride 97 \
  --max-control-steps 220 \
  --regen-strategy none \
  --eval-rank "${RANK}" \
  --eval-world-size "${WORLD_SIZE}" \
  --rollout-dir "${REPO_ROOT}/${OUT_DIR}" \
  --metrics-dir "${REPO_ROOT}/${OUT_DIR}/metrics" \
  --t5-embeddings-path "${T5_EMB}" \
  --no-use-cuda-graphs \
  --save-videos \
  --max-videos 0 \
  "$@" \
  2>&1 | tee -a "${REPO_ROOT}/${OUT_DIR}/run_rank${RANK}.log"
