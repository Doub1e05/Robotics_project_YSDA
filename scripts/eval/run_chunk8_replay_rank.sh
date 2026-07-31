#!/usr/bin/env bash
set -euo pipefail

SOURCE_EPISODE="${1:?source_total_episode_idx}"
GPU="${2:?gpu}"
LABEL="${3:?label}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BASE_OUT_DIR="${BASE_OUT_DIR:-eval_outputs/libero_spatial/chunk8_decoder_replay/task0_init9_2sec_gpu24}"
OUT_DIR="${BASE_OUT_DIR}/${LABEL}"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"
SOURCE_METRICS="${SOURCE_METRICS:-${REPO_ROOT}/eval_outputs/libero_spatial/encoder_decoder_replay/task0_init9_20rollouts_6sec_gpu36_v2/metrics/rank0/episode_traces.jsonl}"
SOURCE_RANK_DIR="${SOURCE_RANK_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial/encoder_decoder_replay/task0_init9_20rollouts_6sec_gpu36_v2/rank0}"
NUM_REPLAYS="${NUM_REPLAYS:-10}"
REPLAY_HORIZON_STEPS="${REPLAY_HORIZON_STEPS:-40}"
REPLAY_SEED_BASE="${REPLAY_SEED_BASE:-80000}"

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/.mplconfig"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite.log"
export MPLCONFIGDIR="${REPO_ROOT}/.mplconfig"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/model:${REPO_ROOT}/eval/libero/LIBERO"

exec "$PYTHON" scripts/eval/replay_from_saved_chunk.py \
  --source-metrics-path "${SOURCE_METRICS}" \
  --source-rank-dir "${SOURCE_RANK_DIR}" \
  --out-dir "${REPO_ROOT}/${OUT_DIR}" \
  --total-episode-idx "${SOURCE_EPISODE}" \
  --chunk-id 8 \
  --num-replays "${NUM_REPLAYS}" \
  --replay-horizon-steps "${REPLAY_HORIZON_STEPS}" \
  --replay-seed-base "${REPLAY_SEED_BASE}" \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --episode-idx 9 \
  --gpu-label "gpu${GPU}" \
  --t5-embeddings-path "${T5_EMB}" \
  2>&1 | tee -a "${REPO_ROOT}/${OUT_DIR}/run.log"
