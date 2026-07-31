#!/usr/bin/env bash
# Full LIBERO-Spatial benchmark: 10 tasks × 10 inits = 100 episodes, consensus-only planning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

GPU="${GPU:-1}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-100}"
DATE_TAG="${DATE_TAG:-$(TZ=Asia/Almaty date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial/consensus_medoid/consensus_only_100ep_gpu${GPU}_${DATE_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite_consensus_only_100ep_gpu${GPU}.log"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:/home/motovilovil/Robotics/robotics_project/LIBERO-PRO"

echo "=== LIBERO-Spatial full benchmark · Consensus-only (cost = C[i]) ==="
echo "GPU=${GPU} episodes=${MAX_EVAL_EPISODES} (${NUM_TRIALS_PER_TASK} inits × 10 tasks)"
echo "λ_c=λ_s=λ_g=0"
echo "Output: ${OUT_DIR}"

cd "${REPO_ROOT}"

"${PYTHON}" eval/libero/run.py \
  --vam-experiment-name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
  --vam-video-model-path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt" \
  --vam-action-model-path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt" \
  --vam-dataset-statistics-path "${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_spatial_one.json" \
  --vam-img-horizon 5 \
  --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 0 \
  --vam-num-execute-actions 5 \
  --task-suite-name libero_spatial \
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --max-eval-episodes "${MAX_EVAL_EPISODES}" \
  --max-control-steps 220 \
  --seed 0 \
  --t5-embeddings-path "${T5_EMB}" \
  --regen-strategy consensus_medoid \
  --regen-num-candidates 3 \
  --consensus-medoid-horizon 5 \
  --consensus-medoid-temporal-discount 0.9 \
  --consensus-medoid-translation-weight 1.0 \
  --consensus-medoid-rotation-weight 0.5 \
  --consensus-medoid-gripper-weight 0.25 \
  --consensus-medoid-continuity-weight 0.0 \
  --consensus-medoid-smoothness-weight 0.0 \
  --consensus-medoid-gripper-switch-weight 0.0 \
  --no-save-rollout-videos \
  --no-use-cuda-graphs \
  --rollout-dir "${OUT_DIR}" \
  --metrics-dir "${OUT_DIR}/metrics" \
  2>&1 | tee "${LOG_DIR}/eval.log"

echo
echo "=== DONE ==="
echo "Summary: ${OUT_DIR}/metrics/summary.json"
