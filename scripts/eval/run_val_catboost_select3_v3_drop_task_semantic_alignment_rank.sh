#!/usr/bin/env bash
# Usage: <rank> <cuda_device>
set -euo pipefail

RANK="${1:?rank}"
GPU="${2:?gpu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="eval_outputs/libero_spatial/strategy_runs/catboost_select3_v3_drop_task_semantic_alignment_val30"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"
SEL="$(tr -d '\n' < artifacts/libero_spatial/v3_drop_task_semantic_alignment/val_eval_episodes_all30.txt)"

mkdir -p "$OUT_DIR/videos/rank${RANK}" "$OUT_DIR/metrics/rank${RANK}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/libero/LIBERO"

exec "$PYTHON" eval/libero/run.py \
  --vam_experiment_name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
  --vam_video_model_path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt" \
  --vam_action_model_path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt" \
  --vam_dataset_statistics_path "${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_spatial_one.json" \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 \
  --vam_stop_video_denoising_step 0 --vam_num_execute_actions 5 \
  --task_suite_name libero_spatial --num_trials_per_task 10 \
  --max_eval_episodes 30 --max_control_steps 220 \
  --eval_rank "${RANK}" --eval_world_size 3 \
  --t5_embeddings_path "${T5_EMB}" \
  --regen_model_path "${REPO_ROOT}/artifacts/libero_spatial/v3_drop_task_semantic_alignment/catboost_chunk_regen.cbm" \
  --regen_strategy catboost_select --regen_num_candidates 3 --regen_max_attempts 0 \
  --selected_episodes "${SEL}" --no-use-cuda-graphs \
  --rollout_dir "${REPO_ROOT}/${OUT_DIR}/videos/rank${RANK}" \
  --metrics_dir "${REPO_ROOT}/${OUT_DIR}/metrics/rank${RANK}" \
  2>&1 | tee -a "${REPO_ROOT}/${OUT_DIR}/run_rank${RANK}.log"
