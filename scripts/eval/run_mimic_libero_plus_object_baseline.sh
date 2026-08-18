#!/usr/bin/env bash
# Baseline mimic-video evaluation on the standard LIBERO-Object suite packaged by LIBERO-PLUS.
# Default: 10 tasks × 10 fixed initial states = 100 episodes.
set -euo pipefail

GPU="${GPU:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-100}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-220}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_object_plus/baseline_${MAX_EVAL_EPISODES}ep_t4_${DATE_TAG}}"
LIBERO_ROOT="${REPO_ROOT}/LIBERO-plus"
CONFIG_DIR="${REPO_ROOT}/.cache/libero_config_plus"

VIDEO_CKPT="${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt"
ACTION_CKPT="${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000029997.pt"
STATS="${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_object_one.json"
T5_EMBEDDINGS="${REPO_ROOT}/model/checkpoints/libero_t5_embeddings.pkl"

for path in "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${T5_EMBEDDINGS}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
mkdir -p "${OUT_DIR}/videos" "${OUT_DIR}/metrics" "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/config.yaml" <<EOF2
benchmark_root: ${LIBERO_ROOT}/libero/libero
bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_ROOT}/libero/libero/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_ROOT}/libero/libero/assets
EOF2

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${CUDA_HOME:-${REPO_ROOT}/model/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export LIBERO_CONFIG_PATH="${CONFIG_DIR}"
export PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:${REPO_ROOT}"

echo "=== LIBERO-PLUS Object baseline ==="
echo "episodes=${MAX_EVAL_EPISODES}; trials/task=${NUM_TRIALS_PER_TASK}; max-control-steps=${MAX_CONTROL_STEPS}; gpu=${GPU}"
echo "output=${OUT_DIR}"
exec "${PYTHON}" "${REPO_ROOT}/eval/libero/run.py" \
  --vam_experiment_name w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128 \
  --vam_video_model_path "${VIDEO_CKPT}" \
  --vam_action_model_path "${ACTION_CKPT}" \
  --vam_dataset_statistics_path "${STATS}" \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 --vam_stop_video_denoising_step 0 --vam_num_execute_actions 5 \
  --task_suite_name libero_object --num_trials_per_task "${NUM_TRIALS_PER_TASK}" --max_eval_episodes "${MAX_EVAL_EPISODES}" --max_control_steps "${MAX_CONTROL_STEPS}" \
  --eval_rank 0 --eval_world_size 1 --seed 0 --t5_embeddings_path "${T5_EMBEDDINGS}" \
  --regen_strategy none --use-fp16 --no-use-cuda-graphs --no-save-rollout-videos \
  --rollout_dir "${OUT_DIR}/videos" --metrics_dir "${OUT_DIR}/metrics" 2>&1 | tee "${OUT_DIR}/run.log"
