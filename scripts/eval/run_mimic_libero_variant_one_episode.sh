#!/usr/bin/env bash
# Run one mimic-video episode against a LIBERO-compatible benchmark variant.
# Usage: bash scripts/eval/run_mimic_libero_variant_one_episode.sh <plus|pro> [gpu]
set -euo pipefail

VARIANT="${1:?usage: $0 <plus|pro> [gpu]}"
GPU="${2:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${VARIANT}" in
  plus) LIBERO_ROOT="${REPO_ROOT}/LIBERO-plus" ;;
  pro) LIBERO_ROOT="${REPO_ROOT}/LIBERO-PRO" ;;
  *) echo "variant must be plus or pro" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial_${VARIANT}/smoke_1ep}"
CONFIG_DIR="${REPO_ROOT}/.cache/libero_config_${VARIANT}"
mkdir -p "${OUT_DIR}/videos" "${OUT_DIR}/metrics" "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/config.yaml" <<EOF2
benchmark_root: ${LIBERO_ROOT}/libero/libero
bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_ROOT}/libero/libero/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_ROOT}/libero/libero/assets
EOF2
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled WANDB_SILENT=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# transformer-engine needs NVRTC from the CUDA pip wheel: the driver alone does not supply a toolkit.
export CUDA_HOME="${CUDA_HOME:-${REPO_ROOT}/model/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export LIBERO_CONFIG_PATH="${CONFIG_DIR}"
export PYTHONPATH="${LIBERO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:${REPO_ROOT}"
exec "${PYTHON}" "${REPO_ROOT}/eval/libero/run.py" \
  --vam_experiment_name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
  --vam_video_model_path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt" \
  --vam_action_model_path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt" \
  --vam_dataset_statistics_path "${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_spatial_one.json" \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 --vam_stop_video_denoising_step 0 --vam_num_execute_actions 5 \
  --task_suite_name libero_spatial --num_trials_per_task 1 --max_eval_episodes 1 --max_control_steps "${MAX_CONTROL_STEPS:-120}" \
  --eval_rank 0 --eval_world_size 1 --t5_embeddings_path "${REPO_ROOT}/model/checkpoints/libero_t5_embeddings.pkl" \
  --regen_strategy none --use-fp16 --no-use-cuda-graphs --rollout_dir "${OUT_DIR}/videos" --metrics_dir "${OUT_DIR}/metrics" \
  2>&1 | tee "${OUT_DIR}/run.log"
