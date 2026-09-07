#!/usr/bin/env bash
# Run one visual-matching INT-ACT OOD SIMPLER episode with the official finetuned VAM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_bridge_baseline_ood_smoke}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"
T5_EMBEDDINGS="${T5_EMBEDDINGS:?Set T5_EMBEDDINGS to the INT-ACT OOD embedding file}"
INTACT_MS2="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim"

VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"
OVERLAY="${INTACT_MS2}/data/real_inpainting/bridge_real_eval_1.png"

for path in "${PYTHON}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${T5_EMBEDDINGS}" "${OVERLAY}" "${INTACT_MS2}/mani_skill2_real2sim/envs/custom_scenes/put_on_in_new.py"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export SAPIEN_RENDER_CUDA_ORDINAL=0
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
export TOKENIZERS_PARALLELISM=false
export MS2_REAL2SIM_ASSET_DIR="${INTACT_MS2}/data"
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/bridge/SimplerEnv:${INTACT_MS2}:${REPO_ROOT}/eval/libero:${PYTHONPATH:-}"

cd "${OUT_DIR}"
exec "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \
  --ckpt-path mimic_video_bridge_baseline_intact_ood_smoke \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "${MAX_EPISODE_STEPS:-60}" \
  --env-name PutGreenCubeOnPlateInScene-v2 --scene-name bridge_table_1_v1 \
  --additional-env-save-tags intact_ood_smoke \
  --rgb-overlay-path "${OVERLAY}" \
  --robot-init-x-range 0.147 0.147 1 --robot-init-y-range 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 1 \
  --robot-init-rot-quat-center 0 0 0 1 \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256 \
  --vam-video-model-path "${VIDEO_CKPT}" \
  --vam-action-model-path "${ACTION_CKPT}" \
  --vam-dataset-statistics-path "${STATS}" \
  --vam-img-horizon 5 --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \
  --vam-prompt-embeddings-path "${T5_EMBEDDINGS}" \
  2>&1 | tee run.log
