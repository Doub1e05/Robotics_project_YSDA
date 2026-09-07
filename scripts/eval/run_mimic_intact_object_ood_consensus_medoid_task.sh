#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; TASK_KEY="${2:?task_key}"; ENV_NAME="${3:?env_name}"; INSTRUCTION="${4:?instruction}"; OUT_ROOT="${5:?out_root}"; GPU="${6:-7}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"; EMBEDDINGS="${T5_EMBEDDINGS:?T5_EMBEDDINGS is required}"
VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"; INTACT_MS2="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim"
for path in "${PYTHON}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${EMBEDDINGS}"; do [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }; done
export CUDA_VISIBLE_DEVICES="${GPU}" SAPIEN_RENDER_CUDA_ORDINAL=0 VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select TOKENIZERS_PARALLELISM=false
export MS2_REAL2SIM_ASSET_DIR="${INTACT_MS2}/data" MIMIC_VIDEO_SAMPLING_SEED="${SEED}"
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/bridge/SimplerEnv:${REPO_ROOT}/eval/int-act/third_party/SimplerEnv:${INTACT_MS2}:${REPO_ROOT}/eval/libero:${PYTHONPATH:-}"
SEED_ROOT="${OUT_ROOT}/seed${SEED}"; mkdir -p "${SEED_ROOT}/logs" "${SEED_ROOT}/result"; cd "${SEED_ROOT}"
BRIDGE_DIR="${INTACT_MS2}/data/real_inpainting"; ACTION_CKPT_NAME="w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
exec "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \
  --ckpt-path "mimic_video_intact_object_ood_consensus_medoid_seed${SEED}" --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 --env-name "${ENV_NAME}" --scene-name bridge_table_1_v1 \
  --additional-env-save-tags "intact_object_ood_consensus_medoid_seed${SEED}" --rgb-overlay-path "${BRIDGE_DIR}/bridge_real_eval_1.png" \
  --robot-init-x-range 0.147 0.147 1 --robot-init-y-range 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name "${ACTION_CKPT_NAME}" --vam-video-model-path "${VIDEO_CKPT}" --vam-action-model-path "${ACTION_CKPT}" \
  --vam-dataset-statistics-path "${STATS}" --vam-img-horizon 5 --vam-lowdim-horizon 1 --vam-stop-video-denoising-step 0 \
  --vam-num-execute-actions 5 --vam-consensus-medoid-only --vam-consensus-num-candidates 3 \
  --vam-prompt-embeddings-path "${EMBEDDINGS}" --logging-dir "${SEED_ROOT}/result"
