#!/usr/bin/env bash
# Evaluate two Bridge tasks per GPU with action/encoder-hidden rank-fusion planning.
set -euo pipefail
RANK="${1:?usage: $0 <rank 0..1> <gpu>}"
GPU="${2:?usage: $0 <rank 0..1> <gpu>}"
(( RANK >= 0 && RANK < 2 )) || { echo "rank must be 0..1" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
EMBEDDINGS="${T5_EMBEDDINGS:?T5_EMBEDDINGS is required}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"
VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"
for path in "${PYTHON}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${EMBEDDINGS}"; do [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }; done
export CUDA_VISIBLE_DEVICES="${GPU}"
export SAPIEN_RENDER_CUDA_ORDINAL=0
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/libero:${REPO_ROOT}/eval/bridge/SimplerEnv:${PYTHONPATH:-}"
RANK_DIR="${OUT_ROOT}/ranks/rank${RANK}_gpu${GPU}"
RESULT_DIR="${OUT_ROOT}/result"
mkdir -p "${RANK_DIR}" "${RESULT_DIR}"
cd "${RANK_DIR}"
run_task() {
  local env_name="$1" instruction="$2" scene_name="$3" robot="$4" overlay="$5" x="$6" y="$7" max_steps="$8"
  "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \
    --ckpt-path "ftcosmos_stop0_rank_fusion_k3" --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps "${max_steps}" --logging-dir "${RESULT_DIR}" \
    --env-name "${env_name}" --scene-name "${scene_name}" --rgb-overlay-path "${overlay}" \
    --robot-init-x-range "${x}" "${x}" 1 --robot-init-y-range "${y}" "${y}" 1 \
    --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --vam-experiment-name w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256 \
    --vam-video-model-path "${VIDEO_CKPT}" --vam-action-model-path "${ACTION_CKPT}" \
    --vam-dataset-statistics-path "${STATS}" --vam-img-horizon 5 --vam-lowdim-horizon 1 \
    --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \
    --vam-consensus-rank-fusion --vam-consensus-num-candidates 3 \
    --vam-prompt-embeddings-path "${EMBEDDINGS}" 2>&1 | tee -a "${RANK_DIR}/run.log"
}
BRIDGE_DIR="${REPO_ROOT}/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting"
if (( RANK == 0 )); then
  run_task PutCarrotOnPlateInScene-v0 "put carrot on plate" bridge_table_1_v1 widowx "${BRIDGE_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
  run_task PutSpoonOnTableClothInScene-v0 "put the spoon on the towel" bridge_table_1_v1 widowx "${BRIDGE_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
else
  run_task StackGreenCubeOnYellowCubeBakedTexInScene-v0 "stack the green block on the yellow block" bridge_table_1_v1 widowx "${BRIDGE_DIR}/bridge_real_eval_1.png" 0.147 0.028 60
  run_task PutEggplantInBasketScene-v0 "put eggplant into yellow basket" bridge_table_1_v2 widowx_sink_camera_setup "${BRIDGE_DIR}/bridge_sink.png" 0.127 0.06 120
fi
