#!/usr/bin/env bash
set -euo pipefail
RANK="${1:?usage: $0 <rank 0..3> <gpu> <seed> <out_root>}"
GPU="${2:?usage: $0 <rank 0..3> <gpu> <seed> <out_root>}"
SEED="${3:?usage: $0 <rank 0..3> <gpu> <seed> <out_root>}"
OUT_ROOT="${4:?usage: $0 <rank 0..3> <gpu> <seed> <out_root>}"
(( RANK >= 0 && RANK < 4 )) || { echo "rank must be 0..3" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"
EMBEDDINGS="${T5_EMBEDDINGS:?T5_EMBEDDINGS is required}"
VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"
for path in "${PYTHON}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${EMBEDDINGS}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="${GPU}"
export SAPIEN_RENDER_CUDA_ORDINAL=0
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
export TOKENIZERS_PARALLELISM=false
INTACT_MS2="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim"
export MS2_REAL2SIM_ASSET_DIR="${INTACT_MS2}/data"
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/bridge/SimplerEnv:${REPO_ROOT}/eval/int-act/third_party/SimplerEnv:${INTACT_MS2}:${REPO_ROOT}/eval/libero:${PYTHONPATH:-}"
export MIMIC_VIDEO_SAMPLING_SEED="${SEED}"
SEED_ROOT="${OUT_ROOT}/seed${SEED}"
mkdir -p "${SEED_ROOT}/logs" "${SEED_ROOT}/result"
cd "${SEED_ROOT}"
BRIDGE_DIR="${INTACT_MS2}/data/real_inpainting"
VIDEO_CKPT_NAME="v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
ACTION_CKPT_NAME="w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
run_task() {
  local task="$1" env_name="$2" instruction="$3"
  "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \
    --ckpt-path "mimic_video_intact_object_ood_seed${SEED}" \
    --robot widowx --policy-setup widowx_bridge --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name "${env_name}" --scene-name bridge_table_1_v1 \
    --additional-env-save-tags "intact_object_ood_seed${SEED}" \
    --rgb-overlay-path "${BRIDGE_DIR}/bridge_real_eval_1.png" \
    --robot-init-x-range 0.147 0.147 1 --robot-init-y-range 0.028 0.028 1 \
    --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --vam-experiment-name "${ACTION_CKPT_NAME}" \
    --vam-video-model-path "${VIDEO_CKPT}" --vam-action-model-path "${ACTION_CKPT}" \
    --vam-dataset-statistics-path "${STATS}" --vam-img-horizon 5 --vam-lowdim-horizon 1 \
    --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \
    --vam-prompt-embeddings-path "${EMBEDDINGS}" --logging-dir "${SEED_ROOT}/result" \
    2>&1 | tee -a "${SEED_ROOT}/logs/${task}.log"
}
case "${RANK}" in
  0)
    run_task widowx_cube_on_plate_clean PutGreenCubeOnPlateInScene-v2 "put green cube on plate"
    run_task widowx_small_plate_on_green_cube_clean PutSmallPlateOnGreenCubeInScene-v2 "put the small plate on the green cube"
    run_task widowx_carrot_on_sponge_clean PutCarrotOnSpongeLargerInScene-v2 "put carrot on sponge"
    run_task widowx_eggplant_on_sponge_clean PutEggplantOnSpongeLargerInScene-v2 "put eggplant on sponge"
    ;;
  1)
    run_task widowx_coke_can_on_plate_clean PutCokeCanOnPlateInScene-v2 "put coke can on plate"
    run_task widowx_pepsi_on_plate_clean PutPepsiCanOnPlateInScene-v2 "put pepsi can on plate"
    run_task widowx_orange_juice_on_plate_clean PutOrangeJuiceOnPlateInScene-v2 "put orange juice on plate"
    run_task widowx_nut_on_plate_clean PutNutOnPlateInScene-v2 "put nut on plate"
    ;;
  2)
    run_task widowx_carrot_on_keyboard_clean PutCarrotOnKeyboardInScene-v2 "put carrot on keyboard"
    run_task widowx_eggplant_on_keyboard_clean PutEggplantOnKeyboardInScene-v2 "put eggplant on keyboard"
    run_task widowx_carrot_on_ramekin_clean PutCarrotOnRamekinInScene-v2 "put carrot on ramekin"
    run_task widowx_carrot_on_wheel_clean PutCarrotOnWheelInScene-v2 "put carrot on wheel"
    ;;
  3)
    run_task widowx_coke_can_on_keyboard_clean PutCokeCanOnKeyboardInScene-v2 "put coke can on keyboard"
    run_task widowx_coke_can_on_ramekin_clean PutCokeCanOnRamekinInScene-v2 "put coke can on ramekin"
    run_task widowx_coke_can_on_wheel_clean PutCokeCanOnWheelInScene-v2 "put coke can on wheel"
    run_task widowx_nut_on_wheel_clean PutNutOnWheelInScene-v2 "put nut on wheel"
    ;;
  *) echo "unknown rank ${RANK}" >&2; exit 2;;
esac
