#!/usr/bin/env bash
# Re-run a selected, labelled set of SIMPLER-Bridge episodes with full latent diagnostics.
set -euo pipefail

RANK="${1:?usage: $0 <rank 0..3> <gpu>}"
GPU="${2:?usage: $0 <rank 0..3> <gpu>}"
MANIFEST="${3:?manifest TSV}"
(( RANK >= 0 && RANK < 4 )) || { echo "rank must be 0..3" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
EMBEDDINGS="${T5_EMBEDDINGS:?T5_EMBEDDINGS is required}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"
VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"
for path in "${PYTHON}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${EMBEDDINGS}" "${MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export SAPIEN_RENDER_CUDA_ORDINAL=0
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/libero:${REPO_ROOT}/eval/bridge/SimplerEnv:${PYTHONPATH:-}"
RANK_DIR="${OUT_ROOT}/ranks/rank${RANK}_gpu${GPU}"
RESULT_DIR="${OUT_ROOT}/result"
mkdir -p "${RANK_DIR}" "${RESULT_DIR}"
cd "${RANK_DIR}"
BRIDGE_DIR="${REPO_ROOT}/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting"

run_episode() {
  local task="$1" episode="$2" expected="$3"
  local scene robot overlay x y max_steps
  case "${task}" in
    PutCarrotOnPlateInScene-v0) scene=bridge_table_1_v1; robot=widowx; overlay="${BRIDGE_DIR}/bridge_real_eval_1.png"; x=0.147; y=0.028; max_steps=60 ;;
    PutSpoonOnTableClothInScene-v0) scene=bridge_table_1_v1; robot=widowx; overlay="${BRIDGE_DIR}/bridge_real_eval_1.png"; x=0.147; y=0.028; max_steps=60 ;;
    StackGreenCubeOnYellowCubeBakedTexInScene-v0) scene=bridge_table_1_v1; robot=widowx; overlay="${BRIDGE_DIR}/bridge_real_eval_1.png"; x=0.147; y=0.028; max_steps=60 ;;
    PutEggplantInBasketScene-v0) scene=bridge_table_1_v2; robot=widowx_sink_camera_setup; overlay="${BRIDGE_DIR}/bridge_sink.png"; x=0.127; y=0.06; max_steps=120 ;;
    *) echo "Unknown task: ${task}" >&2; exit 2 ;;
  esac
  echo "[rank ${RANK}] ${task} episode=${episode} expected=${expected}"
  "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \
    --ckpt-path "ftcosmos_stop0_latent_diagnostics" \
    --robot "${robot}" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps "${max_steps}" --logging-dir "${RESULT_DIR}" \
    --env-name "${task}" --scene-name "${scene}" --rgb-overlay-path "${overlay}" \
    --robot-init-x-range "${x}" "${x}" 1 --robot-init-y-range "${y}" "${y}" 1 \
    --obj-variation-mode episode --obj-episode-range "${episode}" "$((episode + 1))" \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --vam-experiment-name w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256 \
    --vam-video-model-path "${VIDEO_CKPT}" --vam-action-model-path "${ACTION_CKPT}" \
    --vam-dataset-statistics-path "${STATS}" --vam-img-horizon 5 --vam-lowdim-horizon 1 \
    --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \
    --vam-prompt-embeddings-path "${EMBEDDINGS}" \
    --vam-diagnostics-mode all --vam-representation-layer-indices 4 8 12 16 20 24 28 \
    --vam-decoder-capture-block-indices 8 16 23 \
    2>&1 | tee -a "${RANK_DIR}/run.log"
}

while IFS=$'\t' read -r assigned_rank task episode expected; do
  [[ "${assigned_rank}" == "rank" ]] && continue
  [[ "${assigned_rank}" == "${RANK}" ]] || continue
  run_episode "${task}" "${episode}" "${expected}"
done < "${MANIFEST}"
