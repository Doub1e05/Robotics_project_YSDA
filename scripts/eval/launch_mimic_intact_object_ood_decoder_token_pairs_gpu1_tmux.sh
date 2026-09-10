#!/usr/bin/env bash
# Run two MIMIC-Video decoder action-token-medoid Object OOD evaluations on GPU 1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_decoder_token_pairs_gpu1}"
GPU="${GPU:-1}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_decoder_action_token_medoid_pairs_gpu1_${DATE_TAG}}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"

[[ -x "${PYTHON}" && -f "${EMBEDDINGS}" ]] || { echo "Missing Python or T5 embeddings" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
for label in seeds_4_5_6 seeds_7_8_9; do mkdir -p "${OUT_ROOT}/${label}/logs" "${OUT_ROOT}/${label}/result"; done

RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export TOKENIZERS_PARALLELISM=false
TASKS=(
"widowx_cube_on_plate_clean|PutGreenCubeOnPlateInScene-v2|put green cube on plate"
"widowx_small_plate_on_green_cube_clean|PutSmallPlateOnGreenCubeInScene-v2|put the small plate on the green cube"
"widowx_carrot_on_sponge_clean|PutCarrotOnSpongeLargerInScene-v2|put carrot on sponge"
"widowx_eggplant_on_sponge_clean|PutEggplantOnSpongeLargerInScene-v2|put eggplant on sponge"
"widowx_coke_can_on_plate_clean|PutCokeCanOnPlateInScene-v2|put coke can on plate"
"widowx_pepsi_on_plate_clean|PutPepsiCanOnPlateInScene-v2|put pepsi can on plate"
"widowx_carrot_on_keyboard_clean|PutCarrotOnKeyboardInScene-v2|put carrot on keyboard"
"widowx_coke_can_on_keyboard_clean|PutCokeCanOnKeyboardInScene-v2|put coke can on keyboard"
)
run_arm() {
  local output="\$1" seeds="\$2" tag="\$3"
  local log="\${output}/logs/orchestrator.log"
  for spec in "\${TASKS[@]}"; do
    IFS='|' read -r task env instruction <<< "\${spec}"
    printf '[%s] %s on GPU ${GPU}, candidates=%s\n' "\$(date -Is)" "\${task}" "\${seeds}" | tee -a "\${log}"
    MIMIC_VIDEO_CANDIDATE_SEEDS="\${seeds}" OUT_ROOT="\${output}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
      CUDA_VISIBLE_DEVICES="${GPU}" SAPIEN_RENDER_CUDA_ORDINAL=0 VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select \\
      MS2_REAL2SIM_ASSET_DIR="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim/data" \\
      PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/bridge/SimplerEnv:${REPO_ROOT}/eval/int-act/third_party/SimplerEnv:${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim:${REPO_ROOT}/eval/libero:\${PYTHONPATH:-}" \\
      "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \\
        --ckpt-path "mimic_video_intact_object_ood_decoder_action_token_medoid_\${tag}" --robot widowx --policy-setup widowx_bridge \\
        --control-freq 5 --sim-freq 500 --max-episode-steps 60 --env-name "\${env}" --scene-name bridge_table_1_v1 \\
        --additional-env-save-tags "intact_object_ood_decoder_action_token_medoid_\${tag}" \\
        --rgb-overlay-path "${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png" \\
        --robot-init-x-range 0.147 0.147 1 --robot-init-y-range 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 24 \\
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \\
        --vam-experiment-name w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256 \\
        --vam-video-model-path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt" \\
        --vam-action-model-path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt" \\
        --vam-dataset-statistics-path "${REPO_ROOT}/model/checkpoints/dataset_statistics/bridge.json" \\
        --vam-img-horizon 5 --vam-lowdim-horizon 1 --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \\
        --vam-latent-medoid-strategy decoder_action_tokens --vam-consensus-num-candidates 3 \\
        --vam-prompt-embeddings-path "${EMBEDDINGS}" --logging-dir "\${output}/result" \\
        2>&1 | tee -a "\${output}/logs/\${task}.log"
  done
  "${PYTHON}" scripts/eval/collect_intact_object_ood_8task_decoder_token_sr.py --out-root "\${output}" --candidate-seeds "\${seeds}" --tag "intact_object_ood_decoder_action_token_medoid_\${tag}" > "\${output}/logs/summary.log" 2>&1
}
run_arm "${OUT_ROOT}/seeds_4_5_6" 4,5,6 seeds_4_5_6 &
pid_a=\$!
run_arm "${OUT_ROOT}/seeds_7_8_9" 7,8,9 seeds_7_8_9 &
pid_b=\$!
wait "\${pid_a}"
wait "\${pid_b}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
