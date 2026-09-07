#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_baseline_resume_4gpu}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${OUT_ROOT}/t5_embeddings/intact_object_ood.pt}"
CHECKPOINT_DIR="${REPO_ROOT}/model/checkpoints"
VIDEO_CKPT="${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_CKPT="${CHECKPOINT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CHECKPOINT_DIR}/dataset_statistics/bridge.json"
INTACT_MS2="${REPO_ROOT}/eval/int-act/third_party/ManiSkill2_real2sim"

for path in "${PYTHON}" "${EMBEDDINGS}" "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/seed2/result" "${OUT_ROOT}/seed2/logs"
RUNNER="${OUT_ROOT}/resume_missing_seed2_4gpu.sh"
cat > "${RUNNER}" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/resume_missing_seed2_4gpu.log") 2>&1
run_part() {
  local gpu="\$1" task="\$2" env_name="\$3" instruction="\$4" start="\$5" end="\$6"
  local log="${OUT_ROOT}/seed2/logs/resume_\${task}_gpu\${gpu}_\${start}_\${end}.log"
  export CUDA_VISIBLE_DEVICES="\${gpu}" SAPIEN_RENDER_CUDA_ORDINAL=0 VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
  export TOKENIZERS_PARALLELISM=false MS2_REAL2SIM_ASSET_DIR="${INTACT_MS2}/data" MIMIC_VIDEO_SAMPLING_SEED=2
  export PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/eval/bridge/SimplerEnv:${REPO_ROOT}/eval/int-act/third_party/SimplerEnv:${INTACT_MS2}:${REPO_ROOT}/eval/libero:\${PYTHONPATH:-}"
  mkdir -p "\$(dirname "\${log}")"
  "${PYTHON}" "${REPO_ROOT}/eval/bridge/SimplerEnv/simpler_env/main_inference.py" \\
    --ckpt-path mimic_video_intact_object_ood_seed2 --robot widowx --policy-setup widowx_bridge \\
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 --env-name "\${env_name}" --scene-name bridge_table_1_v1 \\
    --additional-env-save-tags intact_object_ood_seed2 \\
    --rgb-overlay-path "${INTACT_MS2}/data/real_inpainting/bridge_real_eval_1.png" \\
    --robot-init-x-range 0.147 0.147 1 --robot-init-y-range 0.028 0.028 1 \\
    --obj-variation-mode episode --obj-episode-range "\${start}" "\${end}" \\
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \\
    --vam-experiment-name w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256 \\
    --vam-video-model-path "${VIDEO_CKPT}" --vam-action-model-path "${ACTION_CKPT}" \\
    --vam-dataset-statistics-path "${STATS}" --vam-img-horizon 5 --vam-lowdim-horizon 1 \\
    --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \\
    --vam-prompt-embeddings-path "${EMBEDDINGS}" --logging-dir "${OUT_ROOT}/seed2/result" \\
    > "\${log}" 2>&1
}
pids=()
run_part 4 small_plate PutSmallPlateOnGreenCubeInScene-v2 "put the small plate on the green cube" 10 17 & pids[0]=\$!
run_part 5 small_plate PutSmallPlateOnGreenCubeInScene-v2 "put the small plate on the green cube" 17 24 & pids[1]=\$!
run_part 6 carrot_sponge PutCarrotOnSpongeLargerInScene-v2 "put carrot on sponge" 0 24 & pids[2]=\$!
run_part 7 eggplant_sponge PutEggplantOnSpongeLargerInScene-v2 "put eggplant on sponge" 0 24 & pids[3]=\$!
status=0
for pid in "\${pids[@]}"; do wait "\${pid}" || status=1; done
"${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "${OUT_ROOT}"
exit "\${status}"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n resume "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
