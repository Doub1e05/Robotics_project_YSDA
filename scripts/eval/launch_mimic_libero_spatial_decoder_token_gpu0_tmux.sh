#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
T5_EMB="${T5_EMB:-${REPO_ROOT}/model/checkpoints/libero_t5_embeddings.pkl}"
GPU="${MIMIC_GPU:-0}"
CANDIDATE_SEEDS="${MIMIC_CANDIDATE_SEEDS:-0,1,2}"
SESSION="${MIMIC_SESSION:-mimic_libero_spatial_decoder_token_gpu0}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${MIMIC_OUT_ROOT:-${REPO_ROOT}/eval_outputs/libero_spatial/mimic_video_decoder_action_token_medoid_seeds_0_1_2_gpu0_${DATE_TAG}}"

tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/metrics"

RUNNER="${OUT_ROOT}/run_all.sh"
cp "${BASH_SOURCE[0]}" "${OUT_ROOT}/launcher_snapshot.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
export CUDA_VISIBLE_DEVICES="${GPU}"
export MIMIC_VIDEO_CANDIDATE_SEEDS="${CANDIDATE_SEEDS}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${OUT_ROOT}/logs/robosuite.log"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/eval/libero/LIBERO:${REPO_ROOT}/model:${PYTHONPATH:-}"

"${PYTHON}" eval/libero/run.py \
  --vam-experiment-name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
  --vam-video-model-path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt" \
  --vam-action-model-path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt" \
  --vam-dataset-statistics-path "${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_spatial_one.json" \
  --vam-img-horizon 5 --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 5 \
  --task-suite-name libero_spatial --num-trials-per-task 10 \
  --max-eval-episodes 100 --max-control-steps 220 --seed 0 \
  --t5-embeddings-path "${T5_EMB}" \
  --regen-strategy decoder_action_token_medoid --regen-num-candidates 3 \
  --no-save-rollout-videos --no-use-cuda-graphs \
  --rollout-dir "${OUT_ROOT}" --metrics-dir "${OUT_ROOT}/metrics"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n libero_spatial_decoder "bash ${RUNNER}"
printf 'session=%s\noutput=%s\ncandidate_seeds=%s\n' "${SESSION}" "${OUT_ROOT}" "${CANDIDATE_SEEDS}"
