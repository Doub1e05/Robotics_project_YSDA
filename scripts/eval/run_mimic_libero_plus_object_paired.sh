#!/usr/bin/env bash
# Sequential, paired LIBERO-PLUS Object evaluation: baseline first, then consensus-only.
# Both arms use identical task order, seed, checkpoints, and control settings.
set -euo pipefail

GPU="${GPU:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-100}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-220}"
SEED="${SEED:-0}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/libero_object_plus/paired_${MAX_EVAL_EPISODES}ep_seed${SEED}_${DATE_TAG}}"
LIBERO_ROOT="${REPO_ROOT}/LIBERO-plus"
CONFIG_DIR="${REPO_ROOT}/.cache/libero_config_plus"

VIDEO_CKPT="${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt"
ACTION_CKPT="${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000029997.pt"
STATS="${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_object_one.json"
T5_EMBEDDINGS="${REPO_ROOT}/model/checkpoints/libero_t5_embeddings.pkl"

for path in "${VIDEO_CKPT}" "${ACTION_CKPT}" "${STATS}" "${T5_EMBEDDINGS}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
mkdir -p "${OUT_ROOT}" "${CONFIG_DIR}"
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

run_arm() {
  local arm="$1"
  local strategy="$2"
  local out_dir="${OUT_ROOT}/${arm}"
  mkdir -p "${out_dir}/metrics"
  echo "=== ${arm} (strategy=${strategy}) ==="
  "${PYTHON}" "${REPO_ROOT}/eval/libero/run.py" \
    --vam_experiment_name w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128 \
    --vam_video_model_path "${VIDEO_CKPT}" \
    --vam_action_model_path "${ACTION_CKPT}" \
    --vam_dataset_statistics_path "${STATS}" \
    --vam_img_horizon 5 --vam_lowdim_horizon 1 --vam_stop_video_denoising_step 0 --vam_num_execute_actions 5 \
    --task_suite_name libero_object --num_trials_per_task "${NUM_TRIALS_PER_TASK}" --max_eval_episodes "${MAX_EVAL_EPISODES}" --max_control_steps "${MAX_CONTROL_STEPS}" \
    --eval_rank 0 --eval_world_size 1 --seed "${SEED}" --t5_embeddings_path "${T5_EMBEDDINGS}" \
    --regen_strategy "${strategy}" --regen_num_candidates 3 \
    --consensus_medoid_horizon 5 --consensus_medoid_temporal_discount 0.9 \
    --consensus_medoid_translation_weight 1.0 --consensus_medoid_rotation_weight 0.5 --consensus_medoid_gripper_weight 0.25 \
    --consensus_medoid_continuity_weight 0.0 --consensus_medoid_smoothness_weight 0.0 --consensus_medoid_gripper_switch_weight 0.0 \
    --use-fp16 --no-use-cuda-graphs --no-save-rollout-videos --summary-only \
    --rollout_dir "${out_dir}/videos" --metrics_dir "${out_dir}/metrics" 2>&1 | tee "${out_dir}/run.log"
}

# Do not reorder: the comparison contract is baseline, then consensus-only.
run_arm baseline none
run_arm consensus_only consensus_medoid

python3 - "${OUT_ROOT}" <<"PY"
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
baseline = json.loads((root / "baseline/metrics/summary.json").read_text())
consensus = json.loads((root / "consensus_only/metrics/summary.json").read_text())
comparison = {
    "benchmark": "LIBERO-PLUS / libero_object",
    "comparison": "paired sequential baseline then consensus-only",
    "baseline": baseline,
    "consensus_only": consensus,
    "success_rate_delta": consensus["success_rate"] - baseline["success_rate"],
    "success_count_delta": consensus["num_successes"] - baseline["num_successes"],
}
(root / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
PY
