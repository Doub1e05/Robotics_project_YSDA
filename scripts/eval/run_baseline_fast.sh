#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu}"
RUN_START="${2:?run_start}"
RUN_END="${3:?run_end}"
SEED_BASE="${4:?seed_base}"
OUT_DIR="${5:?out_dir}"

REPO_ROOT="/home/motovilovil/Robotics/robotics_project/mimic-video"
PYTHON="/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python"
T5_EMB="/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl"

mkdir -p "${OUT_DIR}"
CSV_PATH="${OUT_DIR}/gpu${GPU}_runs.csv"
printf "run_id,gpu,seed,success\n" > "${CSV_PATH}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite_fast_baseline_gpu${GPU}.log"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/model:${REPO_ROOT}/eval/libero/LIBERO"

cd "${REPO_ROOT}"

for idx in $(seq "${RUN_START}" "${RUN_END}"); do
  seed=$((SEED_BASE + idx))
  echo "[GPU${GPU}] start baseline run ${idx} seed=${seed}"
  output=$(
    "${PYTHON}" eval/libero/run.py \
      --vam-experiment-name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
      --vam-video-model-path "${REPO_ROOT}/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt" \
      --vam-action-model-path "${REPO_ROOT}/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt" \
      --vam-dataset-statistics-path "${REPO_ROOT}/model/checkpoints/dataset_statistics/libero_spatial_one.json" \
      --vam-img-horizon 5 \
      --vam-lowdim-horizon 1 \
      --vam-stop-video-denoising-step 0 \
      --vam-num-execute-actions 5 \
      --task-suite-name libero_spatial \
      --num-trials-per-task 10 \
      --max-eval-episodes 1 \
      --selected-episodes 0:9 \
      --max-control-steps 120 \
      --seed "${seed}" \
      --t5-embeddings-path "${T5_EMB}" \
      --regen-strategy none \
      --no-write-chunk-metrics \
      --no-write-candidate-metrics \
      --no-write-representation-metrics \
      --no-write-action-metrics \
      --no-save-rollout-videos \
      --no-persist-metrics \
      --no-use-cuda-graphs 2>&1
  )
  printf "%s\n" "${output}"
  success=$(
    OUTPUT_TEXT="${output}" python3 - <<'PY'
import os, re
text = os.environ.get("OUTPUT_TEXT", "")
matches = re.findall(r"Total successes: (\d+)", text)
print(matches[-1] if matches else "")
PY
  )
  if [[ -z "${success}" ]]; then
    echo "[GPU${GPU}] failed to parse result for seed=${seed}" >&2
    exit 1
  fi
  printf "%s,%s,%s,%s\n" "${idx}" "${GPU}" "${seed}" "${success}" >> "${CSV_PATH}"
  echo "[GPU${GPU}] done baseline run ${idx} seed=${seed}"
done

echo "[GPU${GPU}] baseline batch complete"
