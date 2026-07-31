#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python}"
T5_EMB="${T5_EMB:-/home/motovilovil/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl}"

GPU="${GPU:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
SEED_BASE="${SEED_BASE:-10000}"
SEED_STRIDE="${SEED_STRIDE:-1}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-220}"
DATE_TAG="${DATE_TAG:-$(TZ=Asia/Almaty date +%Y%m%d)}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_outputs/libero_spatial_pro/episode2_ramekin_consensus_medoid_${NUM_ROLLOUTS}x_gpu${GPU}_${DATE_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOSUITE_LOG_PATH="${REPO_ROOT}/logs/robosuite_episode2_consensus_medoid_gpu${GPU}.log"
export LIBERO_CONFIG_PATH="${REPO_ROOT}/eval_outputs/libero_config"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval/libero:${REPO_ROOT}/model:/home/motovilovil/Robotics/robotics_project/LIBERO-PRO"

echo "=== Episode 2 Consensus-medoid evaluation ==="
echo "GPU=${GPU} rollouts=${NUM_ROLLOUTS} seeds=${SEED_BASE}..$((SEED_BASE + SEED_STRIDE * (NUM_ROLLOUTS - 1)))"
echo "Output: ${OUT_DIR}"
echo

"${PYTHON}" "${REPO_ROOT}/scripts/eval/run_boundary_multi_seed.py" \
  --task-id 1 \
  --episode-idx 0 \
  --task-suite-name libero_spatial_object \
  --num-rollouts "${NUM_ROLLOUTS}" \
  --seed-base "${SEED_BASE}" \
  --seed-stride "${SEED_STRIDE}" \
  --max-control-steps "${MAX_CONTROL_STEPS}" \
  --regen-strategy consensus_medoid \
  --regen-num-candidates 3 \
  --consensus-medoid-horizon 5 \
  --consensus-medoid-temporal-discount 0.9 \
  --consensus-medoid-translation-weight 1.0 \
  --consensus-medoid-rotation-weight 0.5 \
  --consensus-medoid-gripper-weight 0.25 \
  --consensus-medoid-continuity-weight 0.25 \
  --consensus-medoid-smoothness-weight 0.10 \
  --consensus-medoid-gripper-switch-weight 0.10 \
  --diagnostics-mode default \
  --replay-payload-mode none \
  --no-save-videos \
  --no-use-cuda-graphs \
  --no-resume \
  --rollout-dir "${OUT_DIR}" \
  --t5-embeddings-path "${T5_EMB}" \
  --progress-label "ep2 consensus-medoid ${NUM_ROLLOUTS}x" \
  2>&1 | tee "${LOG_DIR}/eval.log"

"${PYTHON}" - <<PY
import collections
import json
from pathlib import Path

root = Path("${OUT_DIR}")
metrics = root / "metrics"
episodes = [
    json.loads(line)
    for line in (metrics / "episode_traces.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
selected = collections.Counter()
selected_costs = []
unselected_costs = []
for episode in episodes:
    for chunk in episode.get("chunks", []):
        selected_idx = int(chunk["selected_candidate_idx"])
        selected[selected_idx] += 1
        for candidate in chunk.get("candidate_chunk_metrics", []):
            cost = candidate.get("metrics", {}).get("consensus_medoid_cost")
            if cost is None:
                continue
            if int(candidate["candidate_idx"]) == selected_idx:
                selected_costs.append(float(cost))
            else:
                unselected_costs.append(float(cost))

summary = json.loads((metrics / "summary.json").read_text(encoding="utf-8"))
analysis = {
    **summary,
    "strategy": "consensus_medoid",
    "num_candidates": 3,
    "selection_horizon": 5,
    "selected_candidate_counts": {str(key): value for key, value in sorted(selected.items())},
    "num_recorded_queries": int(sum(selected.values())),
    "mean_selected_cost": sum(selected_costs) / len(selected_costs) if selected_costs else None,
    "mean_unselected_cost": sum(unselected_costs) / len(unselected_costs) if unselected_costs else None,
}
(root / "consensus_analysis.json").write_text(
    json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(analysis, indent=2, ensure_ascii=False))
PY

echo
echo "=== DONE ==="
echo "Summary:    ${OUT_DIR}/metrics/summary.json"
echo "Candidates: ${OUT_DIR}/metrics/candidate_chunk_metrics.csv"
echo "Analysis:   ${OUT_DIR}/consensus_analysis.json"
