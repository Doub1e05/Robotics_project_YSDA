#!/usr/bin/env bash
set -euo pipefail

NUM_RUNS="${1:-100}"
OUT_ROOT="${2:-/home/motovilovil/Robotics/robotics_project/mimic-video/eval_outputs/libero_spatial/decoder_metric_select/task0_init9_${NUM_RUNS}rollouts_rank_combo_v1_layer23}"

REPO_ROOT="/home/motovilovil/Robotics/robotics_project/mimic-video"
RUNNER="${REPO_ROOT}/scripts/eval/run_decoder_metric_select_fast.sh"
GPUS=(1 2 5)
SEED_BASE=81000
LAYER_IDX=23
METRIC_NAME="decoder_rank_combo_v1"
ACTION_SUBSET="full_chunk"
REDUCE_MODE="max"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

partition_ranges() {
  local total="$1"
  local workers="$2"
  local base=$(( total / workers ))
  local extra=$(( total % workers ))
  local start=1
  local worker count end

  for worker in $(seq 0 $((workers - 1))); do
    count="${base}"
    if [[ "${worker}" -lt "${extra}" ]]; then
      count=$((count + 1))
    fi
    if [[ "${count}" -le 0 ]]; then
      echo "0 0"
      continue
    fi
    end=$((start + count - 1))
    echo "${start} ${end}"
    start=$((end + 1))
  done
}

combine_csvs() {
  local combined_path="$1"
  local first=1
  : > "${combined_path}"

  local gpu
  for gpu in "${GPUS[@]}"; do
    local csv_path="${OUT_ROOT}/gpu${gpu}_runs.csv"
    if [[ ! -f "${csv_path}" ]]; then
      continue
    fi
    if [[ "${first}" -eq 1 ]]; then
      cat "${csv_path}" > "${combined_path}"
      first=0
    else
      tail -n +2 "${csv_path}" >> "${combined_path}"
    fi
  done
}

write_summary() {
  local combined_path="$1"
  local summary_path="$2"
  python3 - "${combined_path}" "${summary_path}" <<'PY'
import csv
import json
import sys
from pathlib import Path

combined_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

rows = []
if combined_path.exists():
    with combined_path.open() as f:
        rows = list(csv.DictReader(f))

successes = sum(int(row["success"]) for row in rows)
total = len(rows)
summary = {
    "total_runs": total,
    "successes": successes,
    "failures": total - successes,
    "success_rate": (successes / total) if total else None,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

echo "Output root: ${OUT_ROOT}"
echo "GPUs: ${GPUS[*]}"
echo "Runs: ${NUM_RUNS}"
echo "Decoder selection: layer=${LAYER_IDX}, metric=${METRIC_NAME}, subset=${ACTION_SUBSET}, reduce=${REDUCE_MODE}"

declare -a pids=()
idx=0
while read -r run_start run_end; do
  gpu="${GPUS[$idx]}"
  if [[ "${run_start}" -eq 0 ]]; then
    echo "[rank_combo_v1] skip GPU ${gpu}"
    idx=$((idx + 1))
    continue
  fi

  log_path="${LOG_DIR}/gpu${gpu}.log"
  echo "[rank_combo_v1] GPU ${gpu} -> runs ${run_start}-${run_end}"
  (
    bash "${RUNNER}" "${gpu}" "${run_start}" "${run_end}" "${SEED_BASE}" "${OUT_ROOT}" "${LAYER_IDX}" "${METRIC_NAME}" "${ACTION_SUBSET}" "${REDUCE_MODE}"
  ) >"${log_path}" 2>&1 &
  pids+=( "$!" )
  idx=$((idx + 1))
done < <(partition_ranges "${NUM_RUNS}" "${#GPUS[@]}")

for pid in "${pids[@]}"; do
  wait "${pid}"
done

combine_csvs "${OUT_ROOT}/all_runs.csv"
write_summary "${OUT_ROOT}/all_runs.csv" "${OUT_ROOT}/summary.json"

echo "Done."
echo "Results: ${OUT_ROOT}/all_runs.csv"
