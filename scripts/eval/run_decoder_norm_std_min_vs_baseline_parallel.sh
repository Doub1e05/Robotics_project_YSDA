#!/usr/bin/env bash
set -euo pipefail

NUM_RUNS="${1:-70}"
OUT_ROOT="${2:-/home/motovilovil/Robotics/robotics_project/mimic-video/eval_outputs/libero_spatial/decoder_metric_select/task0_init9_${NUM_RUNS}rollouts_norm_std_layer23_min_vs_baseline}"

REPO_ROOT="/home/motovilovil/Robotics/robotics_project/mimic-video"
SMART_RUNNER="${REPO_ROOT}/scripts/eval/run_decoder_metric_select_fast.sh"
BASELINE_RUNNER="${REPO_ROOT}/scripts/eval/run_baseline_fast.sh"
GPUS=(1 2 5)
LAYER_IDX=23
METRIC_NAME="decoder_norm_std"
ACTION_SUBSET="full_chunk"
REDUCE_MODE="min"
SEED_BASE_SMART=41000
SEED_BASE_BASELINE=51000

SMART_OUT_DIR="${OUT_ROOT}/smart_select"
BASELINE_OUT_DIR="${OUT_ROOT}/baseline"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${SMART_OUT_DIR}" "${BASELINE_OUT_DIR}" "${LOG_DIR}"

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

run_stage() {
  local stage_name="$1"
  local out_dir="$2"
  local seed_base="$3"
  local runner_script="$4"
  shift 4
  local -a runner_extra_args=( "$@" )
  local -a pids=()

  echo "== Stage: ${stage_name} =="

  local idx=0
  while read -r run_start run_end; do
    local gpu="${GPUS[$idx]}"
    if [[ "${run_start}" -eq 0 ]]; then
      echo "[${stage_name}] skip GPU ${gpu}"
      idx=$((idx + 1))
      continue
    fi

    local log_path="${LOG_DIR}/${stage_name}_gpu${gpu}.log"
    echo "[${stage_name}] GPU ${gpu} -> runs ${run_start}-${run_end}"

    (
      bash "${runner_script}" "${gpu}" "${run_start}" "${run_end}" "${seed_base}" "${out_dir}" "${runner_extra_args[@]}"
    ) >"${log_path}" 2>&1 &
    pids+=( "$!" )
    idx=$((idx + 1))
  done < <(partition_ranges "${NUM_RUNS}" "${#GPUS[@]}")

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

run_stage_smart() {
  run_stage \
    "smart_select" \
    "${SMART_OUT_DIR}" \
    "${SEED_BASE_SMART}" \
    "${SMART_RUNNER}" \
    23 \
    "decoder_norm_std" \
    "full_chunk" \
    "min"
}

run_stage_baseline() {
  run_stage \
    "baseline" \
    "${BASELINE_OUT_DIR}" \
    "${SEED_BASE_BASELINE}" \
    "${BASELINE_RUNNER}"
}

combine_csvs() {
  local stage_dir="$1"
  local combined_path="$2"
  local first=1
  : > "${combined_path}"

  local gpu
  for gpu in "${GPUS[@]}"; do
    local csv_path="${stage_dir}/gpu${gpu}_runs.csv"
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
  local stage_dir="$1"
  local combined_path="$2"
  local summary_path="$3"
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
echo "Runs per stage: ${NUM_RUNS}"
echo "Decoder selection: layer=${LAYER_IDX}, metric=${METRIC_NAME}, subset=${ACTION_SUBSET}, reduce=${REDUCE_MODE}"

run_stage_smart
combine_csvs "${SMART_OUT_DIR}" "${SMART_OUT_DIR}/all_runs.csv"
write_summary "${SMART_OUT_DIR}" "${SMART_OUT_DIR}/all_runs.csv" "${SMART_OUT_DIR}/summary.json"

run_stage_baseline
combine_csvs "${BASELINE_OUT_DIR}" "${BASELINE_OUT_DIR}/all_runs.csv"
write_summary "${BASELINE_OUT_DIR}" "${BASELINE_OUT_DIR}/all_runs.csv" "${BASELINE_OUT_DIR}/summary.json"

echo "Done."
echo "Smart select: ${SMART_OUT_DIR}/all_runs.csv"
echo "Baseline: ${BASELINE_OUT_DIR}/all_runs.csv"
