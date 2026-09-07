#!/usr/bin/env bash
# Resume the current 8-task consensus evaluation, then run a clean second
# evaluation with a different fixed candidate-seed set on the same GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_consensus_two_candidate_sets_gpu2_3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
FIRST_OUT_ROOT="${FIRST_OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_consensus_8tasks_candidates_2_996_997_gpu5_20260907_103910}"
SECOND_OUT_ROOT="${SECOND_OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_consensus_8tasks_candidates_3_998_999_gpu2_3_${DATE_TAG}}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"

FIRST_OUT_ROOT="$(realpath -m "${FIRST_OUT_ROOT}")"
SECOND_OUT_ROOT="$(realpath -m "${SECOND_OUT_ROOT}")"
[[ -d "${FIRST_OUT_ROOT}" ]] || { echo "Missing first output directory: ${FIRST_OUT_ROOT}" >&2; exit 1; }
[[ -x "${PYTHON}" && -f "${EMBEDDINGS}" ]] || { echo "Missing Python or T5 embeddings" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }

mkdir -p "${FIRST_OUT_ROOT}/logs" "${SECOND_OUT_ROOT}/logs"
RUNNER="${FIRST_OUT_ROOT}/run_candidate_sets_gpu2_gpu3_${DATE_TAG}.sh"

cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${FIRST_OUT_ROOT}/logs/run_candidate_sets_gpu2_gpu3_${DATE_TAG}.log") 2>&1

PYTHON="${PYTHON}"
EMBEDDINGS="${EMBEDDINGS}"

run_task() {
  local output_root="\$1" candidate_seeds="\$2" gpu="\$3" task="\$4" env="\$5" instruction="\$6"
  local log attempt
  log="\${output_root}/seed0/logs/\${task}_gpu\${gpu}.log"
  mkdir -p "\$(dirname "\${log}")"
  printf '[%s] set=%s gpu=%s task=%s start\n' "\$(date -Is)" "\${candidate_seeds}" "\${gpu}" "\${task}" | tee -a "\${log}"
  for attempt in 1 2; do
    if MIMIC_VIDEO_CANDIDATE_SEEDS="\${candidate_seeds}" OUT_ROOT="\${output_root}" T5_EMBEDDINGS="\${EMBEDDINGS}" \\
      bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh 0 "\${task}" "\${env}" "\${instruction}" "\${output_root}" "\${gpu}" >> "\${log}" 2>&1; then
      return 0
    fi
    printf '[%s] set=%s gpu=%s task=%s attempt=%s failed; retrying\n' "\$(date -Is)" "\${candidate_seeds}" "\${gpu}" "\${task}" "\${attempt}" | tee -a "\${log}"
    sleep 10
  done
  return 1
}

run_set() {
  local output_root="\$1" candidate_seeds="\$2"
  printf '[%s] candidate_set=%s evaluation_start output=%s\n' "\$(date -Is)" "\${candidate_seeds}" "\${output_root}"

  worker_gpu2() {
    run_task "\${output_root}" "\${candidate_seeds}" 2 widowx_cube_on_plate_clean PutGreenCubeOnPlateInScene-v2 "put green cube on plate"
    run_task "\${output_root}" "\${candidate_seeds}" 2 widowx_small_plate_on_green_cube_clean PutSmallPlateOnGreenCubeInScene-v2 "put the small plate on the green cube"
    run_task "\${output_root}" "\${candidate_seeds}" 2 widowx_eggplant_on_sponge_clean PutEggplantOnSpongeLargerInScene-v2 "put eggplant on sponge"
    run_task "\${output_root}" "\${candidate_seeds}" 2 widowx_pepsi_on_plate_clean PutPepsiCanOnPlateInScene-v2 "put pepsi can on plate"
    run_task "\${output_root}" "\${candidate_seeds}" 2 widowx_coke_can_on_keyboard_clean PutCokeCanOnKeyboardInScene-v2 "put coke can on keyboard"
  }
  worker_gpu3() {
    run_task "\${output_root}" "\${candidate_seeds}" 3 widowx_carrot_on_sponge_clean PutCarrotOnSpongeLargerInScene-v2 "put carrot on sponge"
    run_task "\${output_root}" "\${candidate_seeds}" 3 widowx_coke_can_on_plate_clean PutCokeCanOnPlateInScene-v2 "put coke can on plate"
    run_task "\${output_root}" "\${candidate_seeds}" 3 widowx_carrot_on_keyboard_clean PutCarrotOnKeyboardInScene-v2 "put carrot on keyboard"
  }

  worker_gpu2 & local pid2=\$!
  worker_gpu3 & local pid3=\$!
  local status=0
  wait "\${pid2}" || status=1
  wait "\${pid3}" || status=1
  "\${PYTHON}" scripts/eval/collect_intact_object_ood_8task_sr.py --out-root "\${output_root}" --candidate-seeds "\${candidate_seeds}"
  (( status == 0 )) || return "\${status}"
  printf '[%s] candidate_set=%s evaluation_complete\n' "\$(date -Is)" "\${candidate_seeds}"
}

# The first root is resumed in place; evaluator-level video checks skip prior rollouts.
run_set "${FIRST_OUT_ROOT}" "2,996,997"
run_set "${SECOND_OUT_ROOT}" "3,998,999"
printf '[%s] Both candidate-seed evaluations complete\n' "\$(date -Is)"
EOF

chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'Started tmux session: %s\nFirst (resume): %s\nSecond (new): %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${FIRST_OUT_ROOT}" "${SECOND_OUT_ROOT}" "${SESSION}"
