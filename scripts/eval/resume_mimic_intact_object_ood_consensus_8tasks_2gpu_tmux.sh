#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the existing 8-task output directory}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
SESSION="${SESSION:-mimic_intact_object_ood_consensus_2_996_997_gpu2_3}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"
[[ -d "${OUT_ROOT}" ]] || { echo "Missing OUT_ROOT: ${OUT_ROOT}" >&2; exit 1; }
[[ -x "${PYTHON}" && -f "${EMBEDDINGS}" ]] || { echo "Missing Python or T5 embeddings" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
RUNNER="${OUT_ROOT}/resume_gpu2_gpu3.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/resume_gpu2_gpu3.log") 2>&1
OUT_ROOT="${OUT_ROOT}"
EMBEDDINGS="${EMBEDDINGS}"
export MIMIC_VIDEO_CANDIDATE_SEEDS="2,996,997"

run_task() {
  local gpu="\$1" task="\$2" env="\$3" instruction="\$4" log attempt
  log="\${OUT_ROOT}/seed0/logs/\${task}_resume_gpu\${gpu}.log"
  mkdir -p "\$(dirname "\${log}")"
  printf '[%s] gpu=%s task=%s candidate_seeds=%s resume_start\n' "\$(date -Is)" "\${gpu}" "\${task}" "\${MIMIC_VIDEO_CANDIDATE_SEEDS}" | tee -a "\${log}"
  for attempt in 1 2; do
    if OUT_ROOT="\${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh 0 "\${task}" "\${env}" "\${instruction}" "\${OUT_ROOT}" "\${gpu}" >> "\${log}" 2>&1; then
      "${PYTHON}" scripts/eval/collect_intact_object_ood_8task_sr.py --out-root "\${OUT_ROOT}" --candidate-seeds "\${MIMIC_VIDEO_CANDIDATE_SEEDS}" || true
      return 0
    fi
    printf '[%s] gpu=%s task=%s attempt=%s failed; retrying\n' "\$(date -Is)" "\${gpu}" "\${task}" "\${attempt}" | tee -a "\${log}"
    sleep 10
  done
  return 1
}

worker5() {
  run_task 2 widowx_small_plate_on_green_cube_clean PutSmallPlateOnGreenCubeInScene-v2 "put the small plate on the green cube"
  run_task 2 widowx_eggplant_on_sponge_clean PutEggplantOnSpongeLargerInScene-v2 "put eggplant on sponge"
  run_task 2 widowx_pepsi_on_plate_clean PutPepsiCanOnPlateInScene-v2 "put pepsi can on plate"
  run_task 2 widowx_coke_can_on_keyboard_clean PutCokeCanOnKeyboardInScene-v2 "put coke can on keyboard"
}
worker6() {
  run_task 3 widowx_carrot_on_sponge_clean PutCarrotOnSpongeLargerInScene-v2 "put carrot on sponge"
  run_task 3 widowx_coke_can_on_plate_clean PutCokeCanOnPlateInScene-v2 "put coke can on plate"
  run_task 3 widowx_carrot_on_keyboard_clean PutCarrotOnKeyboardInScene-v2 "put carrot on keyboard"
}
worker5 & pid5=\$!
worker6 & pid6=\$!
status=0
wait "\${pid5}" || status=1
wait "\${pid6}" || status=1
"${PYTHON}" scripts/eval/collect_intact_object_ood_8task_sr.py --out-root "\${OUT_ROOT}" --candidate-seeds "\${MIMIC_VIDEO_CANDIDATE_SEEDS}"
printf '[%s] Complete status=%s\n' "\$(date -Is)" "\${status}"
exit "\${status}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood_resume "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
