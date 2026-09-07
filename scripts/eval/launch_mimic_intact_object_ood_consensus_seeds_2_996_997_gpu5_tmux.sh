#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_consensus_2_996_997_gpu5}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_consensus_8tasks_candidates_2_996_997_gpu5_${DATE_TAG}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing embeddings: ${EMBEDDINGS}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs"
RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
OUT_ROOT="${OUT_ROOT}"
EMBEDDINGS="${EMBEDDINGS}"
CANDIDATE_SEEDS="2,996,997"
GPU=5
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
for spec in "\${TASKS[@]}"; do
  IFS='|' read -r task env instruction <<< "\${spec}"
  log="\${OUT_ROOT}/seed0/logs/\${task}.log"
  mkdir -p "\$(dirname "\${log}")"
  printf '[%s] task=%s gpu=%s candidate_seeds=%s\n' "\$(date -Is)" "\${task}" "\${GPU}" "\${CANDIDATE_SEEDS}" | tee -a "\${log}"
  MIMIC_VIDEO_CANDIDATE_SEEDS="\${CANDIDATE_SEEDS}" OUT_ROOT="\${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh 0 "\${task}" "\${env}" "\${instruction}" "\${OUT_ROOT}" "\${GPU}" 2>&1 | tee -a "\${log}"
  "${PYTHON}" scripts/eval/collect_intact_object_ood_8task_sr.py --out-root "\${OUT_ROOT}" --candidate-seeds "\${CANDIDATE_SEEDS}" || true
done
"${PYTHON}" scripts/eval/collect_intact_object_ood_8task_sr.py --out-root "\${OUT_ROOT}" --candidate-seeds "\${CANDIDATE_SEEDS}"
printf '[%s] Complete\n' "\$(date -Is)"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
