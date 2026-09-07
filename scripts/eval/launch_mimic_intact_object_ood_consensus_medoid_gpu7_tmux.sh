#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_consensus_medoid_gpu7}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_consensus_medoid_16tasks_3seeds_gpu7_${DATE_TAG}}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing T5 embeddings: ${EMBEDDINGS}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs"
RUNNER="${OUT_ROOT}/run_all.sh"
cat > "${RUNNER}" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
OUT_ROOT="${OUT_ROOT}"; EMBEDDINGS="${EMBEDDINGS}"
TASKS=(
"widowx_cube_on_plate_clean|PutGreenCubeOnPlateInScene-v2|put green cube on plate"
"widowx_small_plate_on_green_cube_clean|PutSmallPlateOnGreenCubeInScene-v2|put the small plate on the green cube"
"widowx_carrot_on_sponge_clean|PutCarrotOnSpongeLargerInScene-v2|put carrot on sponge"
"widowx_eggplant_on_sponge_clean|PutEggplantOnSpongeLargerInScene-v2|put eggplant on sponge"
"widowx_coke_can_on_plate_clean|PutCokeCanOnPlateInScene-v2|put coke can on plate"
"widowx_pepsi_on_plate_clean|PutPepsiCanOnPlateInScene-v2|put pepsi can on plate"
"widowx_orange_juice_on_plate_clean|PutOrangeJuiceOnPlateInScene-v2|put orange juice on plate"
"widowx_nut_on_plate_clean|PutNutOnPlateInScene-v2|put nut on plate"
"widowx_carrot_on_keyboard_clean|PutCarrotOnKeyboardInScene-v2|put carrot on keyboard"
"widowx_eggplant_on_keyboard_clean|PutEggplantOnKeyboardInScene-v2|put eggplant on keyboard"
"widowx_carrot_on_ramekin_clean|PutCarrotOnRamekinInScene-v2|put carrot on ramekin"
"widowx_carrot_on_wheel_clean|PutCarrotOnWheelInScene-v2|put carrot on wheel"
"widowx_coke_can_on_keyboard_clean|PutCokeCanOnKeyboardInScene-v2|put coke can on keyboard"
"widowx_coke_can_on_ramekin_clean|PutCokeCanOnRamekinInScene-v2|put coke can on ramekin"
"widowx_coke_can_on_wheel_clean|PutCokeCanOnWheelInScene-v2|put coke can on wheel"
"widowx_nut_on_wheel_clean|PutNutOnWheelInScene-v2|put nut on wheel"
)
for seed in 0 1 2; do
  for spec in "\${TASKS[@]}"; do
    IFS='|' read -r task env instruction <<< "\${spec}"
    log="\${OUT_ROOT}/seed\${seed}/logs/\${task}.log"; mkdir -p "\$(dirname "\${log}")"
    printf '[%s] seed=%s task=%s\n' "\$(date -Is)" "\${seed}" "\${task}"
    if ! OUT_ROOT="\${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh "\${seed}" "\${task}" "\${env}" "\${instruction}" "\${OUT_ROOT}" 2>&1 | tee -a "\${log}"; then
      printf '[%s] retry seed=%s task=%s\n' "\$(date -Is)" "\${seed}" "\${task}" >&2
      sleep 5
      OUT_ROOT="\${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh "\${seed}" "\${task}" "\${env}" "\${instruction}" "\${OUT_ROOT}" 2>&1 | tee -a "\${log}"
    fi
    "${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "\${OUT_ROOT}" || true
  done
done
"${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "\${OUT_ROOT}"
printf '[%s] Complete\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}" "${REPO_ROOT}/scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
