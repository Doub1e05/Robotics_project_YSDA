#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_consensus_medoid_4gpu}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT must point to existing medoid output}"
OUT_ROOT="$(realpath -m "${OUT_ROOT}")"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/t5_embeddings/intact_object_ood.pt}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing embeddings: ${EMBEDDINGS}" >&2; exit 1; }
[[ -d "${OUT_ROOT}" ]] || { echo "Missing output root: ${OUT_ROOT}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}/logs"
RUNNER="${OUT_ROOT}/resume_4gpu.sh"
cat > "${RUNNER}" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/resume_4gpu.log") 2>&1
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
run_one() {
  local seed="\$1" gpu="\$2" spec="\$3" task env instruction dir count log
  IFS='|' read -r task env instruction <<< "\${spec}"
  dir="\${OUT_ROOT}/seed\${seed}/result/\${env}_intact_object_ood_consensus_medoid_seed\${seed}"
  count=\$(find "\${dir}" -name '*.mp4' 2>/dev/null | wc -l)
  if (( count >= 24 )); then printf '[%s] skip seed=%s task=%s (%s/24)\n' "\$(date -Is)" "\${seed}" "\${task}" "\${count}"; return 0; fi
  log="\${OUT_ROOT}/seed\${seed}/logs/\${task}_gpu\${gpu}.log"; mkdir -p "\$(dirname "\${log}")"
  for attempt in 1 2; do
    printf '[%s] seed=%s gpu=%s task=%s attempt=%s\n' "\$(date -Is)" "\${seed}" "\${gpu}" "\${task}" "\${attempt}" | tee -a "\${log}"
    if OUT_ROOT="\${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" bash scripts/eval/run_mimic_intact_object_ood_consensus_medoid_task.sh "\${seed}" "\${task}" "\${env}" "\${instruction}" "\${OUT_ROOT}" "\${gpu}" >> "\${log}" 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}
for seed in 0 1 2; do
  pids=(); status=0
  for worker in 0 1 2 3; do
    gpu=\$((worker + 4))
    (
      worker_status=0
      for ((idx=worker; idx<\${#TASKS[@]}; idx+=4)); do run_one "\${seed}" "\${gpu}" "\${TASKS[idx]}" || worker_status=1; done
      exit "\${worker_status}"
    ) & pids[\${worker}]=\$!
  done
  for worker in 0 1 2 3; do wait "\${pids[\${worker}]}" || status=1; done
  "${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "\${OUT_ROOT}" || true
  printf '[%s] seed=%s workers_done status=%s\n' "\$(date -Is)" "\${seed}" "\${status}"
done
"${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "\${OUT_ROOT}"
printf '[%s] Complete\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
