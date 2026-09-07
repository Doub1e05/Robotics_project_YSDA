#!/usr/bin/env bash
# Launch full INT-ACT Object Diversity evaluation: 16 tasks x 24 episodes x 3 seeds.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_object_ood_3seed}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_${DATE_TAG}}"
GPU_LIST="${GPUS:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_LIST}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then echo "tmux session already exists: ${SESSION}" >&2; exit 1; fi
(( ${#GPUS[@]} == 4 )) || { echo "Need exactly four GPUs" >&2; exit 2; }
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/t5_embeddings"
RUNNER="${OUT_ROOT}/run_all.sh"
tee "${RUNNER}" >/dev/null <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${OUT_ROOT}/logs/orchestrator.log") 2>&1
T5_DIR="${REPO_ROOT}/model/checkpoints/text_encoder/t5-11b"
EMBEDDINGS="${OUT_ROOT}/t5_embeddings/intact_object_ood.pt"
printf '[%s] Downloading T5-11B if needed\\n' "\$(date -Is)"
bash scripts/eval/download_t5_11b_for_simpler_bridge.sh
if [[ ! -f "\${EMBEDDINGS}" ]]; then
  printf '[%s] Computing 16 Object OOD prompt embeddings on CPU\\n' "\$(date -Is)"
  "${PYTHON}" scripts/eval/precompute_intact_object_ood_t5_embeddings.py --output "\${EMBEDDINGS}" --t5-dir "\${T5_DIR}"
else
  printf '[%s] Reusing existing embeddings %s\\n' "\$(date -Is)" "\${EMBEDDINGS}"
fi
for seed in 0 1 2; do
  mkdir -p "${OUT_ROOT}/seed${seed}/logs" "${OUT_ROOT}/seed${seed}/result"
  printf '[%s] Starting seed %s on GPUs ${GPUS[*]}\\n' "\$(date -Is)" "\${seed}"
  pids=()
  for rank in 0 1 2 3; do
    gpu=\${GPUS[\${rank}]}
    OUT_ROOT="${OUT_ROOT}" T5_EMBEDDINGS="\${EMBEDDINGS}" \\
      bash scripts/eval/run_mimic_intact_object_ood_rank.sh "\${rank}" "\${gpu}" "\${seed}" "${OUT_ROOT}" > "${OUT_ROOT}/seed\${seed}/logs/rank\${rank}_gpu\${gpu}.log" 2>&1 &
    pids[\${rank}]=\$!
  done
  status=0
  for rank in 0 1 2 3; do
    if ! wait "\${pids[\${rank}]}"; then echo "seed \${seed} rank \${rank} failed" >&2; status=1; fi
  done
  "${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "${OUT_ROOT}" || status=1
  (( status == 0 )) || exit "\${status}"
done
printf '[%s] Final summary\\n' "\$(date -Is)"
"${PYTHON}" scripts/eval/collect_intact_object_ood_sr.py --out-root "${OUT_ROOT}"
printf '[%s] Complete\\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n object_ood "bash ${RUNNER}"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${OUT_ROOT}" "${SESSION}"
