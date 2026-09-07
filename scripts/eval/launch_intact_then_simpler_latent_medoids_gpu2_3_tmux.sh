#!/usr/bin/env bash
# Resume the interrupted INT-ACT fixed-candidate evaluation, then evaluate four
# distinct K=3 planners on the full 96-rollout SIMPLER-Bridge protocol.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_intact_then_simpler_latent_medoids_gpu2_3}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
INTACT_RUNNER="${INTACT_RUNNER:-${REPO_ROOT}/eval_outputs/intact_simpler/mimic_video_object_ood_consensus_8tasks_candidates_2_996_997_gpu5_20260907_103910/run_candidate_sets_gpu2_gpu3_20260907_150147.sh}"
SIMPLER_ROOT="${SIMPLER_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/latent_medoid_four_selectors_fixedseeds_2_996_997_${DATE_TAG}}"
PYTHON="${PYTHON:-${REPO_ROOT}/model/.venv/bin/python}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt}"

SIMPLER_ROOT="$(realpath -m "${SIMPLER_ROOT}")"
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${EMBEDDINGS}" ]] || { echo "Missing T5 embeddings: ${EMBEDDINGS}" >&2; exit 1; }
[[ -f "${INTACT_RUNNER}" ]] || { echo "Missing INT-ACT resume runner: ${INTACT_RUNNER}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 1; }

mkdir -p "${SIMPLER_ROOT}/logs"
RUNNER="${SIMPLER_ROOT}/run_all.sh"
tee "${RUNNER}" > /dev/null <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
exec > >(tee -a "${SIMPLER_ROOT}/logs/orchestrator.log") 2>&1

printf '[%s] Resuming interrupted INT-ACT fixed-candidate sequence\n' "\$(date -Is)"
bash "${INTACT_RUNNER}"
printf '[%s] INT-ACT sequence complete; starting SIMPLER-Bridge selectors\n' "\$(date -Is)"

run_arm() {
  local selector="\$1" arm_root="${SIMPLER_ROOT}/\$1"
  mkdir -p "\${arm_root}/logs"
  printf '[%s] selector=%s start\n' "\$(date -Is)" "\${selector}"
  MIMIC_VIDEO_CANDIDATE_SEEDS="2,996,997" OUT_ROOT="\${arm_root}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
    bash scripts/eval/run_mimic_simpler_bridge_consensus_selector_rank.sh 0 2 "\${selector}" > "\${arm_root}/logs/rank0_gpu2.log" 2>&1 &
  local pid2=\$!
  MIMIC_VIDEO_CANDIDATE_SEEDS="2,996,997" OUT_ROOT="\${arm_root}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
    bash scripts/eval/run_mimic_simpler_bridge_consensus_selector_rank.sh 1 3 "\${selector}" > "\${arm_root}/logs/rank1_gpu3.log" 2>&1 &
  local pid3=\$!
  local status=0
  wait "\${pid2}" || status=1
  wait "\${pid3}" || status=1
  (( status == 0 )) || { echo "selector \${selector} failed" >&2; return "\${status}"; }
  "${PYTHON}" scripts/eval/collect_simpler_bridge_sr.py --out-root "\${arm_root}"
  printf '[%s] selector=%s complete\n' "\$(date -Is)" "\${selector}"
}

run_arm encoder_latent_medoid
run_arm encoder_latent_medoid_robust
run_arm decoder_action_token_medoid
run_arm action_consensus
printf '[%s] All requested SIMPLER-Bridge evaluations complete\n' "\$(date -Is)"
EOF2
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n "intact_then_simpler" "bash ${RUNNER}"
printf 'Started tmux session: %s\nSIMPLER output root: %s\nAttach: tmux attach -t %s\n' "${SESSION}" "${SIMPLER_ROOT}" "${SESSION}"
