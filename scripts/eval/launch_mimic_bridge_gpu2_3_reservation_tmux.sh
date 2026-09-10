#!/usr/bin/env bash
# Temporarily reserve GPU 2 and 3 with reproducible MIMIC-Video Bridge slices.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_bridge_gpu2_3_reservation}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/mimic_gpu2_3_reservation_${DATE_TAG}}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt}"

[[ -f "${EMBEDDINGS}" ]] || { echo "Missing T5 embeddings: ${EMBEDDINGS}" >&2; exit 1; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 1; }

mkdir -p "${OUT_ROOT}/logs"
RUNNER="${OUT_ROOT}/run_reservation.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
OUT_ROOT="${OUT_ROOT}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
  bash scripts/eval/run_mimic_simpler_bridge_ftcosmos_rank.sh 0 2 > "${OUT_ROOT}/logs/rank0_gpu2.log" 2>&1 &
pid2=\$!
OUT_ROOT="${OUT_ROOT}" T5_EMBEDDINGS="${EMBEDDINGS}" \\
  bash scripts/eval/run_mimic_simpler_bridge_ftcosmos_rank.sh 1 3 > "${OUT_ROOT}/logs/rank1_gpu3.log" 2>&1 &
pid3=\$!
wait "\${pid2}"
wait "\${pid3}"
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s "${SESSION}" -n bridge "bash ${RUNNER}"
printf 'session=%s\noutput=%s\n' "${SESSION}" "${OUT_ROOT}"
