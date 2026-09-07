#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-mimic_simpler_bridge_latent_diag_30}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_outputs/simpler_bridge/latent_diagnostics_30_${DATE_TAG}}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/scripts/eval/simpler_bridge_latent_diagnostics_30_manifest.tsv}"
EMBEDDINGS="${T5_EMBEDDINGS:-${REPO_ROOT}/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt}"
[[ -f "$MANIFEST" && -f "$EMBEDDINGS" ]] || { echo "Missing manifest or embeddings" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session already exists: $SESSION" >&2; exit 1; fi
mkdir -p "$OUT_ROOT/logs"
RUNNER="$OUT_ROOT/run_all.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
exec > >(tee -a "$OUT_ROOT/logs/orchestrator.log") 2>&1
pids=()
for rank in 0 1 2 3; do
  gpu=\$((rank + 4))
  OUT_ROOT="$OUT_ROOT" T5_EMBEDDINGS="$EMBEDDINGS" \
    bash scripts/eval/run_mimic_simpler_bridge_latent_diagnostics_rank.sh "\$rank" "\$gpu" "$MANIFEST" > "$OUT_ROOT/logs/rank\${rank}_gpu\${gpu}.log" 2>&1 &
  pids[\$rank]=\$!
done
status=0
for rank in 0 1 2 3; do wait "\${pids[\$rank]}" || status=1; done
if (( status != 0 )); then exit \$status; fi
model/.venv/bin/python scripts/eval/collect_simpler_bridge_latent_diagnostics.py --out-root "$OUT_ROOT" --manifest "$MANIFEST"
EOF
chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" -n diagnostics "bash $RUNNER"
printf 'Started tmux session: %s\nOutput: %s\nAttach: tmux attach -t %s\n' "$SESSION" "$OUT_ROOT" "$SESSION"
