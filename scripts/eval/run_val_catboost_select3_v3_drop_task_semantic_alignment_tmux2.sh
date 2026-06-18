#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="mimic_val30_select3_v3_no_tsa"
RUNNER="${REPO_ROOT}/scripts/eval/run_val_catboost_select3_v3_drop_task_semantic_alignment_rank.sh"

chmod +x "$RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n "gpu2" "bash '$RUNNER' 0 2; echo DONE rank0; bash"
tmux new-window -t "$SESSION" -n "gpu3" "bash '$RUNNER' 1 3; echo DONE rank1; bash"
tmux new-window -t "$SESSION" -n "gpu4" "bash '$RUNNER' 2 4; echo DONE rank2; bash"

echo "Started tmux: $SESSION (GPUs 2,3,4 -> ranks 0,1,2)"
echo "Attach: tmux attach -t $SESSION"
