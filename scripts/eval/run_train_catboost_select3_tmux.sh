#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="mimic_train70_select3"
RUNNER="${REPO_ROOT}/scripts/eval/run_train_catboost_select3_rank.sh"

chmod +x "$RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n "gpu1" \
  "bash '$RUNNER' 0 1; echo DONE rank0; bash"
tmux new-window -t "$SESSION" -n "gpu2" \
  "bash '$RUNNER' 1 2; echo DONE rank1; bash"
tmux new-window -t "$SESSION" -n "gpu3" \
  "bash '$RUNNER' 2 3; echo DONE rank2; bash"

echo "Started tmux: $SESSION (GPUs 1,2,3 -> ranks 0,1,2)"
echo "Output: eval_outputs/libero_spatial/strategy_runs/catboost_select3_v2_train70_6sec/"
echo "Attach: tmux attach -t $SESSION"
