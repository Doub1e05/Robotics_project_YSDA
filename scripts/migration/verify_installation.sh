#!/usr/bin/env bash
# Fast non-destructive checks after migration; does not download checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_CMD="${CONDA_CMD:-conda}"

for repo in LIBERO-PRO LIBERO-plus cosmos-policy mimic-video openvla-oft; do
  echo "== ${repo} =="
  git -C "${WORKSPACE_DIR}/${repo}" status --short --branch
done

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

"${CONDA_CMD}" run -n openvla-oft python - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA in openvla-oft environment'
print('openvla-oft CUDA:', torch.cuda.get_device_name(0))
PY

"${WORKSPACE_DIR}/mimic-video/model/.venv/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA in mimic-video environment'
print('mimic-video CUDA:', torch.cuda.get_device_name(0))
PY

PYTHONPATH="${WORKSPACE_DIR}/mimic-video/eval/libero" \
  "${WORKSPACE_DIR}/mimic-video/model/.venv/bin/python" - <<'PY'
import numpy as np
from consensus_medoid import consensus_medoid_costs

chunk = np.zeros((5, 10)); chunk[:, 3] = 1; chunk[:, 7] = 1
costs, _, _ = consensus_medoid_costs([chunk, chunk.copy()], previous_action=None, horizon=5)
assert costs == [0.0, 0.0]
print('consensus-medoid import: OK')
PY

echo "Verification passed. Checkpoint and asset locations are documented in SERVER_MIGRATION.md."
