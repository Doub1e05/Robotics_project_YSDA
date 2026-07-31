#!/usr/bin/env bash
# Create the two project environments. Check CUDA / PyTorch compatibility before use.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USER_HOME="${HOME:?HOME must be set}"
CONDA_CMD="${CONDA_CMD:-conda}"
MIMIC_EXTRA="${MIMIC_EXTRA:-cu126}"

command -v "${CONDA_CMD}" >/dev/null 2>&1 || {
  echo "conda is required. Run bootstrap_host.sh and install Miniforge/Miniconda first." >&2
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  echo "uv is required. Run bootstrap_host.sh first." >&2
  exit 1
}

if ! "${CONDA_CMD}" env list | awk '{print $1}' | grep -qx openvla-oft; then
  "${CONDA_CMD}" create -y -n openvla-oft python=3.10
fi

"${CONDA_CMD}" run -n openvla-oft python -m pip install --upgrade pip
"${CONDA_CMD}" run -n openvla-oft python -m pip install torch torchvision torchaudio
"${CONDA_CMD}" run -n openvla-oft python -m pip install -e "${WORKSPACE_DIR}/openvla-oft"
"${CONDA_CMD}" run -n openvla-oft python -m pip install -e "${WORKSPACE_DIR}/LIBERO-PRO"

# LIBERO-plus is optional and shares the OpenVLA environment. Install it only
# after its external assets have been copied into libero/libero/assets.
if [[ "${INSTALL_LIBERO_PLUS:-0}" == "1" ]]; then
  "${CONDA_CMD}" run -n openvla-oft python -m pip install -r "${WORKSPACE_DIR}/LIBERO-plus/requirements.txt"
  "${CONDA_CMD}" run -n openvla-oft python -m pip install -e "${WORKSPACE_DIR}/LIBERO-plus"
fi

cd "${WORKSPACE_DIR}/mimic-video/model"
uv sync --extra "${MIMIC_EXTRA}"

"${CONDA_CMD}" run -n openvla-oft python - <<'PY'
import torch
print(f'OpenVLA environment: torch={torch.__version__}, cuda={torch.cuda.is_available()}')
PY

"${WORKSPACE_DIR}/mimic-video/model/.venv/bin/python" - <<'PY'
import torch
print(f'Mimic environment: torch={torch.__version__}, cuda={torch.cuda.is_available()}')
PY

mkdir -p "${USER_HOME}/.cache/huggingface"
echo "Environment setup completed. Run mimic-video/scripts/migration/verify_installation.sh next."
