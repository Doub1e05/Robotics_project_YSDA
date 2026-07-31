#!/usr/bin/env bash
# Install host packages needed for LIBERO, MuJoCo/EGL, Git and Python env setup.
# This script requires sudo access and is intended for Ubuntu/Debian hosts.
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as a regular user; it invokes sudo only where needed." >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl ffmpeg git git-lfs \
  libegl1 libegl1-mesa-dev libexpat1 libfontconfig1-dev libgl1 \
  libgl1-mesa-dev libglib2.0-0 libglfw3 libglfw3-dev libgles2 \
  libmagickwand-dev libosmesa6-dev libpython3-dev pkg-config \
  tmux unzip wget

git lfs install

if ! command -v conda >/dev/null 2>&1; then
  cat <<'EOF'
Conda is not installed. Install Miniforge or Miniconda, reopen the shell, then
run mimic-video/scripts/migration/setup_python_environments.sh.
Suggested installer: https://github.com/conda-forge/miniforge
EOF
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  echo "uv was installed. Reopen your shell if 'uv' is not yet on PATH."
fi

echo "Host bootstrap completed. Verify NVIDIA drivers with: nvidia-smi"
