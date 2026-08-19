#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="${LIBERO_DOWNLOAD_CACHE:-${ROOT_DIR}/.cache/downloads}"
ARCHIVE="${CACHE_DIR}/libero-plus-assets.zip"
TARGET="${ROOT_DIR}/LIBERO-plus/libero/libero"
URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip?download=true"

mkdir -p "${CACHE_DIR}" "${TARGET}"
echo "[$(date -Is)] downloading ${URL}"
curl -fL --retry 10 --retry-delay 5 --continue-at - "${URL}" -o "${ARCHIVE}"
echo "[$(date -Is)] validating archive"
unzip -t "${ARCHIVE}" >/dev/null
echo "[$(date -Is)] extracting into ${TARGET}"
unzip -q -o "${ARCHIVE}" -d "${TARGET}"
if [[ -d "${TARGET}/assets/assets" ]]; then
  shopt -s dotglob
  mv "${TARGET}/assets/assets"/* "${TARGET}/assets/"
  rmdir "${TARGET}/assets/assets"
  shopt -u dotglob
fi
echo "[$(date -Is)] done: $(du -sh "${TARGET}/assets" | cut -f1)"
