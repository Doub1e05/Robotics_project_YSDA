#!/usr/bin/env bash
# Download the exact T5-11B checkpoint used by mimic-video, with resumable range parts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${T5_DIR:-${REPO_ROOT}/model/checkpoints/text_encoder/t5-11b}"
TARGET_FILE="${TARGET_DIR}/pytorch_model.bin"
EXPECTED_BYTES=45229452544
PARTS="${T5_DOWNLOAD_PARTS:-16}"
PART_DIR="${TARGET_DIR}/.pytorch_model.parts"
URL="https://huggingface.co/jonpai/mimic-video/resolve/main/text_encoder/t5-11b/pytorch_model.bin?download=true"

mkdir -p "${TARGET_DIR}" "${PART_DIR}"
for file in config.json tokenizer.json README.md; do
  if [[ ! -s "${TARGET_DIR}/${file}" ]]; then
    curl -fsSL --retry 10 --retry-delay 10 \
      "https://huggingface.co/jonpai/mimic-video/resolve/main/text_encoder/t5-11b/${file}?download=true" \
      -o "${TARGET_DIR}/${file}"
  fi
done

if [[ -f "${TARGET_FILE}" ]] && [[ "$(stat -c '%s' "${TARGET_FILE}")" -eq "${EXPECTED_BYTES}" ]]; then
  echo "T5-11B already complete: ${TARGET_FILE}"
  exit 0
fi

download_part() {
  local part="$1"
  local chunk=$(((EXPECTED_BYTES + PARTS - 1) / PARTS))
  local start=$((part * chunk))
  local end=$((start + chunk - 1))
  local existing=0
  local path="${PART_DIR}/part.${part}"
  if (( end >= EXPECTED_BYTES )); then end=$((EXPECTED_BYTES - 1)); fi
  if [[ -f "${path}" ]]; then existing=$(stat -c '%s' "${path}"); fi
  if (( existing == end - start + 1 )); then return; fi
  if (( existing > end - start + 1 )); then rm -f "${path}"; existing=0; fi
  curl -fL --retry 20 --retry-delay 10 --continue-at - \
    --range "$((start + existing))-${end}" "${URL}" -o "${path}"
  [[ "$(stat -c '%s' "${path}")" -eq "$((end - start + 1))" ]]
}

export EXPECTED_BYTES PARTS PART_DIR URL
export -f download_part
seq 0 "$((PARTS - 1))" | xargs -n 1 -P "${PARTS}" bash -c 'download_part "$0"'

: > "${TARGET_FILE}.tmp"
for part in $(seq 0 "$((PARTS - 1))"); do
  cat "${PART_DIR}/part.${part}" >> "${TARGET_FILE}.tmp"
done
[[ "$(stat -c '%s' "${TARGET_FILE}.tmp")" -eq "${EXPECTED_BYTES}" ]]
mv "${TARGET_FILE}.tmp" "${TARGET_FILE}"
echo "T5-11B download complete: ${TARGET_FILE}"
