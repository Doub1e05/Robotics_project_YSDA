#!/usr/bin/env bash
# Create a portable source snapshot of all project repositories.
# Run this on the current server before copying the resulting directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${1:-${WORKSPACE_DIR}/transfer-bundle-$(date -u +%Y%m%dT%H%M%SZ)}"

REPOS=(LIBERO-PRO LIBERO-plus cosmos-policy mimic-video openvla-oft)

mkdir -p "${OUTPUT_DIR}/bundles"

for repo in "${REPOS[@]}"; do
  repo_dir="${WORKSPACE_DIR}/${repo}"
  if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "Missing Git repository: ${repo_dir}" >&2
    exit 1
  fi

  # --all includes local branches and commits that have not been pushed yet.
  git -C "${repo_dir}" bundle create "${OUTPUT_DIR}/bundles/${repo}.bundle" --all
  git -C "${repo_dir}" rev-parse HEAD >"${OUTPUT_DIR}/bundles/${repo}.head"
  git -C "${repo_dir}" branch --show-current >"${OUTPUT_DIR}/bundles/${repo}.branch"
done

cat >"${OUTPUT_DIR}/README.txt" <<EOF
Portable source snapshot created at $(date -u +%FT%TZ).

Copy this directory to the target host, then run:
  git clone <bundle-dir>/bundles/mimic-video.bundle <workspace>/mimic-video
  <workspace>/mimic-video/scripts/migration/restore_workspace_from_bundle.sh <bundle-dir>

This snapshot contains Git history and all local commits, but intentionally does
not contain checkpoints, datasets, Hugging Face caches, videos, logs, or raw
experiment traces. Transfer those separately with rsync; see
mimic-video/docs/SERVER_MIGRATION.md in the restored workspace.
EOF

echo "Created portable source snapshot: ${OUTPUT_DIR}"
