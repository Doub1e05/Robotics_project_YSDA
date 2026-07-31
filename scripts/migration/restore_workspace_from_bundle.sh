#!/usr/bin/env bash
# Restore all repositories from a directory created by export_workspace_bundle.sh.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/transfer-bundle-YYYYMMDDTHHMMSSZ" >&2
  exit 2
fi

BUNDLE_DIR="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPOS=(LIBERO-PRO LIBERO-plus cosmos-policy mimic-video openvla-oft)

for repo in "${REPOS[@]}"; do
  bundle="${BUNDLE_DIR}/bundles/${repo}.bundle"
  head_file="${BUNDLE_DIR}/bundles/${repo}.head"
  branch_file="${BUNDLE_DIR}/bundles/${repo}.branch"
  target="${WORKSPACE_DIR}/${repo}"

  [[ -f "${bundle}" && -f "${head_file}" && -f "${branch_file}" ]] || {
    echo "Incomplete bundle for ${repo}" >&2
    exit 1
  }
  branch="$(<"${branch_file}")"
  commit="$(<"${head_file}")"
  if [[ -e "${target}" ]]; then
    # The bootstrap flow restores mimic-video first to access this script.
    if [[ "${repo}" != "mimic-video" || ! -d "${target}/.git" ]]; then
      echo "Refusing to overwrite existing path: ${target}" >&2
      exit 1
    fi
    if [[ -n "${branch}" ]]; then
      git -C "${target}" checkout -B "${branch}" "${commit}"
    else
      git -C "${target}" checkout --detach "${commit}"
    fi
  else
    git clone "${bundle}" "${target}"
    if [[ -n "${branch}" ]]; then
      git -C "${target}" checkout -B "${branch}" "${commit}"
    else
      git -C "${target}" checkout --detach "${commit}"
    fi
  fi
done

echo "Repositories restored to: ${WORKSPACE_DIR}"
echo "Next: run mimic-video/scripts/migration/setup_python_environments.sh"
