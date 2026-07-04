#!/usr/bin/env bash
# Restore deprecated files from archive/ back to their original paths.
#
# Data source: docs/superpowers/DEPRECATION-manifest.md — every table row
#   | <original path> | <archive path> | <reason> | <commit> |
# is one restore action: `git mv <archive path> <original path>`.
#
# Usage:
#   restore_from_archive.sh --dry-run        # print planned git mv, change nothing
#   restore_from_archive.sh [--all]          # restore every manifest entry
#   restore_from_archive.sh <orig-path>...   # restore only the named original paths
#
# Idempotent: an entry whose original path already exists (or whose archive path
# is missing) is skipped with a note.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
MANIFEST="${REPO_ROOT}/docs/superpowers/DEPRECATION-manifest.md"

dry_run=0
declare -a wanted=()
for arg in "$@"; do
  case "${arg}" in
    --dry-run) dry_run=1 ;;
    --all) : ;;
    -h|--help) grep -E '^#( |$)' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: ${arg}" >&2; exit 2 ;;
    *) wanted+=("${arg}") ;;
  esac
done

if [[ ! -f "${MANIFEST}" ]]; then
  echo "manifest not found: ${MANIFEST}" >&2
  exit 1
fi

# Parse manifest table rows: | orig | archive | reason | commit |
# Keep only rows whose 2nd column path starts with archive/ (skips the header).
restored=0
skipped=0
while IFS='|' read -r _ orig arch _rest; do
  orig="$(echo "${orig}" | xargs)"
  arch="$(echo "${arch}" | xargs)"
  [[ "${arch}" == archive/* ]] || continue

  if [[ ${#wanted[@]} -gt 0 ]]; then
    match=0
    for w in "${wanted[@]}"; do [[ "${w}" == "${orig}" ]] && match=1; done
    [[ ${match} -eq 1 ]] || continue
  fi

  if [[ -e "${REPO_ROOT}/${orig}" ]]; then
    echo "skip (already in place): ${orig}"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ ! -e "${REPO_ROOT}/${arch}" ]]; then
    echo "skip (archive missing): ${arch}" >&2
    skipped=$((skipped + 1))
    continue
  fi

  if [[ ${dry_run} -eq 1 ]]; then
    echo "git mv ${arch} ${orig}"
  else
    mkdir -p "${REPO_ROOT}/$(dirname "${orig}")"
    git -C "${REPO_ROOT}" mv "${arch}" "${orig}"
    echo "restored: ${arch} -> ${orig}"
  fi
  restored=$((restored + 1))
done < "${MANIFEST}"

if [[ ${dry_run} -eq 1 ]]; then
  echo "[dry-run] ${restored} restore action(s) planned, ${skipped} skipped."
else
  echo "restored ${restored} file(s), ${skipped} skipped."
fi
