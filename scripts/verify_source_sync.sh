#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
REQUIRE_CLEAN_WORKTREE="${REQUIRE_CLEAN_WORKTREE:-0}"
MANIFEST="${SOURCE_MANIFEST:-$ROOT/env/experiment_source.sha256}"

cd "$ROOT"

echo "project_root=$ROOT"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_commit="$(git rev-parse HEAD)"
  echo "git_commit=$actual_commit"
  if [ -n "$EXPECTED_COMMIT" ] && [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
    echo "[FAIL] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
    exit 2
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "[WARN] git worktree is not clean"
    git status --short
    if [ "$REQUIRE_CLEAN_WORKTREE" = "1" ]; then
      echo "[FAIL] clean worktree is required for this experiment" >&2
      exit 2
    fi
  else
    echo "[OK] git worktree is clean"
  fi
else
  echo "[WARN] git metadata is unavailable; relying on SHA-256 manifest"
fi

if [ ! -f "$MANIFEST" ]; then
  echo "[FAIL] source manifest not found: $MANIFEST" >&2
  exit 3
fi

sha256sum -c "$MANIFEST"
echo "[OK] experiment source files match $MANIFEST"
