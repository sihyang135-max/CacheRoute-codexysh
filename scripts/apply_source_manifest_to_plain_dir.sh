#!/usr/bin/env bash
# Copy the exact experiment source set from a Git checkout to a plain directory.
set -euo pipefail

SOURCE="${1:?usage: apply_source_manifest_to_plain_dir.sh <source_checkout> <target_dir>}"
TARGET="${2:?usage: apply_source_manifest_to_plain_dir.sh <source_checkout> <target_dir>}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"

SOURCE="$(cd "$SOURCE" && pwd)"
TARGET="$(cd "$TARGET" && pwd)"
MANIFEST="$SOURCE/env/experiment_source.sha256"
BACKUP_ROOT="${BACKUP_ROOT:-$(dirname "$TARGET")/backups/source_sync_$(date +%Y%m%d_%H%M%S)}"

if [ "$TARGET" = "/" ]; then
  echo "[FAIL] refusing to use / as the target" >&2
  exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "[FAIL] source manifest not found: $MANIFEST" >&2
  exit 3
fi

source_commit="unavailable"
if git -C "$SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  source_commit="$(git -C "$SOURCE" rev-parse HEAD)"
  if [ -n "$EXPECTED_COMMIT" ] && [ "$source_commit" != "$EXPECTED_COMMIT" ]; then
    echo "[FAIL] source commit mismatch: expected=$EXPECTED_COMMIT actual=$source_commit" >&2
    exit 4
  fi
elif [ -n "$EXPECTED_COMMIT" ]; then
  echo "[FAIL] EXPECTED_COMMIT was set but source has no Git metadata" >&2
  exit 4
fi

echo "===== Verify source checkout ====="
(
  cd "$SOURCE"
  sha256sum -c env/experiment_source.sha256
)

files=()
while read -r expected rel extra; do
  if [ -z "${expected:-}" ]; then
    continue
  fi
  if [ -n "${extra:-}" ] || [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[FAIL] invalid manifest row: $expected $rel ${extra:-}" >&2
    exit 5
  fi
  case "$rel" in
    ""|/*|../*|*/../*|*/..)
      echo "[FAIL] unsafe manifest path: $rel" >&2
      exit 5
      ;;
  esac
  if [ ! -f "$SOURCE/$rel" ]; then
    echo "[FAIL] source file missing: $rel" >&2
    exit 5
  fi
  files+=("$rel")
done < "$MANIFEST"
display_files=("env/experiment_source.sha256" "${files[@]}")

echo
echo "===== Files to synchronize ====="
for rel in "${display_files[@]}"; do
  if [ ! -e "$TARGET/$rel" ]; then
    status="NEW"
  elif cmp -s "$SOURCE/$rel" "$TARGET/$rel"; then
    status="IDENTICAL"
  else
    status="UPDATE"
  fi
  printf '%-10s %s\n' "$status" "$rel"
done

echo
read -r -p "Back up and synchronize this source set to $TARGET? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo "Canceled."
  exit 0
fi

mkdir -p "$BACKUP_ROOT"
for rel in "${display_files[@]}"; do
  if [ -e "$TARGET/$rel" ]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$rel")"
    cp -a "$TARGET/$rel" "$BACKUP_ROOT/$rel"
  fi
done
for rel in "${files[@]}"; do
  mkdir -p "$TARGET/$(dirname "$rel")"
  cp -a "$SOURCE/$rel" "$TARGET/$rel"
done
mkdir -p "$TARGET/env"
cp -a "$MANIFEST" "$TARGET/env/experiment_source.sha256"

printf 'source_commit=%s\nsource_checkout=%s\ntarget=%s\n' \
  "$source_commit" "$SOURCE" "$TARGET" > "$BACKUP_ROOT/source-sync.txt"

echo
echo "===== Verify synchronized target ====="
(
  cd "$TARGET"
  sha256sum -c env/experiment_source.sha256
)

echo "[DONE] source_commit=$source_commit"
echo "[DONE] backup=$BACKUP_ROOT"
