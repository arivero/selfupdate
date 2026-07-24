#!/usr/bin/env bash
# Stage selected immutable teacher-cache directories into node-local tmpfs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SELFUPDATE_TEACHER_CACHE_SOURCE:-$ROOT/runs/teacher_cache_h100/artifacts_exact_full}"
DEST_ROOT="${SELFUPDATE_TEACHER_CACHE_DEST:-/dev/shm/$USER/selfupdate-teacher-cache}"
shifted=0

mkdir -p "$DEST_ROOT"
rm -f "$DEST_ROOT/.selfupdate-teacher-stage-ready"
if [[ "$#" -eq 0 ]]; then
  set -- Qwen3.5-4B-rag_system-remove-885f57b6f4eb9221
fi
for name in "$@"; do
  [[ -f "$SOURCE_ROOT/$name/index.json" ]] || {
    echo "missing teacher cache: $SOURCE_ROOT/$name" >&2
    exit 2
  }
  source_bytes=$(du -sb "$SOURCE_ROOT/$name" | awk '{print $1}')
  existing_bytes=0
  if [[ -d "$DEST_ROOT/$name" ]]; then
    existing_bytes=$(du -sb "$DEST_ROOT/$name" | awk '{print $1}')
  fi
  available_bytes=$(df -B1 --output=avail "$DEST_ROOT" | tail -n 1 | tr -d ' ')
  needed_bytes=$((source_bytes - existing_bytes))
  (( needed_bytes < 0 )) && needed_bytes=0
  if (( available_bytes < needed_bytes )); then
    echo "insufficient /dev/shm capacity for $name: need ${needed_bytes} bytes, have ${available_bytes} bytes" >&2
    exit 3
  fi
  mkdir -p "$DEST_ROOT/$name"
  rsync -a --delete "$SOURCE_ROOT/$name/" "$DEST_ROOT/$name/"
  cmp "$SOURCE_ROOT/$name/index.json" "$DEST_ROOT/$name/index.json"
  echo "staged $name bytes=$(du -sb "$DEST_ROOT/$name" | awk '{print $1}')"
  shifted=$((shifted + 1))
done
printf 'host=%s\nsource=%s\ncaches=%s\ncompleted=%s\n' \
  "$(hostname -s)" "$SOURCE_ROOT" "$shifted" "$(date -Is)" \
  > "$DEST_ROOT/.selfupdate-teacher-stage-ready"
echo "ready $DEST_ROOT caches=$shifted"
