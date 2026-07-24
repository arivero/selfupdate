#!/usr/bin/env bash
# Stage selected immutable teacher-cache directories into node-local tmpfs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SELFUPDATE_TEACHER_CACHE_SOURCE:-$ROOT/runs/teacher_cache_h100/artifacts_exact_full}"
SHM_USER="$(id -un)"
DEST_ROOT="${SELFUPDATE_TEACHER_CACHE_DEST:-/dev/shm/$SHM_USER/selfupdate-teacher-cache}"
shifted=0

if [[ "$#" -eq 0 ]]; then
  set -- Qwen3.5-4B-rag_system-remove-885f57b6f4eb9221
fi
# Validate every source name before touching SHM or collecting stale entries.
for name in "$@"; do
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid teacher cache name: $name" >&2; exit 2; }
  [[ -f "$SOURCE_ROOT/$name/index.json" ]] || { echo "missing teacher cache: $SOURCE_ROOT/$name" >&2; exit 2; }
done
if [[ "${SELFUPDATE_SHM_LEASE_GC:-0}" == 1 ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "REFUSED: SHM GC requires Slurm allocation" >&2; exit 2; }
  mkdir -p "/dev/shm/$SHM_USER"
  exec 8>"/dev/shm/$SHM_USER/.selfupdate-shm-stage.lock"
  flock 8
  lease_args=()
  for name in "$@"; do lease_args+=(--path "$DEST_ROOT/$name"); done
  "$ROOT/scripts/shm_lease_gc.sh" claim "${lease_args[@]}"
  SELFUPDATE_SHM_STAGE_LOCK_HELD=1 "$ROOT/scripts/shm_lease_gc.sh" gc
fi
[[ ! -L "$DEST_ROOT" ]] || { echo "REFUSED: teacher SHM root is a symlink" >&2; exit 2; }
mkdir -p "$DEST_ROOT"
rm -f "$DEST_ROOT/.selfupdate-teacher-stage-ready"
for name in "$@"; do
  [[ ! -L "$DEST_ROOT/$name" ]] || { echo "REFUSED: teacher cache target is a symlink" >&2; exit 2; }
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
