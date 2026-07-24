#!/usr/bin/env bash
# Stage selected Hugging Face snapshots from the account cache to node-local
# /tmp or RAM-backed /dev/shm. Safe to rerun: rsync resumes partial copies; the ready marker is
# written only after every requested model has copied successfully.
set -euo pipefail

SOURCE="${SELFUPDATE_HF_SOURCE:-$HOME/.cache/huggingface}"
SHM_MODE=0
if [[ "${1:-}" == "--shm" ]]; then
  SHM_USER="$(id -un)"
  DEST="${SELFUPDATE_HF_STAGE:-/dev/shm/$SHM_USER/selfupdate-hf-cache}"
  SHM_MODE=1
  shift
else
  DEST="${SELFUPDATE_HF_STAGE:-/tmp/$USER/selfupdate-hf-cache}"
fi
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(Qwen3-0.6B Qwen3-1.7B Qwen3-4B)
fi

snapshot_dir() {
  local model="$1"
  if [[ "$model" == */* ]]; then
    printf 'models--%s\n' "${model//\//--}"
  else
    printf 'models--Qwen--%s\n' "$model"
  fi
}

for model in "${MODELS[@]}"; do
  snap="$(snapshot_dir "$model")"
  [[ -d "$SOURCE/hub/$snap" ]] || {
    echo "missing source snapshot: $SOURCE/hub/$snap" >&2
    exit 2
  }
done

if [[ "$SHM_MODE" -eq 1 && "${SELFUPDATE_SHM_LEASE_GC:-0}" == 1 ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "REFUSED: SHM GC requires Slurm allocation" >&2; exit 2; }
  SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  [[ ! -L "$DEST" ]] || { echo "REFUSED: HF SHM root is a symlink" >&2; exit 2; }
  mkdir -p "/dev/shm/$SHM_USER"
  exec 8>"/dev/shm/$SHM_USER/.selfupdate-shm-stage.lock"
  flock 8
  lease_args=()
  for model in "${MODELS[@]}"; do
    lease_args+=(--path "$DEST/hub/$(snapshot_dir "$model")")
  done
  "$SCRIPT_ROOT/scripts/shm_lease_gc.sh" claim "${lease_args[@]}"
  SELFUPDATE_SHM_STAGE_LOCK_HELD=1 "$SCRIPT_ROOT/scripts/shm_lease_gc.sh" gc
fi

mkdir -p "$DEST/hub"
exec 9>"$DEST/.stage.lock"
flock 9
rm -f "$DEST/.selfupdate-hf-stage-ready"
for model in "${MODELS[@]}"; do
  snap="$(snapshot_dir "$model")"
  echo "staging $model -> $DEST" >&2
  rsync -aH --partial "$SOURCE/hub/$snap" "$DEST/hub/"
done
touch "$DEST/.selfupdate-hf-stage-ready"
echo "ready: $DEST (point HF_HOME at this cache to use it)"
