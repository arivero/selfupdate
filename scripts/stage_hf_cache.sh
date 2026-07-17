#!/usr/bin/env bash
# Stage selected Hugging Face snapshots from the account cache to node-local
# /tmp or RAM-backed /dev/shm. Safe to rerun: rsync resumes partial copies; the ready marker is
# written only after every requested model has copied successfully.
set -euo pipefail

SOURCE="${SELFUPDATE_HF_SOURCE:-$HOME/.cache/huggingface}"
if [[ "${1:-}" == "--shm" ]]; then
  DEST="${SELFUPDATE_HF_STAGE:-/dev/shm/$USER/selfupdate-hf-cache}"
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
