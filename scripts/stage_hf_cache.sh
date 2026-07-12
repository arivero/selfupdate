#!/usr/bin/env bash
# Stage selected Hugging Face snapshots from the account cache to node-local
# /tmp.  Safe to rerun: rsync resumes partial copies; the ready marker is
# written only after every requested model has copied successfully.
set -euo pipefail

SOURCE="${SELFUPDATE_HF_SOURCE:-$HOME/.cache/huggingface}"
DEST="${SELFUPDATE_HF_STAGE:-/tmp/$USER/selfupdate-hf-cache}"
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(Qwen3-0.6B Qwen3-1.7B Qwen3-4B)
fi

for model in "${MODELS[@]}"; do
  [[ -d "$SOURCE/hub/models--Qwen--$model" ]] || {
    echo "missing source snapshot: $SOURCE/hub/models--Qwen--$model" >&2
    exit 2
  }
done

mkdir -p "$DEST/hub"
exec 9>"$DEST/.stage.lock"
flock 9
rm -f "$DEST/.selfupdate-hf-stage-ready"
for model in "${MODELS[@]}"; do
  echo "staging Qwen/$model -> $DEST" >&2
  rsync -aH --partial "$SOURCE/hub/models--Qwen--$model" "$DEST/hub/"
done
touch "$DEST/.selfupdate-hf-stage-ready"
echo "ready: $DEST (container_exec.sh will now prefer this cache)"
