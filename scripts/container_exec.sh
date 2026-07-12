#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SELFUPDATE_SIF:-$ROOT/containers/pytorch-2.11.0-cu128-cudnn9-runtime.sif}"
OVERLAY="${SELFUPDATE_OVERLAY:-$ROOT/containers/selfupdate-python-deps-cu128.sqsh}"
DEV_PYTHON_HOST="${SELFUPDATE_DEV_PYTHON_HOST:-/tmp/$USER/selfupdate-dev-python}"
DEV_PYTHON_CONTAINER="/dev-python"
# Model snapshots are shared at the account cache.  Do not create a second,
# incomplete Hugging Face cache under the checkout merely because the repo is
# mounted at /work inside Singularity.
HF_CACHE_HOST="${SELFUPDATE_HF_CACHE_HOST:-$HOME/.cache/huggingface}"
HF_CACHE_CONTAINER="/hf-cache"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/$USER/singularity-cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/$USER/singularity-tmp}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
CONTAINER_HOME="${CONTAINER_HOME:-/tmp/$USER/selfupdate-home}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$TMPDIR" "$CONTAINER_HOME" "$DEV_PYTHON_HOST" "$HF_CACHE_HOST"

overlay_args=()
if [[ -f "$OVERLAY" ]]; then
  overlay_args=(--overlay "$OVERLAY")
fi

exec singularity exec --nv \
  --cleanenv \
  "${overlay_args[@]}" \
  --home "$CONTAINER_HOME:/home/$USER" \
  --env PYTHONPATH="$DEV_PYTHON_CONTAINER:/opt/selfupdate-python:/work/src" \
  --env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  --env SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-8}" \
  --env HF_HOME="$HF_CACHE_CONTAINER" \
  --env TRANSFORMERS_CACHE="$HF_CACHE_CONTAINER" \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env XDG_CACHE_HOME=/tmp/xdg-cache \
  --bind "$DEV_PYTHON_HOST:$DEV_PYTHON_CONTAINER" \
  --bind "$HF_CACHE_HOST:$HF_CACHE_CONTAINER" \
  --bind "$ROOT:/work" \
  "$IMAGE" \
  "$@"
