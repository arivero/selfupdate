#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SELFUPDATE_SIF:-$ROOT/containers/pytorch-2.11.0-cu128-cudnn9-runtime.sif}"
OVERLAY="${SELFUPDATE_OVERLAY:-$ROOT/containers/selfupdate-python-deps-cu128.sqsh}"
DEV_PYTHON_HOST="${SELFUPDATE_DEV_PYTHON_HOST:-/tmp/$USER/selfupdate-dev-python}"
DEV_PYTHON_CONTAINER="/dev-python"
# Prefer an explicitly completed node-local snapshot stage.  Otherwise use
# the account cache; never create an accidental third cache under /work.
HF_STAGE_HOST="${SELFUPDATE_HF_STAGE:-/tmp/$USER/selfupdate-hf-cache}"
if [[ -n "${SELFUPDATE_HF_CACHE_HOST:-}" ]]; then
  HF_CACHE_HOST="$SELFUPDATE_HF_CACHE_HOST"
elif [[ -f "$HF_STAGE_HOST/.selfupdate-hf-stage-ready" ]]; then
  HF_CACHE_HOST="$HF_STAGE_HOST"
else
  HF_CACHE_HOST="$HOME/.cache/huggingface"
fi
HF_CACHE_CONTAINER="/hf-cache"

export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/tmp/$USER/singularity-cache}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/$USER/singularity-tmp}"
export TMPDIR="${TMPDIR:-/tmp/$USER/tmp}"
CONTAINER_HOME="${CONTAINER_HOME:-/tmp/$USER/selfupdate-home}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/torchinductor-cache}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$TMPDIR" "$CONTAINER_HOME" "$DEV_PYTHON_HOST" "$HF_CACHE_HOST"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

overlay_args=()
if [[ -f "$OVERLAY" ]]; then
  overlay_args=(--overlay "$OVERLAY")
fi
device_env=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Singularity --cleanenv otherwise drops the scheduler's physical device
  # selection and makes concurrent single-card jobs collide on cuda:0.
  device_env=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

exec singularity exec --nv \
  --cleanenv \
  --pwd /work \
  "${overlay_args[@]}" \
  "${device_env[@]}" \
  --home "$CONTAINER_HOME:/home/$USER" \
  --env PYTHONPATH="$DEV_PYTHON_CONTAINER:/opt/selfupdate-python:/work/src" \
  --env PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  --env SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-8}" \
  --env HF_HOME="$HF_CACHE_CONTAINER" \
  --env TRANSFORMERS_CACHE="$HF_CACHE_CONTAINER" \
  --env HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}" \
  --env TQDM_DISABLE="${TQDM_DISABLE:-1}" \
  --env TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env XDG_CACHE_HOME=/tmp/xdg-cache \
  --bind "$DEV_PYTHON_HOST:$DEV_PYTHON_CONTAINER" \
  --bind "$HF_CACHE_HOST:$HF_CACHE_CONTAINER" \
  --bind "$ROOT:/work" \
  "$IMAGE" \
  "$@"
