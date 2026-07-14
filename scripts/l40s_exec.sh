#!/usr/bin/env bash
# Driver-560 L40S runtime: reuse the existing cu126 venv and shadow only the
# pure-Python trainer packages from a node-local layer. Never install torch in
# this layer. Build it with scripts/l40s_setup.sh through a delegated launch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${SELFUPDATE_L40S_PYTHON:-$ROOT/../jacobian-lens/.venv/bin/python}"
DEPS="${SELFUPDATE_L40S_DEPS:-/tmp/$USER/selfupdate-l40-python}"
SHM_HF="/dev/shm/$USER/selfupdate-hf-cache"

[[ -x "$BASE_PYTHON" ]] || { echo "missing cu126 Python: $BASE_PYTHON" >&2; exit 2; }
[[ -d "$DEPS/transformers" && -d "$DEPS/peft" ]] || {
  echo "missing L40S dependency layer: run scripts/l40s_setup.sh" >&2
  exit 2
}
if find "$DEPS" -maxdepth 1 -iname 'torch*' -print -quit | grep -q .; then
  echo "refusing L40S layer containing torch; rebuild without dependencies" >&2
  exit 2
fi
[[ -f "$SHM_HF/.selfupdate-hf-stage-ready" ]] || {
  echo "RAM-backed HF cache is not ready: $SHM_HF" >&2
  exit 2
}

export SELFUPDATE_DISABLE_CAUSAL_CONV1D=1
export PYTHONPATH="$ROOT/runtime/l40s:$DEPS:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SHM_HF"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec "$BASE_PYTHON" "$@"
