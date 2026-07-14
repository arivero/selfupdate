#!/usr/bin/env bash
# Driver-560 L40S runtime: reuse the existing cu126 venv and shadow only the
# pure-Python trainer packages from a node-local layer. Never install torch in
# this layer. Build it with scripts/l40s_setup.sh through a delegated launch.
set -euo pipefail

if [[ "${1:-}" == "python" || "${1:-}" == "python3" ]]; then
  echo "usage: scripts/l40s_exec.sh <script.py> [args...]" >&2
  echo "l40s_exec.sh already selects and launches Python; omit the extra '${1}'." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${SELFUPDATE_L40S_PYTHON:-$ROOT/../jacobian-lens/.venv/bin/python}"
DEPS="${SELFUPDATE_L40S_DEPS:-/tmp/$USER/selfupdate-l40-python}"
SHM_HF="/dev/shm/$USER/selfupdate-hf-cache"
SHM_TEACHER="/dev/shm/$USER/selfupdate-teacher-cache"

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

export PYTHONPATH="$ROOT/runtime/l40s:$DEPS:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SHM_HF"
if [[ -f "$SHM_TEACHER/.selfupdate-teacher-stage-ready" ]]; then
  export SELFUPDATE_TEACHER_CACHE_ROOT="$SHM_TEACHER"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# TorchInductor otherwise creates min(32, CPU count) compiler workers per
# trainer. Four concurrent L40S arms produced ~128 workers and starved the
# shape-varying Qwen3.5 training walk while GPUs waited for compilation.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/selfupdate-torchinductor}"

# The host glibc is older than the compiled causal-conv1d wheel.  Lmod's
# glibc module must be entered through its dynamic loader; `module load` alone
# mixes loaders and fails with a GLIBC_PRIVATE symbol error.  Keep the slower
# torch implementation available only as an explicit diagnostic escape hatch.
CAUSAL_BACKEND="${SELFUPDATE_L40S_CAUSAL_CONV:-compiled}"
if [[ "$CAUSAL_BACKEND" == "torch" ]]; then
  export SELFUPDATE_DISABLE_CAUSAL_CONV1D=1
  export SELFUPDATE_CAUSAL_CONV_BACKEND=torch
  exec "$BASE_PYTHON" "$@"
fi
[[ "$CAUSAL_BACKEND" == "compiled" ]] || {
  echo "SELFUPDATE_L40S_CAUSAL_CONV must be compiled or torch" >&2
  exit 2
}

OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
set +u
source /etc/profile.d/lmod.sh
set -u
module load glibc/2.35 >/dev/null
[[ -x "${GLIB235_LINUX_SO:-}" && -d "${GLIB235_LIB:-}" ]] || {
  echo "glibc/2.35 module did not expose its loader and library directory" >&2
  exit 2
}
LIBRARY_PATH="$GLIB235_LIB:/lib64:/usr/lib64"
[[ -z "$OLD_LD_LIBRARY_PATH" ]] || LIBRARY_PATH="$LIBRARY_PATH:$OLD_LD_LIBRARY_PATH"
# `--library-path` configures the glibc-2.35 loader for this Python process.
# Do not export the module's LD_LIBRARY_PATH: subprocesses such as Triton's
# host gcc start through the host loader and must see the original host paths.
if [[ -n "$OLD_LD_LIBRARY_PATH" ]]; then
  export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
else
  unset LD_LIBRARY_PATH
fi
unset SELFUPDATE_DISABLE_CAUSAL_CONV1D
export SELFUPDATE_CAUSAL_CONV_BACKEND=compiled
exec "$GLIB235_LINUX_SO" --library-path "$LIBRARY_PATH" "$BASE_PYTHON" "$@"
