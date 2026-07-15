#!/usr/bin/env bash
# Warm a Lustre-hosted Python runtime into this node's VFS/page cache.
#
# This does not copy or mutate a venv. It parallel-stats the venv and base
# standard-library trees, then imports the requested modules once. Subsequent
# workers on the same node avoid minutes of serial importlib.metadata round
# trips over loose Lustre files.
set -euo pipefail

PYTHON_BIN="${1:?usage: warm_python_runtime.sh PYTHON [MODULE ...]}"
PYTHON_LAUNCHER="$PYTHON_BIN"
shift
[[ -x "$PYTHON_BIN" ]] || { echo "not executable: $PYTHON_BIN" >&2; exit 2; }
MODULES=("$@")
if [[ ${#MODULES[@]} -eq 0 ]]; then
  MODULES=(torch transformers)
fi
WORKERS="${SELFUPDATE_PYTHON_WARM_WORKERS:-16}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "SELFUPDATE_PYTHON_WARM_WORKERS must be positive" >&2
  exit 2
}

PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
BASE_PREFIX="$($PYTHON_BIN -c 'import sys; print(sys.base_prefix)')"
VENV_ROOT="$(cd "$(dirname "$PYTHON_LAUNCHER")/.." 2>/dev/null && pwd || true)"
# The resolved executable can live in a shared uv base while its site-packages
# live beside the original venv launcher. Recover that original root as well.
if [[ -z "$VENV_ROOT" || ! -d "$VENV_ROOT/lib" ]]; then
  VENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

roots=()
[[ -d "$VENV_ROOT/lib" ]] && roots+=("$VENV_ROOT/lib")
[[ -d "$BASE_PREFIX/lib" && "$BASE_PREFIX/lib" != "$VENV_ROOT/lib" ]] \
  && roots+=("$BASE_PREFIX/lib")

started=$SECONDS
files=0
for root in "${roots[@]}"; do
  count="$(find "$root" -type f 2>/dev/null | wc -l)"
  files=$((files + count))
  find "$root" -type f -print0 2>/dev/null \
    | xargs -0 -r -P"$WORKERS" -n512 stat --format=%s \
    >/dev/null 2>&1
done

"$PYTHON_BIN" - "${MODULES[@]}" <<'PY'
import importlib
import sys

for name in sys.argv[1:]:
    module = importlib.import_module(name)
    print(f"python warm import={name} version={getattr(module, '__version__', 'n/a')}")
PY
echo "python warm ready host=$(hostname -s) files=$files elapsed_s=$((SECONDS-started)) python=$PYTHON_BIN"
