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

PYTHON_REAL="$(readlink -f "$PYTHON_BIN")"
BASE_STDLIB="$($PYTHON_LAUNCHER -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"
VENV_ROOT="$(cd "$(dirname "$PYTHON_LAUNCHER")/.." 2>/dev/null && pwd || true)"
# The resolved executable can live in a shared uv base while its site-packages
# live beside the original venv launcher. Recover that original root as well.
if [[ -z "$VENV_ROOT" || ! -d "$VENV_ROOT/lib" ]]; then
  VENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

roots=()
[[ -d "$VENV_ROOT/lib" ]] && roots+=("$VENV_ROOT/lib")
[[ -d "$BASE_STDLIB" && "$BASE_STDLIB" != "$VENV_ROOT/lib" ]] \
  && roots+=("$BASE_STDLIB")

started=$SECONDS
trees=0
for root in "${roots[@]}"; do
  trees=$((trees + 1))
  # One parallel metadata walk is the warm-up.  The historical file-count
  # telemetry performed a complete serial `find | wc` first, doubling the
  # Lustre traversal and making this helper itself a multi-minute cold-start
  # bottleneck on the large Anaconda base tree.
  # The Anaconda base's stdlib directory also contains a giant unrelated
  # site-packages tree. Requested imports below fault the dependencies that
  # are actually used; pre-statting every package/test in the distribution
  # defeats the purpose of a bounded runtime warm-up.
  find "$root" -path "$BASE_STDLIB/site-packages" -prune -o \
    -type f -print0 2>/dev/null \
    | xargs -0 -r -P"$WORKERS" -n512 stat --format=%s \
    >/dev/null 2>&1
done

"$PYTHON_LAUNCHER" - "${MODULES[@]}" <<'PY'
import importlib
import sys

for name in sys.argv[1:]:
    module = importlib.import_module(name)
    print(f"python warm import={name} version={getattr(module, '__version__', 'n/a')}")
PY
echo "python warm ready host=$(hostname -s) trees=$trees elapsed_s=$((SECONDS-started)) python=$PYTHON_LAUNCHER resolved=$PYTHON_REAL"
