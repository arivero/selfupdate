#!/usr/bin/env bash
# Build the node-local Python runtime for this checkout, fast.
#
# WHY node-local: a venv is tens of thousands of small files. On Lustre that
# metadata cost dominates -- a cold `import torch` from a Lustre venv has been
# measured not finishing inside two minutes on agpuh01. In /tmp (node-local
# NVMe on the tested nodes) the same venv is created in seconds and imports in
# tens of seconds cold, near-instantly warm.
#
# WHY create and not copy: a venv bakes absolute paths into pyvenv.cfg and
# every console-script shebang. It cannot be relocated by copying. Creating a
# fresh one per node is both correct and faster than copying one.
#
# /tmp is node-local: run this ONCE PER NODE. It is disposable -- delete and
# rerun rather than repairing. Nothing scientific lives here.
#
# Usage:
#   scripts/venv_setup.sh                     # build (idempotent-ish; see --force)
#   scripts/venv_setup.sh --force             # delete and rebuild
#   SELFUPDATE_VENV=/tmp/$USER/other scripts/venv_setup.sh
#
# Then run everything through the interpreter it prints, from the repo root:
#   /tmp/$USER/selfupdate-venv/bin/python scripts/train.py --config ... --experiment ...
#
# There is deliberately NO `pip install -e .` here. The entry points in
# scripts/ pin their own tree with sys.path.insert(0, <repo>/src), so a bare
# `import selfupdate` fails loudly rather than silently resolving to a sibling
# checkout. Keep it that way: one venv can serve several checkouts safely.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SELFUPDATE_VENV:-/tmp/$USER/selfupdate-venv}"
PYTHON_VERSION="${SELFUPDATE_PYTHON_VERSION:-3.12}"

# A cancelled build can leave the target directory behind before uv has
# written pyvenv.cfg/bin/python.  uv deliberately refuses to overlay that
# directory, so recognize and discard this one disposable state here rather
# than making every batch wrapper special-case it.  Automatic deletion is
# restricted to this user's node-local /tmp subtree.
VENV="$(realpath -m -- "$VENV")"
VENV_PARENT="$(realpath -m -- "/tmp/$USER")"
mkdir -p -- "$VENV_PARENT"
remove_disposable_venv() {
  if [[ "$VENV" == "$VENV_PARENT" || "$VENV" != "$VENV_PARENT/"* ]]; then
    echo "error: refusing to delete venv outside $VENV_PARENT: $VENV" >&2
    exit 1
  fi
  rm -rf -- "$VENV"
}

# uv resolves and installs far faster than pip and is already on this cluster.
UV="${UV:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  echo "error: uv not found on PATH." >&2
  echo "  expected: /fs/agustina/arivero/supercomplex/.local/bin/uv" >&2
  exit 1
fi

# Python HTTPS on this cluster rejects the proxy chain without an explicit CA
# bundle; uv honours SSL_CERT_FILE too.
export SSL_CERT_FILE="${SSL_CERT_FILE:-/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem}"
# Keep the uv download cache node-local as well; it is disposable.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER/uv-cache}"

# Several full-node jobs can transition within seconds on the same host.  A
# node-local lock prevents one launch from checking a venv while another is
# deleting or populating it.
exec 9>"$VENV_PARENT/.selfupdate-venv-setup.lock"
flock -x 9

venv_has_core() {
  [[ -x "$VENV/bin/python" && -f "$VENV/pyvenv.cfg" ]] || return 1
  "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

expected = {
    "torch": "2.11.0+cu128",
    "transformers": "5.12.1",
    "kernels": "0.12.0",
    "safetensors": "0.8.0",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
}
raise SystemExit(any(version(name) != want for name, want in expected.items()))
PY
}

venv_has_optional() {
  "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

expected = {"datasets": "5.0.0", "nvidia-ml-py": "13.610.43"}
raise SystemExit(any(version(name) != want for name, want in expected.items()))
PY
}

if [[ "${1:-}" == "--force" ]]; then
  remove_disposable_venv
fi
CORE_READY=0
venv_has_core && CORE_READY=1
if [[ "$CORE_READY" == 1 ]]; then
  if [[ "${SELFUPDATE_VENV_CORE_ONLY:-0}" == 1 ]] || venv_has_optional; then
    echo "venv already present: $VENV  (use --force to rebuild)"
    exit 0
  fi
  echo "core venv present; installing missing optional runtime" >&2
elif [[ -e "$VENV" || -L "$VENV" ]]; then
  echo "removing incomplete node-local venv: $VENV" >&2
  remove_disposable_venv
fi

# requirements-cu128.txt is the ONE source of truth for the pins -- do not
# duplicate versions here. It carries the cu128 extra-index-url, so torch's
# CUDA build comes from it too. torch is the one dependency that must match
# the node's driver: cu128 needs a >=12.8-capable driver (agpuh01: 565.57.01
# -> OK). An older-driver node (driver-560 L40S) needs a different torch, so
# it needs a different requirements file -- not an edit to this script.
REQS="${SELFUPDATE_REQUIREMENTS:-$ROOT/requirements-cu128.txt}"
if [[ ! -f "$REQS" ]]; then
  echo "error: requirements file not found: $REQS" >&2
  exit 1
fi

# --index-strategy unsafe-best-match: requirements-cu128.txt carries
# `--extra-index-url .../whl/cu128`, and uv by default only considers versions
# from the FIRST index that lists a package (a dependency-confusion guard).
# The pytorch index lists tqdm etc. but not at our pinned versions, so the
# default strategy fails with "tqdm was found on download.pytorch.org, but not
# at the requested version". pip searches both indexes implicitly; this flag
# restores that behaviour. Both indexes here are trusted, and every version is
# pinned in the requirements file, which is what actually bounds the risk.
UV_INDEX_ARGS=(--index-strategy unsafe-best-match)

if [[ "$CORE_READY" == 0 ]]; then
  echo "building $VENV (python $PYTHON_VERSION) from $(basename "$REQS") ..."
  "$UV" venv "$VENV" --python "$PYTHON_VERSION"
  "$UV" pip install --python "$VENV/bin/python" "${UV_INDEX_ARGS[@]}" -r "$REQS"
  venv_has_core || { echo "error: core venv install did not validate" >&2; exit 1; }
fi

# requirements-optional.txt is named "optional" but is NOT optional for the
# supported trainer: src/selfupdate/eval/standard.py does a module-level
# `from datasets import load_dataset`, and any config with
# eval.standard_damage_every_epochs > 0 (i.e. the normal ones) reaches it
# during epoch-zero telemetry. Skipping it dies with
# "ModuleNotFoundError: No module named 'datasets'" AFTER model load, teacher
# cache load and epoch-zero recall -- minutes of GPU time in. Installed by
# default. Set SELFUPDATE_VENV_CORE_ONLY=1 only for a self-contained entry
# point (currently trainv6.py) that owns its vendored standard evaluator and
# does not import selfupdate.eval.standard.
# The library is needed even though the standard subsets are vendored under
# data/eval/ at pinned revisions; vendoring removes the DOWNLOAD, not the
# import.
if [[ "${SELFUPDATE_VENV_CORE_ONLY:-0}" != "1" ]]; then
  "$UV" pip install --python "$VENV/bin/python" "${UV_INDEX_ARGS[@]}" -r "$ROOT/requirements-optional.txt"
  venv_has_optional || { echo "error: optional venv install did not validate" >&2; exit 1; }
fi

echo
echo "done: $VENV"
echo "verify with: scripts/venv_check.sh"
echo "use with:    $VENV/bin/python scripts/train.py --config ... --experiment ..."
