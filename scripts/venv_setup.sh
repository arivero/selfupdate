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

if [[ "${1:-}" == "--force" ]]; then
  rm -rf "$VENV"
fi
if [[ -x "$VENV/bin/python" ]]; then
  echo "venv already present: $VENV  (use --force to rebuild)"
  exit 0
fi

echo "building $VENV (python $PYTHON_VERSION) ..."
"$UV" venv "$VENV" --python "$PYTHON_VERSION"

# Torch comes from the cu128 index and is pinned: it is the one dependency
# whose CUDA build must match the node's driver. cu128 needs a >=12.8-capable
# driver (agpuh01: 565.57.01 -> OK). Older-driver nodes need a matching torch,
# not this pin -- check `nvidia-smi` before assuming.
"$UV" pip install --python "$VENV/bin/python" \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Everything else. kernels MUST stay ==0.12.0 with transformers 5.12.1:
# kernels 0.16 breaks ALL model loading with
# "ValueError: Either a revision or a version ...".
"$UV" pip install --python "$VENV/bin/python" \
  transformers==5.12.1 \
  accelerate==1.14.0 \
  peft==0.19.1 \
  kernels==0.12.0 \
  safetensors pyyaml pandas tabulate matplotlib tqdm

echo
echo "done: $VENV"
echo "verify with: scripts/venv_check.sh"
echo "use with:    $VENV/bin/python scripts/train.py --config ... --experiment ..."
