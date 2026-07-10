#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_PYTHON_HOST="${SELFUPDATE_DEV_PYTHON_HOST:-/tmp/$USER/selfupdate-dev-python}"
mkdir -p "$DEV_PYTHON_HOST"

exec "$ROOT/scripts/container_exec.sh" python -m pip "$@" \
  --target /dev-python \
  --no-cache-dir \
  --no-warn-script-location
