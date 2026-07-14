#!/usr/bin/env bash
# Create the thin node-local dependency layer used by l40s_exec.sh.
# This command produces an install log and must be run by a small delegated
# agent under the repository's logged-launch rule.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${SELFUPDATE_L40S_PYTHON:-$ROOT/../jacobian-lens/.venv/bin/python}"
DEPS="${SELFUPDATE_L40S_DEPS:-/tmp/$USER/selfupdate-l40-python}"
mkdir -p "$DEPS"
uv pip install --python "$BASE_PYTHON" --target "$DEPS" --no-deps \
  'transformers==5.12.1' 'peft==0.19.1' 'kernels==0.12.0'
if find "$DEPS" -maxdepth 1 -iname 'torch*' -print -quit | grep -q .; then
  echo "unexpected torch package in $DEPS" >&2
  exit 2
fi
PYTHONPATH="$DEPS:$ROOT/src" "$BASE_PYTHON" - <<'PY'
import torch, transformers, peft, kernels
assert torch.version.cuda == "12.6", torch.version.cuda
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "peft", peft.__version__,
      "kernels", kernels.__version__)
PY
