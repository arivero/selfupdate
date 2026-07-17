#!/usr/bin/env bash
# Verify the node-local venv actually works on THIS node, before a campaign
# spends GPU-minutes discovering it does not.
#
# Checks, in order of what has actually bitten us:
#   1. the interpreter exists
#   2. torch imports and its CUDA build matches the node's driver
#   3. a real bf16 CUDA matmul executes (import success != working CUDA)
#   4. the library pins are exactly what the repo requires
#   5. `selfupdate` resolves to THIS checkout, not a sibling
#
# Usage: scripts/venv_check.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SELFUPDATE_VENV:-/tmp/$USER/selfupdate-venv}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "FAIL: no interpreter at $PY" >&2
  echo "  build it with: scripts/venv_setup.sh" >&2
  exit 1
fi

echo "venv:   $VENV"
echo "node:   $(hostname -s)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader || true
echo

export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SELFUPDATE_ROOT="$ROOT" "$PY" - <<'PY'
import os, sys

root = os.environ["SELFUPDATE_ROOT"]
sys.path.insert(0, os.path.join(root, "src"))
failures = []

import torch
print(f"python       {sys.version.split()[0]}")
print(f"torch        {torch.__version__} (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    failures.append(
        "torch.cuda.is_available() is False -- the torch CUDA build likely "
        "does not match this node's driver (check nvidia-smi; cu128 needs a "
        ">=12.8-capable driver)")
else:
    print(f"devices      {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    # Import success does not prove CUDA works; execute a real kernel.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    val = float((x @ x).float().mean())
    print(f"bf16 matmul  ok ({val:.6f})")

# kernels==0.12.0 is load-bearing: 0.16 breaks ALL model loading with
# "ValueError: Either a revision or a version ...".
EXPECTED = {
    "transformers": "5.12.1",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "kernels": "0.12.0",
}
for name, want in EXPECTED.items():
    try:
        mod = __import__(name)
        got = getattr(mod, "__version__", "?")
    except Exception as exc:
        failures.append(f"{name}: import failed ({exc})")
        continue
    status = "ok" if got == want else "MISMATCH"
    print(f"{name:<12} {got} ({status}, want {want})")
    if got != want:
        failures.append(f"{name} is {got}, expected {want}")

for name in ("safetensors", "yaml", "pandas", "tabulate", "matplotlib", "tqdm"):
    try:
        __import__(name)
    except Exception as exc:
        failures.append(f"{name}: import failed ({exc})")

# The eval path is where a half-installed venv actually bites, and it bites
# LATE: selfupdate/eval/standard.py imports `datasets` at module level, and a
# config with eval.standard_damage_every_epochs > 0 only reaches it during
# epoch-zero telemetry -- after model load, teacher-cache load and epoch-zero
# recall. Importing it here turns minutes of wasted GPU into an instant fail.
try:
    import datasets  # noqa: F401
    print(f"datasets     {datasets.__version__}")
except Exception as exc:
    failures.append(
        f"datasets: import failed ({exc}) -- training with standard-damage "
        "eval will die at epoch-zero telemetry. Install "
        "requirements-optional.txt (scripts/venv_setup.sh does by default)")
try:
    from selfupdate.eval.standard import STANDARD_TASKS  # noqa: F401
    print(f"eval.standard ok ({', '.join(list(STANDARD_TASKS)[:3])})")
except Exception as exc:
    failures.append(f"selfupdate.eval.standard: import failed ({exc})")

# The venv carries no editable install by design; entry points pin their own
# tree. Confirm we resolved to THIS checkout and not a sibling.
import selfupdate
resolved = os.path.realpath(selfupdate.__file__)
expected_prefix = os.path.realpath(os.path.join(root, "src"))
print(f"selfupdate   {resolved}")
if not resolved.startswith(expected_prefix):
    failures.append(
        f"selfupdate resolved to {resolved}, outside this checkout "
        f"({expected_prefix}) -- a stray editable install is routing imports "
        "across checkouts")

print()
if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: venv is usable on this node")
PY
