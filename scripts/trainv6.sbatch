#!/bin/bash
#SBATCH --job-name=trainv6
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=runs/sbatch_trainv6_%j.out
#SBATCH --error=runs/sbatch_trainv6_%j.err
#
# Experimental pipeline v6. The Python entry point is a deliberate monolith;
# this wrapper only provides the node runtime and passes the exact same
# arguments through the CPU data gate and the training invocation.
#
# sbatch --account=supercomplex scripts/trainv6.sbatch \
#   --preset gemma4_31b --method causal_residual --run-name v6_g31b_cr_s17

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }

ARGS=("$@")
PRESET=""
for ((INDEX=0; INDEX<${#ARGS[@]}; INDEX++)); do
  case "${ARGS[$INDEX]}" in
    --preset)
      (( INDEX + 1 < ${#ARGS[@]} )) || fail "--preset needs a value"
      PRESET="${ARGS[$((INDEX + 1))]}"
      ;;
    --preset=*)
      PRESET="${ARGS[$INDEX]#--preset=}"
      ;;
  esac
done

case "$PRESET" in
  gemma4_31b) MODEL="google/gemma-4-31B-it" ;;
  qwen36_27b) MODEL="Qwen/Qwen3.6-27B" ;;
  *) fail "use --preset gemma4_31b or --preset qwen36_27b" ;;
esac

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a
  source "$ROOT/.campaign_env.sh"
  set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-v6-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-v6-torchinductor"

echo "==== trainv6 start preset=$PRESET model=$MODEL commit=$(git rev-parse HEAD) $(date -Is) ===="

# Held preflight jobs may wait in Slurm for days.  Pin the exact monolith that
# was reviewed at submission rather than silently running a later worktree
# edit when the job is released.
if [[ -n "${SELFUPDATE_V6_CODE_SHA256:-}" ]]; then
  OBSERVED_SHA256="$(sha256sum scripts/trainv6.py)"
  OBSERVED_SHA256="${OBSERVED_SHA256%% *}"
  [[ "$OBSERVED_SHA256" == "$SELFUPDATE_V6_CODE_SHA256" ]] \
    || fail "trainv6.py changed after submission: expected $SELFUPDATE_V6_CODE_SHA256 got $OBSERVED_SHA256"
fi

scripts/venv_setup.sh || fail "venv setup"
scripts/venv_check.sh || fail "venv check"
PY="/tmp/$USER/selfupdate-venv/bin/python"
export PYTHONPYCACHEPREFIX="/tmp/$USER/selfupdate-v6-pycache-$SLURM_JOB_ID"

scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE=1
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

"$PY" -m py_compile scripts/trainv6.py \
  || fail "monolith compile gate"
"$PY" scripts/trainv6.py --self-test-metrics \
  || fail "metric self-test gate"
"$PY" scripts/trainv6.py "${ARGS[@]}" --dry-data \
  || fail "model-matched data/metric gate"
"$PY" scripts/trainv6.py "${ARGS[@]}" \
  || fail "trainv6"
echo "==== trainv6 complete $(date -Is) ===="
