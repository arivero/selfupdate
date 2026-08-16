#!/bin/bash
#SBATCH --job-name=v5w2_smoke
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=runs/sbatch_v5w2_smoke_%j.out
#SBATCH --error=runs/sbatch_v5w2_smoke_%j.err
#
# v5w2 pre-campaign smoke gate (2026-08-16): exercise every NEW trainer
# feature (delta_vmse, mix, --train-norms, --dora) on micro runs before the
# two screen chains start. On full success writes runs/v5w2_smoke_OK, which
# the chains' head jobs depend on (afterok). Each micro run also executes
# the layerwise isolation certification at startup — that is the law check.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }
rm -f runs/v5w2_smoke_OK

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"

MODEL="google/gemma-4-31B-it"
echo "==== v5w2 smoke start commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/venv_setup.sh || fail "venv setup"
if ! scripts/venv_check.sh; then
  echo "venv check failed — force rebuild" >&2
  rm -rf "/tmp/$USER/selfupdate-venv"
  scripts/venv_setup.sh || fail "venv rebuild"
  scripts/venv_check.sh || fail "venv check after rebuild"
fi
PY="/tmp/$USER/selfupdate-venv/bin/python"
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

"$PY" scripts/trainv5.py --dry-data || fail "data reconstruction gate"

MICRO="--positions aligned --limit 48 --epochs 2 --eval-every 2 \
  --recall-samples 4 --arc-limit 20"
run_smoke() {
  local name="$1"; shift
  echo "---- smoke: $name ($*) $(date -Is)"
  # shellcheck disable=SC2086
  "$PY" scripts/trainv5.py --run-name "v5w2_smoke_$name" $MICRO "$@" \
    || fail "smoke $name"
}
run_smoke dvmse --local-loss delta_vmse --layer-gate topk_abs:1
run_smoke mix   --local-loss mix
run_smoke norms --local-loss vocab_mse --train-norms
run_smoke dora  --local-loss delta_cosine --layer-gate topk_abs:1 --dora

git rev-parse HEAD > runs/v5w2_smoke_OK
echo "==== v5w2 smoke ALL PASS $(date -Is) ===="
