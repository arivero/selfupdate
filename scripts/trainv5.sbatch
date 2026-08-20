#!/bin/bash
#SBATCH --job-name=trainv5_selfdistill
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=runs/sbatch_trainv5_%j.out
#SBATCH --error=runs/sbatch_trainv5_%j.err
#
# Pipeline-v5 experiment 1: self-distillation with remove-view censorship
# (scripts/trainv5.py — a deliberate monolith; see its header for the law).
# Reuses the existing vLLM answers artifact read-only; no cache builds, no
# v4 machinery. Submit from the repo root:
#
#   sbatch --account=supercomplex scripts/trainv5.sbatch [extra trainv5 args]
#
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"

# cross-model (2026-08-20): stage whatever model the args train. TRAIN_MODEL
# env overrides; default stays the campaign model.
MODEL="${TRAIN_MODEL:-google/gemma-4-31B-it}"
echo "==== trainv5 start model=$MODEL commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/venv_setup.sh || fail "venv setup"
if ! scripts/venv_check.sh; then
  # a stale/corrupted node-local venv is disposable, never repaired
  # (three arms died in 5s each to a broken .pth, 2026-08-07..11)
  echo "venv check failed — force rebuild" >&2
  rm -rf "/tmp/$USER/selfupdate-venv"
  scripts/venv_setup.sh || fail "venv rebuild"
  scripts/venv_check.sh || fail "venv check after rebuild"
fi
PY="/tmp/$USER/selfupdate-venv/bin/python"
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
# consume the RAM stage (same contract as launch_v4_stages.sh): snapshots
# resolve from /dev/shm, and offline mode makes a cache miss fail loudly.
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

# data-path gate before touching the GPUs: the round-trip guard must pass
"$PY" scripts/trainv5.py --dry-data || fail "data reconstruction gate"

"$PY" scripts/trainv5.py "$@" || fail "trainv5"
echo "==== trainv5 complete $(date -Is) ===="
