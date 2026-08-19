#!/bin/bash
#SBATCH --job-name=v5_teacher_ceiling
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=runs/sbatch_v5_teacher_ceiling_%j.out
#SBATCH --error=runs/sbatch_v5_teacher_ceiling_%j.err
#
# Teacher-ceiling probe (owner question 2026-08-19): measure the recall
# metric's ceiling — HF greedy from the UNCENSORED prompt under the exact
# recall_eval decode path — plus the censored control, on one node. Eval
# only: no adapters, no training, writes runs/v5_teacher_ceiling/ only.
# Submitted with --nice so it NEVER outranks the gate-screen chains; it
# runs whenever a hole opens (early screen failure, pruned long, or after
# the chains drain).
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

MODEL="google/gemma-4-31B-it"
echo "==== teacher ceiling probe start commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/venv_setup.sh || fail "venv setup"
scripts/venv_check.sh || fail "venv check"
PY="/tmp/$USER/selfupdate-venv/bin/python"
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

"$PY" scripts/v5_teacher_ceiling_probe.py || fail "ceiling probe"
echo "==== teacher ceiling probe DONE $(date -Is) ===="
