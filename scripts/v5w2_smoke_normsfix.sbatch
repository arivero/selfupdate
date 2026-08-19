#!/bin/bash
#SBATCH --job-name=v5w2_smoke_normsfix
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=runs/sbatch_v5w2_smoke_normsfix_%j.out
#SBATCH --error=runs/sbatch_v5w2_smoke_normsfix_%j.err
#
# Smoke for the frozen_base() repair (ultrareview bug_002, 2026-08-19):
# --train-norms previously let the trained norms leak into every
# adapters-off forward; frozen_base() now snapshot/restores them. This
# micro run exercises the repaired path end to end (teacher capture,
# self-anchor, evaluate) before the clean 6h norms re-run, which is
# afterok on the marker this writes.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }
rm -f runs/v5w2_smoke_normsfix_OK

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"

MODEL="google/gemma-4-31B-it"
echo "==== normsfix smoke start commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/venv_setup.sh || fail "venv setup"
scripts/venv_check.sh || fail "venv check"
PY="/tmp/$USER/selfupdate-venv/bin/python"
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

"$PY" scripts/trainv5.py --dry-data || fail "data reconstruction gate"
"$PY" scripts/trainv5.py --run-name v5w2_smoke_normsfix \
  --positions aligned --limit 48 --epochs 2 --eval-every 2 \
  --recall-samples 4 --arc-limit 20 \
  --local-loss delta_cosine --layer-gate topk_abs:1 --train-norms \
  || fail "normsfix micro run"
grep -q '"kind": "done"' runs/v5w2_smoke_normsfix/metrics.jsonl \
  || fail "normsfix smoke did not complete"

git rev-parse HEAD > runs/v5w2_smoke_normsfix_OK
echo "==== normsfix smoke PASS $(date -Is) ===="
