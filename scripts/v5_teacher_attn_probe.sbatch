#!/bin/bash
#SBATCH --job-name=v5_teacher_attn
#SBATCH --partition=agustina_fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=runs/sbatch_v5_teacher_attn_%j.out
#SBATCH --error=runs/sbatch_v5_teacher_attn_%j.err
#
# Teacher attention-by-layer probe (owner question 2026-08-20): CPU-only
# on an idle fat node (496G usable, micro-tested) — bypasses the fully
# occupied GPU queue. One teacher-forced forward per item with eager
# attention; keeps only answer-row query stats per layer.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SELFUPDATE_CPU_THREADS=56
export OMP_NUM_THREADS=56

echo "==== teacher attn probe start commit=$(git rev-parse HEAD) $(date -Is) host=$(hostname) ===="
scripts/venv_setup.sh || fail "venv setup"
PY="/tmp/$USER/selfupdate-venv/bin/python"
"$PY" -c "import torch, transformers; print(torch.__version__, transformers.__version__)" \
  || fail "venv import check (CPU)"

"$PY" scripts/v5_teacher_attn_probe.py || fail "attn probe"
echo "==== teacher attn probe DONE $(date -Is) ===="
