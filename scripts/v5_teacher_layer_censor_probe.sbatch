#!/bin/bash
#SBATCH --job-name=v5_layer_censor
#SBATCH --partition=agustina_fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=runs/sbatch_v5_layer_censor_%j.out
#SBATCH --error=runs/sbatch_v5_layer_censor_%j.err
#
# Layer-selective teacher censorship (owner design 2026-08-20): causal
# ablation of the attention map — block passage attention at chosen
# layers, measure teacher-forced CE/argmax on the teacher's own answers.
# Chained afterok the attention probe; derives the retrieval set S from
# its attn_by_layer.json. CPU fat node, no GPU queue.
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

echo "==== layer censor probe start commit=$(git rev-parse HEAD) $(date -Is) host=$(hostname) ===="
[[ -f runs/v5_teacher_attn/attn_by_layer.json ]] \
  || fail "attention probe output missing"
scripts/venv_setup.sh || fail "venv setup"
PY="/tmp/$USER/selfupdate-venv/bin/python"

"$PY" scripts/v5_teacher_layer_censor_probe.py || fail "layer censor probe"
echo "==== layer censor probe DONE $(date -Is) ===="
