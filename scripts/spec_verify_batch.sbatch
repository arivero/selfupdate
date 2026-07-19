#!/usr/bin/env bash
#SBATCH --job-name=spec_verify_matrix
#SBATCH --account=supercomplex
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=runs/spec_verify/batch_%j.log
# Autonomous continuation of the vLLM-reproduction matrix (owner goal
# 2026-07-19): trainer-native PPP1 rows for the remaining single-node models.
# 122b/PPP8/397B need interactive supervision — see runs/spec_verify/NEXT_LEASE.md.
set -x
cd "$SLURM_SUBMIT_DIR"
scripts/venv_setup.sh
scripts/venv_check.sh || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True SELFUPDATE_CPU_THREADS=8
for key in 27b 35b 26b 31b; do
  echo "==== MATRIX ROW $key ===="
  bash scripts/spec_verify_matrix.sh "$key" 0 || echo "ROW $key FAILED rc=$?"
done
echo "==== BATCH DONE $(date) ===="
