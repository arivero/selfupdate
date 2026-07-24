#!/bin/bash
#SBATCH --job-name=v4-report
#SBATCH --partition=agustina_thin
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/sbatch_v4_report_%j.out
#SBATCH --error=runs/sbatch_v4_report_%j.err
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$#" -eq 1 ]] || { echo "usage: sbatch scripts/v4_report.sbatch RUN_DIR" >&2; exit 2; }
[[ -d "$1" ]] || { echo "missing run directory: $1" >&2; exit 2; }
echo "==== report start run=$1 commit=$(git rev-parse HEAD) $(date -Is) ===="
/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3 scripts/v4_run_report.py "$1"
echo "==== report complete run=$1 $(date -Is) ===="
