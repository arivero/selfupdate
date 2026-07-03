#!/usr/bin/env bash
#SBATCH --job-name=selfupdate-h100
#SBATCH --partition=rasmia_hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm_h100_%j.log
# H100 lane: same shared-Lustre repo, own queue file + per-node scheduler
# state (AGENTS.md "Multi-node"). Submit from the repo root:
#   sbatch scripts/slurm_h100.sbatch
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ngpu=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
GPUS="$(seq -s' ' 0 $((ngpu - 1)))" \
MAX_PER_GPU=1 \
QUEUE=scripts/queue_h100.tsv \
SCHED="runs/.sched-$(hostname -s)" \
JOBLOG="runs/pipeline_sched_$(hostname -s).log" \
exec bash scripts/gpu_scheduler.sh
