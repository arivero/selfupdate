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
export SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
[ "$SELFUPDATE_CPU_THREADS" -gt 22 ] 2>/dev/null && SELFUPDATE_CPU_THREADS=22
export OMP_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export MKL_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

# Pass the allocated device ids through VERBATIM. Never renumber from 0:
# without cgroup ConstrainDevices, CUDA_VISIBLE_DEVICES=1 on an "0,2,3"
# allocation lands on a stranger's physical GPU 1 (happened 2026-07-03,
# on rlaplaza's card). Verbatim ids also make the scheduler's
# nvidia-smi -i probes read the right cards.
GPUS="${GPUS:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' ' ')}"  # explicit GPUS env wins

# one-time cold-node warmup: transformers' import stats every file in
# site-packages (importlib.metadata); on a cold Lustre client that is
# minutes of serial metadata round trips. Warm dentries in parallel and
# pre-import once so the real jobs start hot.
find .venv/lib -type f -print0 2>/dev/null | xargs -0 -P16 -n512 stat --format=%s >/dev/null 2>&1
.venv/bin/python -c "import torch, transformers" >/dev/null 2>&1

GPUS="$GPUS" \
MAX_PER_GPU=1 \
QUEUE=scripts/queue_h100.tsv \
SCHED="runs/.sched-$(hostname -s)" \
JOBLOG="runs/pipeline_sched_$(hostname -s).log" \
exec bash scripts/gpu_scheduler.sh
