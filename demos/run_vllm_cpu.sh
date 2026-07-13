#!/usr/bin/env bash
# Run the vLLM CPU baseline inside the official CPU container.
# Usage: demos/run_vllm_cpu.sh <prompts.jsonl> <out_dir> [threads]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SIF="${VLLM_CPU_SIF:-/tmp/$USER/vllm-cpu-0.25.0.sif}"
PROMPTS="$1"; OUT="$2"; THREADS="${3:-32}"

mkdir -p "/tmp/$USER/singularity_tmp"
export SINGULARITY_CACHEDIR="/tmp/$USER/singularity_cache"
export SINGULARITY_TMPDIR="/tmp/$USER/singularity_tmp"

# CPU-backend knobs: KV space in GiB; bind OpenMP to one socket's cores so the
# comparison uses the same core budget as the torch run's --threads.
singularity exec \
  --bind "$REPO:/work" \
  --bind "/fs/agustina/arivero/supercomplex/.cache/huggingface:/hf" \
  --env HF_HOME=/hf,HF_HUB_OFFLINE=1,VLLM_CPU_KVCACHE_SPACE=16,VLLM_CPU_OMP_THREADS_BIND="0-$((THREADS-1))",VLLM_TARGET_DEVICE=cpu \
  "$SIF" \
  python3 /work/demos/generate_vllm_cpu.py --prompts "/work/$PROMPTS" --out "/work/$OUT"
