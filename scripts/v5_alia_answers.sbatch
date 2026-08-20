#!/bin/bash
#SBATCH --job-name=v5_alia_answers
#SBATCH --partition=rasmia_hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=runs/sbatch_v5_alia_answers_%j.out
#SBATCH --error=runs/sbatch_v5_alia_answers_%j.err
#
# ALIA-40b teacher-answers artifact (2026-08-20): the fully-global
# (48/48 full attention) Spanish-native substrate candidate for v5w3
# lacks a responses artifact — this generates it, mirroring the Phase-1
# gemma4_31b_it flags. Infrastructure only: the RAG target-generation
# gate (docs/rag_generation_gate.md) still applies BEFORE any training
# consumes it. Submitted --nice=500: fills spare H100 turn behind the
# science chain, never outranks it.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"
fail() { echo "FATAL: $*" >&2; exit 1; }

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  set -a; source "$ROOT/.campaign_env.sh"; set +a
fi
export SSL_CERT_FILE="${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh}"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export VLLM_CACHE_ROOT="/tmp/$USER/selfupdate-vllm-cache"
export VLLM_CONFIG_ROOT="/tmp/$USER/selfupdate-vllm-config"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"

MODEL="BSC-LT/ALIA-40b-fc-2606"
VLLM_PY="${SELFUPDATE_VLLM_PYTHON:?}"
[[ -x "$VLLM_PY" ]] || fail "vLLM interpreter not found at $VLLM_PY"

echo "==== ALIA answers start commit=$(git rev-parse HEAD) $(date -Is) ===="
scripts/stage_hf_cache.sh --shm "$MODEL" || fail "model staging"
export HF_HOME="/dev/shm/$USER/selfupdate-hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
[[ -f "$HF_HOME/.selfupdate-hf-stage-ready" ]] || fail "HF SHM stage not ready"

GEN_DIR="runs/vllm_h100/alia40b_fc"
[[ -e "$GEN_DIR" ]] && fail "output dir exists: $GEN_DIR (remove stale first)"
mkdir -p "$GEN_DIR"
# --out is an OUTPUT DIRECTORY (see spec_g26b_a4b_campaign.sbatch note,
# caught live on job 422578)
"$VLLM_PY" scripts/benchmark_vllm_generation.py \
  --model "$MODEL" \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 \
  --tensor-parallel-size 4 \
  --max-model-len 6144 \
  --prompt-format native \
  --generation-extra-tokens 96 \
  --out "$GEN_DIR" || fail "vLLM generation"
[[ -s "$GEN_DIR/responses_bs256.jsonl" ]] || fail "no responses written"
echo "==== ALIA answers DONE $(date -Is) ===="
