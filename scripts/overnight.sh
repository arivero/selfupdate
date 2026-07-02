#!/usr/bin/env bash
# Detached overnight pipeline: survives SSH/session death (launch with
#   nohup setsid bash scripts/overnight.sh >> runs/pipeline.log 2>&1 &
# ). Each step is guarded so one failure does not kill the rest; skip logic
# makes the script idempotent — rerun it and it continues where it left off.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*"; }

step() {  # step <name> <done-file> <cmd...>
    local name=$1 done_file=$2; shift 2
    if [ -e "$done_file" ]; then
        log "SKIP $name ($done_file exists)"
        return 0
    fi
    log "START $name"
    if "$@"; then
        log "DONE $name"
    else
        log "FAIL $name (exit $?)"
    fi
}

log "=== overnight pipeline start (pid $$) ==="

step eval-kd-ce runs/kd_ce_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/kd_ce_0p6b_rag/checkpoint

step eval-base runs/base-eval-full/recite.json \
    bash -c "$PY scripts/evaluate.py --base && mkdir -p runs/base-eval-full && cp runs/base-eval/recite.json runs/base-eval-full/recite.json"

step train-lw-summed runs/lw_summed_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_summed_0p6b_rag.yaml

step eval-lw-summed runs/lw_summed_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_summed_0p6b_rag/checkpoint

step train-lw-seq runs/lw_seq_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_seq_0p6b_rag.yaml

step eval-lw-seq runs/lw_seq_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_seq_0p6b_rag/checkpoint

step train-kd-lora runs/kd_lora_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/kd_lora_0p6b_rag.yaml

step eval-kd-lora runs/kd_lora_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/kd_lora_0p6b_rag/checkpoint

step analyze runs/delta_profiles.png \
    $PY scripts/analyze.py --deltas kd_full_0p6b_rag kd_ce_0p6b_rag lw_summed_0p6b_rag lw_seq_0p6b_rag

log "=== overnight pipeline finished ==="
