#!/usr/bin/env bash
# Wave C, EVAL/ANALYSIS lane. Overlaps the training lane: causal analyses
# first (fit alongside any training), then evals each checkpoint as its
# .train_done marker appears. VRAM-guarded.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_vram() { while [ "$(free_mb)" -lt "$1" ]; do sleep 30; done; }

step() {
    local name=$1 done_file=$2 need=$3; shift 3
    if [ -e "$done_file" ]; then log "SKIP $name"; return 0; fi
    wait_vram "$need"
    log "START $name"
    if "$@"; then log "DONE $name"; else log "FAIL $name (exit $?)"; fi
}

wait_marker() { while [ ! -e "$1" ]; do sleep 60; done; }

log "=== overnight_c EVAL lane start (pid $$) ==="

# causal localization on the reciting checkpoint (runs alongside training)
step layer-swap-kd-ce runs/kd_ce_0p6b_rag/eval/layer_swap.csv 3500 \
    $PY scripts/layer_swap.py --run kd_ce_0p6b_rag --limit 8

step logit-lens-kd-ce runs/kd_ce_0p6b_rag/eval/logit_lens.csv 2500 \
    $PY scripts/logit_lens.py --run kd_ce_0p6b_rag --limit 24

for name in lw_summed_ce_0p6b_rag lw_tc_ce_0p6b_rag kd_lora_ce_0p6b_rag; do
    wait_marker "runs/$name/.train_done"
    step "eval-$name" "runs/$name/eval/recite.json" 2500 \
        $PY scripts/evaluate.py --checkpoint "runs/$name/checkpoint"
done

step analyze-c runs/results_c.marker 0 \
    bash -c "$PY scripts/analyze.py --deltas kd_full_0p6b_rag kd_ce_0p6b_rag lw_summed_0p6b_rag lw_seq_0p6b_rag lw_summed_ce_0p6b_rag && touch runs/results_c.marker"

# 1.7B eval (needs the larger base under the adapter)
wait_marker "runs/kd_lora_ce_1p7b_rag/.train_done"
step eval-kd-lora-ce-1p7b runs/kd_lora_ce_1p7b_rag/eval/recite.json 5000 \
    $PY scripts/evaluate.py --experiment configs/experiments/kd_lora_ce_1p7b_rag.yaml \
        --checkpoint runs/kd_lora_ce_1p7b_rag/checkpoint

step analyze-final runs/results_final_c.marker 0 \
    bash -c "$PY scripts/analyze.py && touch runs/results_final_c.marker"

step report runs/report.pdf 0 \
    $PY scripts/report.py

log "=== overnight_c EVAL lane finished ==="
