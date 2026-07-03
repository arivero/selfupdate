#!/usr/bin/env bash
# Round 2: proper LoRA learning rate, longer training. Waits for wave C.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_vram() { sleep $((RANDOM % 40 + 20)); while [ "$(free_mb)" -lt "$1" ]; do sleep 30; done; }
step() {
    local name=$1 done_file=$2 need=$3; shift 3
    if [ -e "$done_file" ]; then log "SKIP $name"; return 0; fi
    wait_vram "$need"
    log "START $name"
    if "$@"; then log "DONE $name"; else log "FAIL $name (exit $?)"; fi
}
while pgrep -f 'scripts/overnight_c_(train|eval)\.sh$' >/dev/null; do sleep 120; done
log "=== overnight_d (round 2) start ==="
for name in kd_lora_ce_hi_0p6b_rag lw_tc_ce_hi_0p6b_rag kd_ce_long_0p6b_rag; do
    step "train-$name" "runs/$name/checkpoint" 9500 \
        $PY scripts/train.py --experiment "configs/experiments/$name.yaml"
    step "eval-$name" "runs/$name/eval/recite.json" 2500 \
        $PY scripts/evaluate.py --checkpoint "runs/$name/checkpoint"
done
step analyze-d runs/results_d.marker 0 \
    bash -c "$PY scripts/analyze.py --deltas kd_full_0p6b_rag kd_ce_0p6b_rag kd_ce_long_0p6b_rag lw_summed_0p6b_rag lw_seq_0p6b_rag lw_summed_ce_0p6b_rag && $PY scripts/report.py && touch runs/results_d.marker"
log "=== overnight_d finished ==="
