#!/usr/bin/env bash
# Round 2 on the second model: hi-lr LoRA recipe at Qwen3-1.7B. Waits for overnight_d.
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
while pgrep -f 'scripts/overnight_d\.sh$' >/dev/null; do sleep 120; done
log "=== overnight_d2 (round 2 @ 1.7B) start ==="
step train-kd-lora-ce-hi-1p7b runs/kd_lora_ce_hi_1p7b_rag/checkpoint 4500 \
    $PY scripts/train.py --experiment configs/experiments/kd_lora_ce_hi_1p7b_rag.yaml
step eval-kd-lora-ce-hi-1p7b runs/kd_lora_ce_hi_1p7b_rag/eval/recite.json 5000 \
    $PY scripts/evaluate.py --experiment configs/experiments/kd_lora_ce_hi_1p7b_rag.yaml \
        --checkpoint runs/kd_lora_ce_hi_1p7b_rag/checkpoint
step report-final runs/report_final.marker 0 \
    bash -c "$PY scripts/analyze.py && $PY scripts/report.py && touch runs/report_final.marker"
log "=== overnight_d2 finished ==="
