#!/usr/bin/env bash
# Wave E: v2 extended-recitation dataset (paraphrases, 24/48-verse windows,
# part-level chunks) + chained full-poem eval. Waits for overnight_d2.
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
while pgrep -f 'scripts/overnight_d2\.sh$' >/dev/null; do sleep 120; done
log "=== overnight_e (v2 data) start ==="

step cache-v2 caches/.v2.marker 8000 \
    bash -c "$PY scripts/build_teacher_cache.py --experiment configs/experiments/kd_ce_v2_0p6b_rag.yaml && touch caches/.v2.marker"

step train-kd-lora-ce-hi-v2 runs/kd_lora_ce_hi_v2_0p6b_rag/checkpoint 2500 \
    $PY scripts/train.py --experiment configs/experiments/kd_lora_ce_hi_v2_0p6b_rag.yaml
step eval-kd-lora-ce-hi-v2 runs/kd_lora_ce_hi_v2_0p6b_rag/eval/recite.json 2500 \
    $PY scripts/evaluate.py --experiment configs/experiments/kd_lora_ce_hi_v2_0p6b_rag.yaml --checkpoint runs/kd_lora_ce_hi_v2_0p6b_rag/checkpoint
step chain-kd-lora-ce-hi-v2 runs/kd_lora_ce_hi_v2_0p6b_rag/eval/recite_long.json 2500 \
    $PY scripts/recite_long.py --checkpoint runs/kd_lora_ce_hi_v2_0p6b_rag/checkpoint

step train-kd-ce-v2 runs/kd_ce_v2_0p6b_rag/checkpoint 9500 \
    $PY scripts/train.py --experiment configs/experiments/kd_ce_v2_0p6b_rag.yaml
step eval-kd-ce-v2 runs/kd_ce_v2_0p6b_rag/eval/recite.json 2500 \
    $PY scripts/evaluate.py --experiment configs/experiments/kd_ce_v2_0p6b_rag.yaml --checkpoint runs/kd_ce_v2_0p6b_rag/checkpoint
step chain-kd-ce-v2 runs/kd_ce_v2_0p6b_rag/eval/recite_long.json 2500 \
    $PY scripts/recite_long.py --checkpoint runs/kd_ce_v2_0p6b_rag/checkpoint

step chain-kd-ce-v1 runs/kd_ce_0p6b_rag/eval/recite_long.json 2500 \
    $PY scripts/recite_long.py --checkpoint runs/kd_ce_0p6b_rag/checkpoint

step report-e runs/report_e.marker 0 \
    bash -c "$PY scripts/analyze.py && $PY scripts/report.py && touch runs/report_e.marker"
log "=== overnight_e finished ==="
