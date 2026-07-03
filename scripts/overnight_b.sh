#!/usr/bin/env bash
# Follow-on comparison: layerwise (a) student-stream vs (b) censored teacher
# stream, both LoRA + online teacher. Waits for overnight.sh to finish, then
# runs the pair and refreshes the analysis. Same idempotent step guards.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*"; }

step() {
    local name=$1 done_file=$2; shift 2
    if [ -e "$done_file" ]; then log "SKIP $name"; return 0; fi
    log "START $name"
    if "$@"; then log "DONE $name"; else log "FAIL $name (exit $?)"; fi
}

# wait for the first pipeline (script process itself, no step race)
while pgrep -f 'scripts/overnight\.sh$' >/dev/null; do
    sleep 60
done

log "=== overnight_b comparison start (pid $$) ==="

step train-lw-summed-lora runs/lw_summed_lora_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_summed_lora_0p6b_rag.yaml

step eval-lw-summed-lora runs/lw_summed_lora_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_summed_lora_0p6b_rag/checkpoint

step train-lw-tc-lora runs/lw_tc_lora_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_tc_lora_0p6b_rag.yaml

step eval-lw-tc-lora runs/lw_tc_lora_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_tc_lora_0p6b_rag/checkpoint

step analyze-final runs/results_final.marker \
    bash -c "$PY scripts/analyze.py && touch runs/results_final.marker"

log "=== overnight_b finished ==="
