#!/usr/bin/env bash
# Wave F: compaction axis (remove vs stub vs stub_gap), hi-lr LoRA recipe.
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
while pgrep -f 'scripts/overnight_e[.]sh$' >/dev/null; do sleep 120; done
log "=== overnight_f (compaction axis) start ==="
for name in kd_lora_ce_hi_stub_0p6b_rag kd_lora_ce_hi_stubgap_0p6b_rag; do
    step "train-$name" "runs/$name/checkpoint" 2500 \
        $PY scripts/train.py --experiment "configs/experiments/$name.yaml"
    step "eval-$name" "runs/$name/eval/recite.json" 2500 \
        $PY scripts/evaluate.py --experiment "configs/experiments/$name.yaml" --checkpoint "runs/$name/checkpoint"
done
step report-f runs/report_final.marker 0 \
    bash -c "$PY scripts/analyze.py && $PY scripts/report.py && touch runs/report_final.marker"
log "=== overnight_f finished ==="
