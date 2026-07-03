#!/usr/bin/env bash
# Wave C, TRAINING lane. Runs trainings back-to-back; evals happen in the
# parallel eval lane (overnight_c_eval.sh). Touches .train_done markers the
# eval lane waits on. VRAM guard keeps the two lanes from colliding.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_vram() { while [ "$(free_mb)" -lt "$1" ]; do sleep 30; done; }

train() {  # train <run_name> <needed_mb>
    local name=$1 need=$2
    if [ -e "runs/$name/.train_done" ]; then log "SKIP train-$name"; return 0; fi
    wait_vram "$need"
    log "START train-$name"
    if $PY scripts/train.py --experiment "configs/experiments/$name.yaml"; then
        touch "runs/$name/.train_done"
        log "DONE train-$name"
    else
        log "FAIL train-$name (exit $?)"
    fi
}

# wait until wave B's training is done (its eval may still run: 1.9 GB)
while pgrep -f 'scripts/train\.py' >/dev/null; do sleep 60; done
log "=== overnight_c TRAIN lane start (pid $$) ==="

train lw_summed_ce_0p6b_rag 8800   # full-FT summed: ~7.8 GB
train lw_tc_ce_0p6b_rag     4000   # LoRA online:    ~3.2 GB
train kd_lora_ce_0p6b_rag   4500   # LoRA online KD: ~3.8 GB

# stretch: Qwen3-1.7B, same KD recipe, online teacher (no cache needed)
if [ ! -e runs/model-1p7b.marker ]; then
    log "START download-1p7b"
    $PY -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B")' \
        && touch runs/model-1p7b.marker && log "DONE download-1p7b" || log "FAIL download-1p7b"
fi
train kd_lora_ce_1p7b_rag 10200    # fp32 base 6.8 GB + activations

log "=== overnight_c TRAIN lane finished ==="
