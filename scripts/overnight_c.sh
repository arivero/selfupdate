#!/usr/bin/env bash
# Third detached wave: causal localization on the reciting checkpoint,
# hybrid layerwise runs (local last-block CE), the LoRA+online KD-CE recipe,
# and a stretch replication at Qwen3-1.7B. Waits for overnight_b.

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

while pgrep -f 'scripts/overnight_b\.sh$' >/dev/null; do
    sleep 60
done

log "=== overnight_c start (pid $$) ==="

# 1. causal localization on the checkpoint that actually recites
step layer-swap-kd-ce runs/kd_ce_0p6b_rag/eval/layer_swap.csv \
    $PY scripts/layer_swap.py --run kd_ce_0p6b_rag --limit 8

step logit-lens-kd-ce runs/kd_ce_0p6b_rag/eval/logit_lens.csv \
    $PY scripts/logit_lens.py --run kd_ce_0p6b_rag --limit 24

# 2. hybrid layerwise: local gold-CE on the last block only
step train-lw-summed-ce runs/lw_summed_ce_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_summed_ce_0p6b_rag.yaml

step eval-lw-summed-ce runs/lw_summed_ce_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_summed_ce_0p6b_rag/checkpoint

step train-lw-tc-ce runs/lw_tc_ce_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/lw_tc_ce_0p6b_rag.yaml

step eval-lw-tc-ce runs/lw_tc_ce_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/lw_tc_ce_0p6b_rag/checkpoint

# 3. the winning KD recipe in adapters (personalization shape), online teacher
step train-kd-lora-ce runs/kd_lora_ce_0p6b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/kd_lora_ce_0p6b_rag.yaml

step eval-kd-lora-ce runs/kd_lora_ce_0p6b_rag/eval/recite.json \
    $PY scripts/evaluate.py --checkpoint runs/kd_lora_ce_0p6b_rag/checkpoint

# 4. refresh analysis (cosines for full-FT runs; profiles incl. LoRA)
step analyze-c runs/results_c.marker \
    bash -c "$PY scripts/analyze.py --deltas kd_full_0p6b_rag kd_ce_0p6b_rag lw_summed_0p6b_rag lw_seq_0p6b_rag lw_summed_ce_0p6b_rag && touch runs/results_c.marker"

# 5. stretch: second model, Qwen3-1.7B, same recipe, no cache needed (online)
step download-1p7b runs/model-1p7b.marker \
    bash -c "$PY -c 'from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-1.7B\")' && touch runs/model-1p7b.marker"

step teacher-recite-1p7b runs/teacher-recite-1p7b.marker \
    bash -c "$PY scripts/teacher_recite.py --experiment configs/experiments/kd_lora_ce_1p7b_rag.yaml --limit 8 --show 0 && touch runs/teacher-recite-1p7b.marker"

step train-kd-lora-ce-1p7b runs/kd_lora_ce_1p7b_rag/checkpoint \
    $PY scripts/train.py --experiment configs/experiments/kd_lora_ce_1p7b_rag.yaml

step eval-kd-lora-ce-1p7b runs/kd_lora_ce_1p7b_rag/eval/recite.json \
    $PY scripts/evaluate.py --experiment configs/experiments/kd_lora_ce_1p7b_rag.yaml --checkpoint runs/kd_lora_ce_1p7b_rag/checkpoint

log "=== overnight_c finished ==="
