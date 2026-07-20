#!/usr/bin/env bash
# GPU watchdog v2: every 3 min, if utilization < 50% and a backlog item's
# VRAM requirement fits in free memory, launch it — even alongside one
# running job (at most 2 concurrent GPU processes). Never duplicates work:
# done-file guards + one backlog item in flight at a time (.watchdog_running).
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] watchdog: $*"; }
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
nproc_gpu() { pgrep -cf 'compressed/(train|evaluate|layer_swap|logit_lens|report|build_teacher_cache|teacher_ceiling|recite_long)[.]py'; }

LOCK=runs/.watchdog_running
rm -f "$LOCK"
log "started v2 (pid $$)"
while :; do
    sleep 180
    [ -e "$LOCK" ] && continue
    u=$(util)
    [ "$u" -ge 50 ] && continue
    [ "$(nproc_gpu)" -ge 2 ] && continue
    fm=$(free_mb)
    while IFS=$'\t' read -r done need cmd; do
        [ -z "$done" ] && continue
        [ -e "$done" ] && continue
        [ "$fm" -lt "$need" ] && continue
        log "util ${u}%, free ${fm}MB -> launching $done (needs ${need}MB)"
        touch "$LOCK"
        ( eval "$cmd" >> runs/pipeline_watchdog.log 2>&1 \
            && log "DONE $done" || log "FAIL $done"; rm -f "$LOCK" ) &
        break
    done < scripts/watchdog_backlog.tsv
    if [ ! -e "$LOCK" ] && [ -e runs/report_final.marker ]; then
        log "backlog drained and final report exists; exiting"; exit 0
    fi
done
