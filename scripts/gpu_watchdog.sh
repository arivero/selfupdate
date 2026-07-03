#!/usr/bin/env bash
# GPU watchdog: every 3 min, if utilization < 50% AND no training/eval
# process is running, launch the next backlog analysis (done-file guarded).
# Never launches trainings (double-training corrupts checkpoints).
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] watchdog: $*"; }
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1; }
busy() { pgrep -f 'scripts/(train|evaluate|layer_swap|logit_lens|report|build_teacher_cache|teacher_recite)\.py' >/dev/null; }

log "started (pid $$)"
while :; do
    sleep 180
    u=$(util)
    if [ "$u" -ge 50 ] || busy; then continue; fi
    launched=0
    while IFS=$'\t' read -r done cmd; do
        [ -z "$done" ] && continue
        [ -e "$done" ] && continue
        log "GPU idle (${u}%), launching backlog: $done"
        if eval "$cmd" >> runs/pipeline_watchdog.log 2>&1; then
            log "DONE $done"
        else
            log "FAIL $done"
        fi
        launched=1
        break
    done < scripts/watchdog_backlog.tsv
    if [ "$launched" -eq 0 ]; then
        if [ -e runs/report_final.marker ]; then log "backlog empty and final report exists; exiting"; exit 0; fi
        log "GPU idle (${u}%) but backlog exhausted; waiting"
    fi
done
