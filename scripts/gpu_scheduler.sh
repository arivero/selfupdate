#!/usr/bin/env bash
# VRAM-aware multi-GPU job scheduler — THE mechanism for packing cards.
#
# Queue: scripts/queue.tsv, tab-separated:
#   done_file <TAB> need_mb <TAB> after <TAB> command
# - done_file: skip the job if this path exists (idempotency)
# - need_mb:   free VRAM required on a device to launch
# - after:     dependency — a done_file that must exist first ("-" = none)
# - command:   run with CUDA_VISIBLE_DEVICES set to the chosen device
#
# Every 60 s, for each GPU in $GPUS (default "0"): if fewer than
# $MAX_PER_GPU jobs are running there and a queue item fits in free VRAM
# (with margin), launch it. Locks (runs/.sched/<name>) prevent duplicates;
# locks of dead pids are reaped. Edit queue.tsv at any time — it is re-read
# every cycle. On 4x L40S: GPUS="0 1 2 3" MAX_PER_GPU=3 scripts/gpu_scheduler.sh
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
GPUS="${GPUS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-3}"
MARGIN_MB=400
QUEUE=scripts/queue.tsv
SCHED=runs/.sched
mkdir -p "$SCHED"

log() { echo "[$(date '+%F %T')] sched: $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }

lock_name() { echo "$1" | sed 's|[/ ]|_|g'; }

running_on() {  # count live jobs assigned to device $1
    local n=0 d
    for d in "$SCHED"/*.lock; do
        [ -e "$d" ] || continue
        read -r pid dev < "$d"
        if kill -0 "$pid" 2>/dev/null; then
            [ "$dev" = "$1" ] && n=$((n + 1))
        else
            rm -f "$d"  # reap dead lock
        fi
    done
    echo "$n"
}

log "started (pid $$, GPUS=$GPUS, MAX_PER_GPU=$MAX_PER_GPU)"
while :; do
    launched=0
    for dev in $GPUS; do
        [ "$(running_on "$dev")" -ge "$MAX_PER_GPU" ] && continue
        fm=$(( $(free_mb "$dev") - MARGIN_MB ))
        while IFS=$'\t' read -r done need after cmd; do
            [ -z "$done" ] && continue
            case "$done" in \#*) continue;; esac
            [ -e "$done" ] && continue
            [ "$after" != "-" ] && [ ! -e "$after" ] && continue
            [ "$need" -gt "$fm" ] && continue
            lk="$SCHED/$(lock_name "$done").lock"
            [ -e "$lk" ] && continue
            echo "$$ $dev" > "$lk"   # provisional; child pid written below
            log "GPU$dev free=${fm}MB -> launch [$done] (need ${need}MB)"
            (
                echo "$BASHPID $dev" > "$lk"
                if CUDA_VISIBLE_DEVICES=$dev eval "$cmd" >> runs/pipeline_sched.log 2>&1; then
                    log "DONE [$done]"
                else
                    log "FAIL [$done] (exit $?)"
                fi
                rm -f "$lk"
            ) &
            launched=1
            break   # one launch per device per cycle; re-check VRAM next round
        done < "$QUEUE"
    done
    # exit when queue fully done and nothing running
    if [ "$launched" -eq 0 ]; then
        busy=0
        for d in "$SCHED"/*.lock; do [ -e "$d" ] && busy=1 && break; done
        if [ "$busy" -eq 0 ]; then
            pending=0
            while IFS=$'\t' read -r done need after cmd; do
                [ -z "$done" ] && continue
                case "$done" in \#*) continue;; esac
                [ -e "$done" ] || pending=1
            done < "$QUEUE"
            if [ "$pending" -eq 0 ]; then log "queue drained; exiting"; exit 0; fi
        fi
    fi
    sleep 60
done
