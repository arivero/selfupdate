#!/usr/bin/env bash
# VRAM-aware multi-GPU job scheduler — THE mechanism for packing cards.
#
# Queue: scripts/queue.tsv, tab-separated:
#   done_file <TAB> need_mb <TAB> after <TAB> command [<TAB> n_gpus]
# - done_file: skip the job if this path exists (idempotency)
# - need_mb:   free VRAM required on EACH assigned device
# - after:     dependency — a done_file that must exist first ("-" = none)
# - command:   run with CUDA_VISIBLE_DEVICES set to the assigned device(s)
# - n_gpus:    optional (default 1). >1 = tensor/FSDP-parallel job: reserves
#   that many devices EXCLUSIVELY (each must be empty of scheduler jobs and
#   have need_mb free); command sees them all in CUDA_VISIBLE_DEVICES —
#   launch such jobs via `accelerate launch` / torchrun inside the command.
#
# Every cycle, for each GPU in $GPUS (default "0"): launch as many ready
# queue items as fit under the configured memory budget and process cap.
# Locks (runs/.sched/<name>) prevent duplicates; locks of dead pids are
# reaped. Edit queue.tsv at any time — it is re-read every cycle.
# On 4x L40S: GPUS="0 1 2 3" MAX_PER_GPU=3 scripts/gpu_scheduler.sh
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
[ "$SELFUPDATE_CPU_THREADS" -gt 22 ] 2>/dev/null && SELFUPDATE_CPU_THREADS=22
export OMP_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export MKL_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$SELFUPDATE_CPU_THREADS"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
GPUS="${GPUS:-0}"
MAX_PER_GPU="${MAX_PER_GPU:-3}"
MAX_MEM_FRACTION="${MAX_MEM_FRACTION:-0.75}"
MEMORY_BUDGET_MB="${MEMORY_BUDGET_MB:-}"
MARGIN_MB="${MARGIN_MB:-400}"
LAUNCHES_PER_GPU_PER_CYCLE="${LAUNCHES_PER_GPU_PER_CYCLE:-$MAX_PER_GPU}"
CYCLE_SLEEP="${CYCLE_SLEEP:-15}"
QUEUE="${QUEUE:-scripts/queue.tsv}"
SCHED="${SCHED:-runs/.sched}"
JOBLOG_DIR="${JOBLOG_DIR:-}"
mkdir -p "$SCHED"
[ -n "$JOBLOG_DIR" ] && mkdir -p "$JOBLOG_DIR"

log() { echo "[$(date '+%F %T')] sched: $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
total_mb() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1"; }

reserved_mb() {  # sum of need_mb held by live locks on device $1
    local sum=0 d pid dev need
    for d in "$SCHED"/*.lock; do
        [ -e "$d" ] || continue
        read -r pid dev need < "$d"
        kill -0 "$pid" 2>/dev/null || continue
        case ",$dev," in *",$1,"*) sum=$(( sum + ${need:-0} ));; esac
    done
    echo "$sum"
}

budget_free_mb() {
    local dev="$1" used resv total budget
    # a just-launched job allocates no VRAM while it downloads/loads, so
    # nvidia-smi alone double-books the card; treat lock reservations as
    # leases and charge whichever is larger.
    used="$(used_mb "$dev")"
    resv="$(reserved_mb "$dev")"
    [ "$resv" -gt "$used" ] && used="$resv"
    if [ -n "$MEMORY_BUDGET_MB" ]; then
        budget="$MEMORY_BUDGET_MB"
    else
        total="$(total_mb "$dev")"
        budget="$(awk -v t="$total" -v f="$MAX_MEM_FRACTION" 'BEGIN { printf "%d", t * f }')"
    fi
    echo $(( budget - used - MARGIN_MB ))
}

lock_name() { echo "$1" | sed 's|[/ ]|_|g'; }

dependency_running() {
    # A path can appear before its producing job has finished publishing it
    # (notably a checkpoint directory while save_pretrained is still writing
    # shards/tokenizer files).  The producing job's live lock is the stronger
    # readiness signal: wait for both the artifact and producer completion.
    local after="$1" lk pid
    lk="$SCHED/$(lock_name "$after").lock"
    [ -e "$lk" ] || return 1
    read -r pid _ < "$lk" || return 1
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    rm -f "$lk"
    return 1
}

running_on() {  # count live jobs assigned to device $1 (devset may be "0,1")
    local n=0 d pid dev
    for d in "$SCHED"/*.lock; do
        [ -e "$d" ] || continue
        read -r pid dev _ < "$d"
        if kill -0 "$pid" 2>/dev/null; then
            case ",$dev," in *",$1,"*) n=$((n + 1));; esac
        else
            rm -f "$d"  # reap dead lock
        fi
    done
    echo "$n"
}

exclusive_on() {  # 1 if device $1 is held by a live multi-GPU (exclusive) job
    local d pid dev
    for d in "$SCHED"/*.lock; do
        [ -e "$d" ] || continue
        read -r pid dev _ < "$d"
        kill -0 "$pid" 2>/dev/null || continue
        case "$dev" in
            *,*) case ",$dev," in *",$1,"*) echo 1; return;; esac;;
        esac
    done
    echo 0
}

echo $$ > "$SCHED/scheduler.pid"
log "started (pid $$, GPUS=$GPUS, MAX_PER_GPU=$MAX_PER_GPU, MAX_MEM_FRACTION=$MAX_MEM_FRACTION, QUEUE=$QUEUE)"
while :; do
    launched=0
    multi_launched=0
    for dev in $GPUS; do
        [ "$multi_launched" -eq 1 ] && break
        [ "$(exclusive_on "$dev")" = 1 ] && continue  # multi-GPU job owns it
        per_dev_launches=0
        fm="$(budget_free_mb "$dev")"
        while [ "$(running_on "$dev")" -lt "$MAX_PER_GPU" ] \
            && [ "$per_dev_launches" -lt "$LAUNCHES_PER_GPU_PER_CYCLE" ] \
            && [ "$fm" -gt 0 ]; do
            [ "$multi_launched" -eq 1 ] && break
            launched_this_pass=0
            while IFS=$'\t' read -r done need after cmd ngpu; do
                [ -z "$done" ] && continue
                case "$done" in \#*) continue;; esac
                [ -e "$done" ] && continue
                # A RAG generation gate writes <marker>.failed.json on a
                # scientific failure and deliberately leaves its success
                # marker absent.  Retrying the same closed gate every 15 s
                # burns a GPU without creating new evidence.  An agent must
                # inspect, repair the prompt/budget, then remove the failed
                # marker only when launching a fresh certification.
                if [[ "$cmd" == *"scripts/rag_generation_gate.py"* ]] \
                    && [ -e "$done.failed.json" ]; then
                    continue
                fi
                if [ "$after" != "-" ]; then
                    [ -e "$after" ] || continue
                    dependency_running "$after" && continue
                fi
                [ "$need" -gt "$fm" ] && continue
                ngpu="${ngpu:-1}"
                devset="$dev"
                if [ "$ngpu" -gt 1 ]; then
                    # multi-GPU (TP/FSDP) job: need $ngpu devices, each empty
                    # of scheduler jobs and under the memory-fraction budget.
                    devset=""
                    for d2 in $GPUS; do
                        [ "$(running_on "$d2")" -eq 0 ] || continue
                        [ "$(budget_free_mb "$d2")" -ge "$need" ] || continue
                        devset="${devset:+$devset,}$d2"
                        [ "$(echo "$devset" | tr ',' '\n' | wc -l)" -ge "$ngpu" ] && break
                    done
                    [ "$(echo "$devset" | tr ',' '\n' | wc -l)" -lt "$ngpu" ] && continue
                fi
                lk="$SCHED/$(lock_name "$done").lock"
                if ! (set -o noclobber; echo "$$ $devset $need" > "$lk") 2>/dev/null; then
                    continue
                fi
                log "GPU[$devset] budget_free=${fm}MB -> launch [$done] (need ${need}MB x$ngpu)"
                (
                    echo "$BASHPID $devset $need" > "$lk"
                    job_log="${JOBLOG:-runs/pipeline_sched.log}"
                    if [ -n "$JOBLOG_DIR" ]; then
                        job_log="$JOBLOG_DIR/$(lock_name "$done").log"
                    fi
                    {
                        echo "done_file=$done"
                        echo "device=$devset"
                        echo "need_mb=$need"
                        echo "command=$cmd"
                        echo "started=$(date '+%F %T')"
                    } >> "$job_log"
                    if CUDA_VISIBLE_DEVICES=$devset eval "$cmd" >> "$job_log" 2>&1; then
                        log "DONE [$done]"
                    else
                        log "FAIL [$done] (exit $?)"
                    fi
                    if [ -e "$lk" ] && read -r lock_pid _ < "$lk" 2>/dev/null && [ "$lock_pid" = "$BASHPID" ]; then
                        rm -f "$lk"
                    fi
                ) &
                launched=1
                launched_this_pass=1
                per_dev_launches=$((per_dev_launches + 1))
                fm=$((fm - need))
                # Multi-GPU launches reserve several devices at once. End the
                # scheduling cycle so later device iterations see settled locks
                # and early model-load allocations before selecting more rows.
                [ "$ngpu" -gt 1 ] && multi_launched=1
                break
            done < "$QUEUE"
            [ "$launched_this_pass" -eq 0 ] && break
        done
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
    sleep "$CYCLE_SLEEP"
done
