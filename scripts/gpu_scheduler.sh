#!/usr/bin/env bash
# Shared-pool VRAM-aware GPU scheduler.
#
# Queue rows are tab-separated and remain backward compatible:
#   done_file need_mb after command [n_gpus [priority [expected_seconds [cache_group]]]]
# Higher priority wins within a class.  Multi-GPU rows are always considered
# before one-GPU rows; expected_seconds lets short jobs backfill while a gang
# is waiting without allowing a long one-GPU job to occupy a needed card.

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
BACKFILL_MAX_SECONDS="${BACKFILL_MAX_SECONDS:-300}"
QUEUE="${QUEUE:-scripts/queue.tsv}"
# SCHED remains the per-scheduler marker for compatibility.  GPU leases are
# deliberately outside it so schedulers on the same or different nodes share
# one allocator pool.
SCHED="${SCHED:-runs/.sched}"
GPU_LEASE_ROOT="${GPU_LEASE_ROOT:-runs/.gpu-leases}"
JOBLOG_DIR="${JOBLOG_DIR:-}"
export GPU_LEASE_ROOT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/gpu_lease.sh
. "$SCRIPT_DIR/gpu_lease.sh"

mkdir -p "$SCHED" "$GPU_LEASE_ROOT"
[ -n "$JOBLOG_DIR" ] && mkdir -p "$JOBLOG_DIR"

log() { echo "[$(date '+%F %T')] sched: $*"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
used_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"; }
total_mb() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1"; }

budget_free_mb_locked() {
    local dev="$1" used resv total budget
    used="$(used_mb "$dev")"
    resv="$(gpu_lease_reserved_locked "$dev")"
    [ "$resv" -gt "$used" ] && used="$resv"
    if [ -n "$MEMORY_BUDGET_MB" ]; then
        budget="$MEMORY_BUDGET_MB"
    else
        total="$(total_mb "$dev")"
        budget="$(awk -v t="$total" -v f="$MAX_MEM_FRACTION" 'BEGIN { printf "%d", t * f }')"
    fi
    echo $((budget - used - MARGIN_MB))
}

dependency_running() {
    local after="$1"
    gpu_lease_job_running "$after"
}

row_ready() {
    local done="$1" after="$2" cmd="$3"
    [ -n "$done" ] || return 1
    [ -e "$done" ] && return 1
    if [[ "$cmd" == *"scripts/rag_generation_gate.py" || \
          "$cmd" == *"scripts/cache_generation_gate.py" ]] &&
        [ -e "$done.failed.json" ]; then
        return 1
    fi
    if [ "$after" != "-" ]; then
        [ -e "$after" ] || return 1
        dependency_running "$after" && return 1
    fi
    return 0
}

queue_order() {
    local out="$1"
    awk -F '\t' '
        /^[[:space:]]*#/ || $1 == "" { next }
        {
          ng = ($5 ~ /^[0-9]+$/ && $5 > 0) ? $5 + 0 : 1
          pr = ($6 ~ /^-?[0-9]+$/) ? $6 + 0 : 0
          ex = ($7 ~ /^[0-9]+$/ && $7 > 0) ? $7 + 0 : 2147483647
          gang = (ng > 1) ? 1 : 0
          cg = ($8 == "" ? "-" : $8)
          printf "%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%s\n", \
             gang, pr, ex, NR, $1, $2, $3, $4, ng, pr, ($7 == "" ? 0 : $7), cg
        }
    ' "$QUEUE" | LC_ALL=C sort -t $'\t' -k1,1nr -k2,2nr -k3,3n -k4,4n > "$out"
}

has_ready_gang() {
    local rank priority expected line done need after cmd ngpu row_priority row_expected cache_group
    local d2 seen="" gpu_count=0
    for d2 in $GPUS; do
        case " $seen " in *" $d2 "*) continue;; esac
        seen="$seen $d2"
        gpu_count=$((gpu_count + 1))
    done
    while IFS=$'\t' read -r rank priority expected line done need after cmd ngpu \
        row_priority row_expected cache_group; do
        [ "${ngpu:-1}" -gt 1 ] 2>/dev/null || continue
        [ "$ngpu" -le "$gpu_count" ] || continue
        row_ready "$done" "$after" "$cmd" || continue
        return 0
    done < "$QUEUE_ORDER"
    return 1
}

choose_devset_locked() {
    local requested="$1" need="$2" preferred="$3" d2 devset="" count=0 seen=""
    if [ "$requested" -le 1 ]; then
        [ "$(gpu_lease_count_locked "$preferred")" -lt "$MAX_PER_GPU" ] || return 1
        [ "$(gpu_lease_exclusive_locked "$preferred")" = 0 ] || return 1
        [ "$(budget_free_mb_locked "$preferred")" -ge "$need" ] || return 1
        printf '%s\n' "$preferred"
        return 0
    fi
    for d2 in $GPUS; do
        case " $seen " in *" $d2 "*) continue;; esac
        seen="$seen $d2"
        [ "$(gpu_lease_count_locked "$d2")" -eq 0 ] || continue
        [ "$(budget_free_mb_locked "$d2")" -ge "$need" ] || continue
        devset="${devset:+$devset,}$d2"
        count=$((count + 1))
        [ "$count" -ge "$requested" ] && break
    done
    [ "$count" -ge "$requested" ] || return 1
    printf '%s\n' "$devset"
}

claim_row() {
    local done="$1" need="$2" cmd="$3" ngpu="$4" priority="$5"
    local expected_seconds="$6" cache_group="$7" preferred="$8" devset lease_id
    local launcher_start
    case "$need" in ''|*[!0-9]*) return 1;; esac
    case "$ngpu" in ''|*[!0-9]*) ngpu=1;; esac
    [ "$ngpu" -gt 0 ] || ngpu=1
    gpu_lease_mutex_acquire
    gpu_lease_reap_stale_locked
    # The same done_file can be present in multiple queue readers.  Check it
    # under the allocator mutex so only one scheduler may claim it.
    if gpu_lease_job_running "$done"; then
        gpu_lease_mutex_release_if_owner
        return 1
    fi
    devset="$(choose_devset_locked "$ngpu" "$need" "$preferred")"
    if [ -z "$devset" ]; then
        gpu_lease_mutex_release_if_owner
        return 1
    fi
    launcher_start="$(gpu_lease_proc_start "$$" || true)"
    lease_id="$(gpu_lease_key "$GPU_LEASE_HOST.$$.${BASHPID:-$$}.$RANDOM.$(date +%s%N)")"
    if ! gpu_lease_claim_locked "$devset" "$need" "$$" "$launcher_start" \
        "$(gpu_lease_key "$done")" "$done" "${expected_seconds:-0}" \
        "${cache_group:--}" "$lease_id"; then
        gpu_lease_mutex_release_if_owner
        return 1
    fi
    gpu_lease_mutex_release_if_owner
    CLAIMED_LEASE_ID="$lease_id"
    CLAIMED_DEVSET="$devset"
    CLAIMED_NGPU="$ngpu"
    return 0
}

worker_main() {
    local lease_id="$1" devset="$2" done="$3" need="$4" cmd="$5" ngpu="$6"
    local job_log rc worker_pid worker_start
    worker_pid="${BASHPID:-$$}"
    worker_start="$(gpu_lease_proc_start "$worker_pid" || true)"
    trap 'gpu_lease_release "$lease_id" "$worker_pid"' EXIT HUP INT TERM
    if ! gpu_lease_handoff "$lease_id" "$worker_pid" "$worker_start"; then
        log "FAIL [$done] (lease handoff failed)"
        return 125
    fi
    job_log="${JOBLOG:-runs/pipeline_sched.log}"
    if [ -n "$JOBLOG_DIR" ]; then
        job_log="$JOBLOG_DIR/$(gpu_lease_key "$done").log"
    fi
    {
        echo "done_file=$done"
        echo "job=$(gpu_lease_key "$done")"
        echo "device=$devset"
        echo "need_mb=$need"
        echo "n_gpus=$ngpu"
        echo "command=$cmd"
        echo "started=$(date '+%F %T')"
    } >> "$job_log"
    if CUDA_VISIBLE_DEVICES="$devset" eval "$cmd" >> "$job_log" 2>&1; then
        log "DONE [$done]"
        rc=0
    else
        rc=$?
        log "FAIL [$done] (exit $rc)"
    fi
    return "$rc"
}

launch_row() {
    local lease_id="$1" devset="$2" done="$3" need="$4" cmd="$5" ngpu="$6"
    (
        # Do not inherit the scheduler's cleanup trap into the worker.
        trap - EXIT HUP INT TERM
        worker_main "$lease_id" "$devset" "$done" "$need" "$cmd" "$ngpu"
    ) &
}

cleanup_scheduler() {
    local rc=$?
    trap - EXIT HUP INT TERM
    gpu_lease_release_launcher "$$" "$(gpu_lease_proc_start "$$" || true)"
    gpu_lease_mutex_release_if_owner
    [ -n "${QUEUE_ORDER:-}" ] && rm -f "$QUEUE_ORDER"
    exit "$rc"
}

echo $$ > "$SCHED/scheduler.pid"
trap cleanup_scheduler EXIT HUP INT TERM
log "started (pid $$, host=$GPU_LEASE_HOST, GPUS=$GPUS, MAX_PER_GPU=$MAX_PER_GPU, MAX_MEM_FRACTION=$MAX_MEM_FRACTION, QUEUE=$QUEUE, GPU_LEASE_ROOT=$GPU_LEASE_ROOT)"

QUEUE_ORDER="$(mktemp "$GPU_LEASE_ROOT/.queue-order.XXXXXX")"
while :; do
    launched=0
    multi_launched=0
    gpu_lease_mutex_acquire
    gpu_lease_reap_stale_locked
    gpu_lease_mutex_release_if_owner
    queue_order "$QUEUE_ORDER" || { log "queue read failed: $QUEUE"; sleep "$CYCLE_SLEEP"; continue; }
    gang_pending=0
    has_ready_gang && gang_pending=1

    for dev in $GPUS; do
        [ "$multi_launched" -eq 1 ] && break
        per_dev_launches=0
        while [ "$per_dev_launches" -lt "$LAUNCHES_PER_GPU_PER_CYCLE" ]; do
            launched_this_pass=0
            while IFS=$'\t' read -r rank priority expected line done need after cmd ngpu \
                row_priority row_expected cache_group; do
                [ -n "$done" ] || continue
                row_ready "$done" "$after" "$cmd" || continue
                ngpu="${ngpu:-1}"
                [ "$ngpu" -gt 0 ] 2>/dev/null || ngpu=1
                if [ "$ngpu" -eq 1 ] && [ "$gang_pending" -eq 1 ]; then
                    # Missing/invalid expected_seconds sorts as long.  It is
                    # safer to leave a card available for a pending gang than
                    # to let an unbounded one-GPU job occupy it indefinitely.
                    backfill_seconds="${row_expected:-0}"
                    case "$backfill_seconds" in ''|*[!0-9]*) backfill_seconds=0;; esac
                    if [ "$backfill_seconds" -eq 0 ] ||
                        [ "$backfill_seconds" -gt "$BACKFILL_MAX_SECONDS" ]; then
                        continue
                    fi
                fi
                if claim_row "$done" "$need" "$cmd" "$ngpu" \
                    "${row_priority:-0}" "${row_expected:-0}" \
                    "${cache_group:--}" "$dev"; then
                    lease_id="$CLAIMED_LEASE_ID"
                    devset="$CLAIMED_DEVSET"
                    launch_row "$lease_id" "$devset" "$done" "$need" "$cmd" "$ngpu"
                    launched=1
                    launched_this_pass=1
                    per_dev_launches=$((per_dev_launches + 1))
                    [ "$ngpu" -gt 1 ] && multi_launched=1
                    break
                fi
            done < "$QUEUE_ORDER"
            [ "$launched_this_pass" -eq 1 ] || break
            [ "$multi_launched" -eq 1 ] && break
        done
    done

    # Only this scheduler's workers keep the scheduler PID in launcher_pid;
    # unrelated schedulers may continue using the shared pool after this queue
    # drains.
    if [ "$launched" -eq 0 ]; then
        if ! gpu_lease_scheduler_busy "$$"; then
            pending=0
            while IFS=$'\t' read -r rank priority expected line done need after cmd ngpu \
                row_priority row_expected cache_group; do
                [ -n "$done" ] || continue
                [ -e "$done" ] || { pending=1; break; }
            done < "$QUEUE_ORDER"
            if [ "$pending" -eq 0 ]; then
                log "queue drained; exiting"
                exit 0
            fi
        fi
    fi
    sleep "$CYCLE_SLEEP"
done
