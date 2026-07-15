#!/usr/bin/env bash
# Shared, filesystem-backed GPU leases for gpu_scheduler.sh.
#
# The allocator mutex is a mkdir lock.  All claim, handoff, reap, and release
# operations which can change a lease happen while it is held.  A lease is
# represented by one inspectable metadata file per physical GPU.

GPU_LEASE_HOST="${GPU_LEASE_HOST:-$(hostname -s)}"
GPU_LEASE_ROOT="${GPU_LEASE_ROOT:-runs/.gpu-leases}"
GPU_LEASE_MUTEX_WAIT="${GPU_LEASE_MUTEX_WAIT:-0.10}"

gpu_lease_init() {
    mkdir -p "$GPU_LEASE_ROOT"
}

gpu_lease_key() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//$'\t'/_}"
    printf '%s' "$value"
}

gpu_lease_lock_path() {
    local gpu="$1" lease_id="$2"
    printf '%s/gpu.%s.%s.%s.lock' "$GPU_LEASE_ROOT" "$GPU_LEASE_HOST" \
        "$gpu" "$lease_id"
}

gpu_lease_field() {
    local key="$1" path="$2"
    awk -F= -v want="$key" '$1 == want { sub(/^[^=]*=/, ""); print; exit }' "$path" 2>/dev/null
}

gpu_lease_proc_start() {
    local pid="$1"
    [ -r "/proc/$pid/stat" ] || return 1
    awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

gpu_lease_pid_live() {
    local host="$1" pid="$2" start="$3" actual
    case "$pid" in ''|*[!0-9]*) return 1;; esac
    if [ "$host" != "$GPU_LEASE_HOST" ]; then
        # A shared filesystem cannot inspect a remote host's PID namespace.
        # Remote leases are therefore conservative; local leases are checked
        # below with both kill -0 and the kernel process-start tick.
        return 0
    fi
    kill -0 "$pid" 2>/dev/null || return 1
    [ -z "$start" ] && return 0
    actual="$(gpu_lease_proc_start "$pid" || true)"
    [ -n "$actual" ] && [ "$actual" = "$start" ]
}

gpu_lease_mutex_owner_path() {
    printf '%s/.allocator.lock/owner' "$GPU_LEASE_ROOT"
}

gpu_lease_mutex_release_if_owner() {
    local owner host pid start actual self_pid
    owner="$(gpu_lease_mutex_owner_path)"
    # The owner and lock directory are removed together by the releasing
    # process.  Treat a vanished owner as an ordinary race, not shell noise.
    read -r host pid start 2>/dev/null < "$owner" || return 0
    [ "$host" = "$GPU_LEASE_HOST" ] || return 0
    self_pid="${BASHPID:-$$}"
    actual="$(gpu_lease_proc_start "$self_pid" || true)"
    [ "$pid" = "$self_pid" ] || return 0
    [ -z "$start" ] || [ "$start" = "$actual" ] || return 0
    rm -f "$owner"
    rmdir "${owner%/owner}" 2>/dev/null || true
}

gpu_lease_mutex_acquire() {
    local mutex owner tmp host pid start actual self_pid
    mutex="$GPU_LEASE_ROOT/.allocator.lock"
    owner="$(gpu_lease_mutex_owner_path)"
    mkdir -p "$GPU_LEASE_ROOT"
    while ! mkdir "$mutex" 2>/dev/null; do
        if read -r host pid start 2>/dev/null < "$owner"; then
            if [ "$host" = "$GPU_LEASE_HOST" ] &&
                ! gpu_lease_pid_live "$host" "$pid" "$start"; then
                rm -f "$owner"
                rmdir "$mutex" 2>/dev/null || true
                continue
            fi
        fi
        sleep "$GPU_LEASE_MUTEX_WAIT"
    done
    tmp="$mutex/owner.$$.$BASHPID.tmp"
    self_pid="${BASHPID:-$$}"
    printf '%s %s %s\n' "$GPU_LEASE_HOST" "$self_pid" \
        "$(gpu_lease_proc_start "$self_pid" || true)" > "$tmp"
    mv -f "$tmp" "$owner"
}

gpu_lease_reap_stale_locked() {
    local path state host pid start
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        state="$(gpu_lease_field state "$path")"
        host="$(gpu_lease_field hostname "$path")"
        if [ "$state" = launcher ]; then
            pid="$(gpu_lease_field launcher_pid "$path")"
            start="$(gpu_lease_field launcher_start "$path")"
        elif [ "$state" = worker ]; then
            pid="$(gpu_lease_field worker_pid "$path")"
            start="$(gpu_lease_field worker_start "$path")"
        else
            continue
        fi
        # Only local PID namespaces are safely reapable.  A malformed or
        # remote lock is left visible for an operator to diagnose.
        if [ "$host" = "$GPU_LEASE_HOST" ] &&
            ! gpu_lease_pid_live "$host" "$pid" "$start"; then
            rm -f "$path"
        fi
    done
}

gpu_lease_count_locked() {
    local gpu="$1" path count=0 dev host
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        host="$(gpu_lease_field hostname "$path")"
        [ "$host" = "$GPU_LEASE_HOST" ] || continue
        dev="$(gpu_lease_field gpu "$path")"
        [ "$dev" = "$gpu" ] && count=$((count + 1))
    done
    printf '%s\n' "$count"
}

gpu_lease_reserved_locked() {
    local gpu="$1" path dev host need sum=0
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        host="$(gpu_lease_field hostname "$path")"
        [ "$host" = "$GPU_LEASE_HOST" ] || continue
        dev="$(gpu_lease_field gpu "$path")"
        [ "$dev" = "$gpu" ] || continue
        need="$(gpu_lease_field need "$path")"
        case "$need" in ''|*[!0-9]*) need=0;; esac
        sum=$((sum + need))
    done
    printf '%s\n' "$sum"
}

gpu_lease_exclusive_locked() {
    local gpu="$1" path devset host
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        host="$(gpu_lease_field hostname "$path")"
        [ "$host" = "$GPU_LEASE_HOST" ] || continue
        devset="$(gpu_lease_field devset "$path")"
        case ",$devset," in *",$gpu,"*)
            case "$devset" in *,*) printf '1\n'; return 0;; esac
        esac
    done
    printf '0\n'
}

gpu_lease_job_running() {
    local done="$1" path state value
    # Deliberately global across hosts: every node may read the same Lustre
    # queue, but a done-file identity may have only one live owner.
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        value="$(gpu_lease_field done "$path")"
        [ "$value" = "$done" ] || continue
        state="$(gpu_lease_field state "$path")"
        [ "$state" = launcher ] || [ "$state" = worker ] || continue
        return 0
    done
    return 1
}

gpu_lease_scheduler_busy() {
    local launcher="$1" path value host
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        host="$(gpu_lease_field hostname "$path")"
        [ "$host" = "$GPU_LEASE_HOST" ] || continue
        value="$(gpu_lease_field launcher_pid "$path")"
        [ "$value" = "$launcher" ] && return 0
    done
    return 1
}

gpu_lease_write_metadata() {
    local path="$1" gpu="$2" state="$3" worker_pid="$4" worker_start="$5"
    local launcher_pid="$6" launcher_start="$7" lease_id="$8" job="$9"
    local done="${10}" need="${11}" devset="${12}" start="${13}"
    local expected="${14}" cache_group="${15}" tmp
    tmp="$path.$$.${BASHPID}.tmp"
    {
        printf 'state=%s\n' "$state"
        printf 'hostname=%s\n' "$GPU_LEASE_HOST"
        printf 'gpu=%s\n' "$gpu"
        printf 'launcher_pid=%s\n' "$launcher_pid"
        printf 'worker_pid=%s\n' "$worker_pid"
        printf 'launcher_start=%s\n' "$launcher_start"
        printf 'worker_start=%s\n' "$worker_start"
        printf 'job=%s\n' "$job"
        printf 'done=%s\n' "$done"
        printf 'need=%s\n' "$need"
        printf 'devset=%s\n' "$devset"
        printf 'start=%s\n' "$start"
        printf 'expected_seconds=%s\n' "$expected"
        printf 'cache_group=%s\n' "$cache_group"
        printf 'lease_id=%s\n' "$lease_id"
    } > "$tmp" || return 1
    mv -f "$tmp" "$path"
}

gpu_lease_claim_locked() {
    local devset="$1" need="$2" launcher_pid="$3" launcher_start="$4"
    local job="$5" done="$6" expected="$7" cache_group="$8" lease_id="$9"
    local start gpu path txn count=0 staged
    start="$(date +%s)"
    txn="$GPU_LEASE_ROOT/.claim.$lease_id"
    mkdir "$txn" 2>/dev/null || return 1

    for gpu in ${devset//,/ }; do
        path="$(gpu_lease_lock_path "$gpu" "$lease_id")"
        [ ! -e "$path" ] || { rmdir "$txn" 2>/dev/null || true; return 1; }
        staged="$txn/gpu.$gpu.lock"
        gpu_lease_write_metadata "$staged" "$gpu" launcher 0 "" \
            "$launcher_pid" "$launcher_start" "$lease_id" "$job" "$done" \
            "$need" "$devset" "$start" "$expected" "$cache_group" || {
            rm -f "$txn"/*
            rmdir "$txn" 2>/dev/null || true
            return 1
        }
        count=$((count + 1))
    done
    [ "$count" -gt 0 ] || { rmdir "$txn" 2>/dev/null || true; return 1; }

    # The allocator mutex makes this a transaction from every other
    # allocator's point of view.  Each destination is still installed with a
    # rename, never a truncating write; a failed rename rolls back our group.
    for gpu in ${devset//,/ }; do
        path="$(gpu_lease_lock_path "$gpu" "$lease_id")"
        staged="$txn/gpu.$gpu.lock"
        if ! mv -f "$staged" "$path"; then
            for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
                [ -e "$path" ] || continue
                [ "$(gpu_lease_field lease_id "$path")" = "$lease_id" ] && rm -f "$path"
            done
            rm -f "$txn"/*
            rmdir "$txn" 2>/dev/null || true
            return 1
        fi
    done
    rmdir "$txn" 2>/dev/null || true
    GPU_LEASE_LAST_ID="$lease_id"
    return 0
}

gpu_lease_handoff() {
    local lease_id="$1" worker_pid="$2" worker_start="${3:-}"
    local path state value launcher_pid launcher_start tmpdir gpu staged
    local paths=() gpus=()
    gpu_lease_mutex_acquire
    gpu_lease_reap_stale_locked
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        value="$(gpu_lease_field lease_id "$path")"
        [ "$value" = "$lease_id" ] || continue
        state="$(gpu_lease_field state "$path")"
        [ "$state" = launcher ] || { gpu_lease_mutex_release_if_owner; return 1; }
        paths+=("$path")
        gpus+=("$(gpu_lease_field gpu "$path")")
    done
    [ "${#paths[@]}" -gt 0 ] || { gpu_lease_mutex_release_if_owner; return 1; }
    launcher_pid="$(gpu_lease_field launcher_pid "${paths[0]}")"
    launcher_start="$(gpu_lease_field launcher_start "${paths[0]}")"
    tmpdir="$GPU_LEASE_ROOT/.handoff.$lease_id"
    mkdir "$tmpdir" 2>/dev/null || { gpu_lease_mutex_release_if_owner; return 1; }
    for path in "${paths[@]}"; do
        gpu="$(gpu_lease_field gpu "$path")"
        staged="$tmpdir/gpu.$gpu.lock"
        gpu_lease_write_metadata "$staged" "$gpu" worker "$worker_pid" "$worker_start" \
            "$launcher_pid" "$launcher_start" "$lease_id" \
            "$(gpu_lease_field job "$path")" "$(gpu_lease_field done "$path")" \
            "$(gpu_lease_field need "$path")" "$(gpu_lease_field devset "$path")" \
            "$(gpu_lease_field start "$path")" \
            "$(gpu_lease_field expected_seconds "$path")" \
            "$(gpu_lease_field cache_group "$path")" || {
            rm -f "$tmpdir"/*; rmdir "$tmpdir" 2>/dev/null || true
            gpu_lease_mutex_release_if_owner
            return 1
        }
    done
    # All replacements occur under the allocator mutex, so allocators cannot
    # observe a launcher/worker split across the members of a gang lease.
    local i
    for i in "${!gpus[@]}"; do
        gpu="${gpus[$i]}"
        path="${paths[$i]}"
        staged="$tmpdir/gpu.$gpu.lock"
        mv -f "$staged" "$path" || {
            rm -f "$tmpdir"/*; rmdir "$tmpdir" 2>/dev/null || true
            gpu_lease_mutex_release_if_owner
            return 1
        }
    done
    rmdir "$tmpdir" 2>/dev/null || true
    gpu_lease_mutex_release_if_owner
}

gpu_lease_release() {
    local lease_id="$1" worker_pid="$2" path value state owner
    gpu_lease_mutex_acquire
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        value="$(gpu_lease_field lease_id "$path")"
        state="$(gpu_lease_field state "$path")"
        owner="$(gpu_lease_field worker_pid "$path")"
        if [ "$value" = "$lease_id" ] && [ "$state" = worker ] &&
            [ "$owner" = "$worker_pid" ]; then
            rm -f "$path"
        fi
    done
    gpu_lease_mutex_release_if_owner
}

gpu_lease_release_launcher() {
    local launcher_pid="$1" launcher_start="${2:-}" path value state owner start host
    gpu_lease_mutex_acquire
    for path in "$GPU_LEASE_ROOT"/gpu.*.lock; do
        [ -e "$path" ] || continue
        host="$(gpu_lease_field hostname "$path")"
        [ "$host" = "$GPU_LEASE_HOST" ] || continue
        state="$(gpu_lease_field state "$path")"
        owner="$(gpu_lease_field launcher_pid "$path")"
        start="$(gpu_lease_field launcher_start "$path")"
        if [ "$state" = launcher ] && [ "$owner" = "$launcher_pid" ] &&
            { [ -z "$launcher_start" ] || [ "$start" = "$launcher_start" ]; }; then
            rm -f "$path"
        fi
    done
    gpu_lease_mutex_release_if_owner
}
