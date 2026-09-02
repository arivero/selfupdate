#!/usr/bin/env bash
# Node-local scratch lifecycle for exclusive H100 batch jobs.
# Source this file, then call selfupdate_job_tmp_init with a namespace such as
# selfupdate-v6.  /tmp is preferred, with /dev/shm as the fail-closed fallback
# when the shared node disk cannot provide useful space.  Only known generated
# names below those user's roots are removed; the reusable /tmp venv is never
# job scratch.

selfupdate_job_tmp_remove() {
  local path="$(realpath -m -- "$1")"
  case "$path" in
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-*-job-*|\
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-v6-triton|\
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-v6-torchinductor|\
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-v6-pycache-*|\
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-triton|\
    "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-torchinductor|\
    "$SELFUPDATE_TMP_USER_ROOT"/uv-cache|\
    "$SELFUPDATE_SHM_USER_ROOT"/selfupdate-*-job-*)
      rm -rf -- "$path"
      ;;
    *)
      echo "FATAL: refusing unsafe job scratch removal: $path" >&2
      return 1
      ;;
  esac
}

selfupdate_job_tmp_cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${SELFUPDATE_JOB_TMP_ROOT:-}" ]]; then
    selfupdate_job_tmp_remove "$SELFUPDATE_JOB_TMP_ROOT" || true
  fi
  exit "$status"
}

selfupdate_job_tmp_init() {
  local namespace="${1:?job scratch namespace required}"
  [[ "$namespace" =~ ^selfupdate-(v5|v6)$ ]] \
    || { echo "FATAL: unsupported job scratch namespace: $namespace" >&2; return 1; }
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] \
    || { echo "FATAL: numeric SLURM_JOB_ID required" >&2; return 1; }

  SELFUPDATE_TMP_USER_ROOT="$(realpath -m -- "/tmp/$USER")"
  SELFUPDATE_SHM_USER_ROOT="$(realpath -m -- "/dev/shm/$USER")"
  [[ "$SELFUPDATE_TMP_USER_ROOT" == /tmp/* ]] \
    || { echo "FATAL: unsafe user tmp root: $SELFUPDATE_TMP_USER_ROOT" >&2; return 1; }
  [[ "$SELFUPDATE_SHM_USER_ROOT" == /dev/shm/* ]] \
    || { echo "FATAL: unsafe user shm root: $SELFUPDATE_SHM_USER_ROOT" >&2; return 1; }
  mkdir -p -- "$SELFUPDATE_TMP_USER_ROOT"

  # Reclaim our abandoned v5/v6 directories before trying to allocate a new
  # inode.  This ordering matters when another account has filled /tmp so
  # completely that mkdir itself fails. Preserve every job Slurm still sees.
  local root stale stale_id state
  for root in "$SELFUPDATE_TMP_USER_ROOT" "$SELFUPDATE_SHM_USER_ROOT"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for stale in "$root"/selfupdate-v5-job-* "$root"/selfupdate-v6-job-*; do
      stale_id="${stale##*-job-}"
      if [[ "$stale_id" != "$SLURM_JOB_ID" ]]; then
        state="$(squeue -h -j "$stale_id" -o '%T' 2>/dev/null || true)"
        [[ -n "$state" ]] && continue
      fi
      selfupdate_job_tmp_remove "$stale"
    done
    shopt -u nullglob
  done

  # Remove cache names used by the pre-lifecycle wrappers.  These jobs reserve
  # all four H100s, so no same-user training worker can still be using them on
  # this node when initialization runs.
  if [[ "$namespace" == selfupdate-v6 ]]; then
    shopt -s nullglob
    for stale in \
      "$SELFUPDATE_TMP_USER_ROOT/selfupdate-v6-triton" \
      "$SELFUPDATE_TMP_USER_ROOT/selfupdate-v6-torchinductor" \
      "$SELFUPDATE_TMP_USER_ROOT"/selfupdate-v6-pycache-*; do
      [[ ! -e "$stale" ]] || selfupdate_job_tmp_remove "$stale"
    done
    shopt -u nullglob
  else
    for stale in \
      "$SELFUPDATE_TMP_USER_ROOT/selfupdate-triton" \
      "$SELFUPDATE_TMP_USER_ROOT/selfupdate-torchinductor"; do
      [[ ! -e "$stale" ]] || selfupdate_job_tmp_remove "$stale"
    done
  fi

  # The old shared UV cache is disposable and was the 7 GiB residue behind
  # the 2026-08-31 launch failures.  Serialize its removal with venv setup.
  (
    flock -x 9
    [[ ! -e "$SELFUPDATE_TMP_USER_ROOT/uv-cache" ]] \
      || selfupdate_job_tmp_remove "$SELFUPDATE_TMP_USER_ROOT/uv-cache"
  ) 9>"$SELFUPDATE_TMP_USER_ROOT/.selfupdate-venv-setup.lock"

  local minimum_kib="${SELFUPDATE_JOB_TMP_MIN_KIB:-8388608}"
  local available_kib
  available_kib="$(df -Pk "$SELFUPDATE_TMP_USER_ROOT" | awk 'NR == 2 {print $4}')"
  if [[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge "$minimum_kib" ]]; then
    SELFUPDATE_JOB_TMP_ROOT="$SELFUPDATE_TMP_USER_ROOT/$namespace-job-$SLURM_JOB_ID"
    selfupdate_job_tmp_remove "$SELFUPDATE_JOB_TMP_ROOT"
    mkdir -p -- "$SELFUPDATE_JOB_TMP_ROOT/tmp"
  else
    echo "job scratch: /tmp has ${available_kib:-unknown} KiB free; using /dev/shm" >&2
    mkdir -p -- "$SELFUPDATE_SHM_USER_ROOT"
    SELFUPDATE_JOB_TMP_ROOT="$SELFUPDATE_SHM_USER_ROOT/$namespace-job-$SLURM_JOB_ID"
    selfupdate_job_tmp_remove "$SELFUPDATE_JOB_TMP_ROOT"
    mkdir -p -- "$SELFUPDATE_JOB_TMP_ROOT/tmp"
  fi
  export SELFUPDATE_JOB_TMP_ROOT
  export TMPDIR="$SELFUPDATE_JOB_TMP_ROOT/tmp"
  export TMP="$TMPDIR" TEMP="$TMPDIR"
  export UV_CACHE_DIR="$SELFUPDATE_JOB_TMP_ROOT/uv-cache"
  export TRITON_CACHE_DIR="$SELFUPDATE_JOB_TMP_ROOT/triton"
  export TORCHINDUCTOR_CACHE_DIR="$SELFUPDATE_JOB_TMP_ROOT/torchinductor"
  export PYTHONPYCACHEPREFIX="$SELFUPDATE_JOB_TMP_ROOT/pycache"

  trap selfupdate_job_tmp_cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

selfupdate_job_tmp_prune_install_cache() {
  [[ -n "${UV_CACHE_DIR:-}" ]] || return 0
  selfupdate_job_tmp_remove "$UV_CACHE_DIR"
}
