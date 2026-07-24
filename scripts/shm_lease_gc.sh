#!/usr/bin/env bash
# Reclaim only this account's unleased selfupdate entries in node-local SHM.
set -euo pipefail

USER_NAME="$(id -un)"
SHM_USER_ROOT="/dev/shm/$USER_NAME"
LEASE_ROOT="$SHM_USER_ROOT/selfupdate-shm-leases"
JOB_ID="${SLURM_JOB_ID:-}"

usage() { echo "usage: $0 claim --path <path>... | gc | release" >&2; exit 2; }
require_job() { [[ "$JOB_ID" =~ ^[0-9]+(_[0-9]+)?$ ]] || { echo "REFUSED: SHM leases require SLURM_JOB_ID" >&2; exit 2; }; }
lease_file() { printf '%s/%s.lease\n' "$LEASE_ROOT" "$JOB_ID"; }

canonical_managed_path() {
  local raw="$1" canon parent leaf
  [[ "$raw" != *$'\n'* ]] || { echo "REFUSED: newline in SHM path" >&2; exit 2; }
  canon="$(realpath -m -- "$raw")"
  case "$canon" in
    "$SHM_USER_ROOT"/selfupdate-teacher-cache/*) parent="$SHM_USER_ROOT/selfupdate-teacher-cache" ;;
    "$SHM_USER_ROOT"/selfupdate-hf-cache/hub/models--*) parent="$SHM_USER_ROOT/selfupdate-hf-cache/hub" ;;
    *) echo "REFUSED: unmanaged SHM path '$raw'" >&2; exit 2 ;;
  esac
  leaf="${canon##*/}"
  [[ "$(dirname -- "$canon")" == "$parent" && "$leaf" =~ ^[A-Za-z0-9._-]+$ && "$raw" == "$canon" ]] || {
    echo "REFUSED: non-canonical SHM path '$raw'" >&2; exit 2;
  }
  [[ ! -L "$canon" ]] || { echo "REFUSED: SHM target is a symlink: $canon" >&2; exit 2; }
  printf '%s\n' "$canon"
}

lock_root() { mkdir -p "$LEASE_ROOT"; exec 9>"$LEASE_ROOT/.gc.lock"; flock 9; }

claim() {
  require_job
  local -a paths=()
  while [[ $# -gt 0 ]]; do
    [[ "$1" == "--path" && $# -ge 2 ]] || usage
    paths+=("$(canonical_managed_path "$2")")
    shift 2
  done
  [[ ${#paths[@]} -gt 0 ]] || usage
  lock_root
  local lease="$(lease_file)" tmp="$LEASE_ROOT/$JOB_ID.lease.$$"
  {
    printf 'job_id=%s\nhost=%s\nupdated=%s\n' "$JOB_ID" "$(hostname -s)" "$(date -Is)"
    [[ ! -f "$lease" ]] || sed -n 's/^path=/path=/p' "$lease"
    printf 'path=%s\n' "${paths[@]}"
  } | awk '!seen[$0]++' > "$tmp"
  mv -f "$tmp" "$lease"
  chmod 600 "$lease"
  echo "claimed SHM lease $JOB_ID on $(hostname -s)"
}

# return 0=active, 1=known absent, 2=Slurm unavailable (must preserve).
# This cluster returns an error for ``squeue -j <completed-id>`` rather than
# an empty result, so query the healthy active-job set once per check instead.
job_is_active() {
  local active_ids
  if ! active_ids="$(squeue -h -u "$USER_NAME" -w "$(hostname -s)" -o '%i' 2>/dev/null)"; then
    return 2
  fi
  grep -Fqx -- "$1" <<< "$active_ids"
}

gc() {
  require_job
  # Stagers hold this lock throughout GC, capacity check, and copy. A manual
  # GC acquires it itself, preventing it from racing a stager.
  if [[ "${SELFUPDATE_SHM_STAGE_LOCK_HELD:-0}" != 1 ]]; then
    mkdir -p "$SHM_USER_ROOT"; exec 8>"$SHM_USER_ROOT/.selfupdate-shm-stage.lock"; flock 8
  fi
  lock_root
  local lease id path rc candidate preserved
  local -a keep=()
  shopt -s nullglob
  for lease in "$LEASE_ROOT"/*.lease; do
    id="$(sed -n 's/^job_id=//p' "$lease" | head -n 1)"
    [[ "$id" =~ ^[0-9]+(_[0-9]+)?$ ]] || { echo "REFUSED: malformed SHM lease $lease" >&2; exit 4; }
    if job_is_active "$id"; then
      while IFS= read -r path; do keep+=("$path"); done < <(sed -n 's/^path=//p' "$lease")
    else
      rc=$?
      [[ "$rc" -ne 2 ]] || { echo "REFUSED: Slurm unavailable; preserving SHM leases" >&2; exit 4; }
      rm -f -- "$lease"
    fi
  done
  for candidate in "$SHM_USER_ROOT"/selfupdate-teacher-cache/* "$SHM_USER_ROOT"/selfupdate-hf-cache/hub/models--*; do
    [[ -e "$candidate" ]] || continue
    [[ ! -L "$candidate" ]] || { echo "REFUSED: managed SHM entry is a symlink: $candidate" >&2; exit 4; }
    preserved=0
    for path in "${keep[@]}"; do
      [[ "$candidate" == "$path" || "$candidate" == "$path"/* || "$path" == "$candidate"/* ]] && { preserved=1; break; }
    done
    if [[ "$preserved" -eq 0 ]]; then echo "pruning stale SHM entry $candidate"; rm -rf -- "$candidate"; fi
  done
}

release() { require_job; lock_root; rm -f -- "$(lease_file)"; echo "released SHM lease $JOB_ID on $(hostname -s)"; }
case "${1:-}" in
  claim) shift; claim "$@" ;;
  gc) [[ $# -eq 1 ]] || usage; gc ;;
  release) [[ $# -eq 1 ]] || usage; release ;;
  *) usage ;;
esac
