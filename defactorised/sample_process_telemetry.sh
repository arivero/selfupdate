#!/usr/bin/env bash
# Sample one exact trainer's host-side activity without touching Lustre trees.
# Cumulative counters are intentional: analysis can difference adjacent rows
# without asking ps/top to construct additional process-wide thread pools.
set -euo pipefail

PATTERN=""
OUT=""
INTERVAL="1"
WAIT_SECONDS="1800"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pattern) PATTERN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PATTERN" && -n "$OUT" ]] || {
  echo "usage: $0 --pattern EXPERIMENT_ARG --out FILE [--interval SEC]" >&2
  exit 2
}

mkdir -p "$(dirname "$OUT")"
printf '%s\n' \
  'timestamp,pid,state,utime_ticks,stime_ticks,minor_faults,major_faults,rss_kib,threads,read_bytes,write_bytes,voluntary_context_switches,nonvoluntary_context_switches,direct_children,compiler_children' \
  > "$OUT"

trainer_pid() {
  local cmdline arg has_entry has_pattern pid
  for cmdline in /proc/[0-9]*/cmdline; do
    [[ -r "$cmdline" ]] || continue
    has_entry=0
    has_pattern=0
    while IFS= read -r -d '' arg; do
      [[ "$arg" == */defactorised/train.py ]] && has_entry=1
      [[ "$arg" == *"$PATTERN"* ]] && has_pattern=1
    done < "$cmdline"
    if (( has_entry && has_pattern )); then
      pid="${cmdline#/proc/}"
      printf '%s\n' "${pid%/cmdline}"
      return 0
    fi
  done
  return 1
}

deadline=$((SECONDS + WAIT_SECONDS))
pid=""
while ! pid="$(trainer_pid)"; do
  (( SECONDS < deadline )) || {
    echo "trainer did not appear within ${WAIT_SECONDS}s: ${PATTERN}" >&2
    exit 1
  }
  sleep 1
done

page_kib=$(( $(getconf PAGESIZE) / 1024 ))
while [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]]; do
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$PATTERN"* && "$cmdline" == *"/defactorised/train.py"* ]] \
    || break

  # /proc/PID/stat field 2 is parenthesized and may contain spaces. Remove it;
  # the resulting zero-based array starts at documented field 3 (state).
  stat_tail="$(sed 's/^.*) //' "/proc/$pid/stat")"
  read -r -a fields <<< "$stat_tail"
  state="${fields[0]}"
  minor_faults="${fields[7]}"
  major_faults="${fields[9]}"
  utime_ticks="${fields[11]}"
  stime_ticks="${fields[12]}"
  threads="${fields[17]}"
  rss_kib=$(( fields[21] * page_kib ))

  read_bytes=0
  write_bytes=0
  while IFS=': ' read -r key value; do
    value="${value//[[:space:]]/}"
    case "$key" in
      read_bytes) read_bytes="$value" ;;
      write_bytes) write_bytes="$value" ;;
    esac
  done < "/proc/$pid/io"

  voluntary=0
  nonvoluntary=0
  while IFS=': ' read -r key value; do
    value="${value//[[:space:]]/}"
    case "$key" in
      voluntary_ctxt_switches) voluntary="$value" ;;
      nonvoluntary_ctxt_switches) nonvoluntary="$value" ;;
    esac
  done < "/proc/$pid/status"

  children="$(<"/proc/$pid/task/$pid/children")"
  child_count=0
  compiler_count=0
  for child in $children; do
    ((child_count += 1))
    if [[ -r "/proc/$child/cmdline" ]]; then
      child_cmd="$(tr '\0' ' ' < "/proc/$child/cmdline")"
      [[ "$child_cmd" == *torch/_inductor* ||
         "$child_cmd" == *compile_worker* ||
         "$child_cmd" == *triton* ]] && ((compiler_count += 1))
    fi
  done

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$pid" "$state" "$utime_ticks" \
    "$stime_ticks" "$minor_faults" "$major_faults" "$rss_kib" "$threads" \
    "$read_bytes" "$write_bytes" "$voluntary" "$nonvoluntary" \
    "$child_count" "$compiler_count" >> "$OUT"
  sleep "$INTERVAL"
done
