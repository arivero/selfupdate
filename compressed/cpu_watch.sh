#!/usr/bin/env bash
# Lightweight CPU-oversubscription logger. Samples every SAMPLE_S seconds and
# appends ONE line to the log ONLY when the host is actually oversubscribed
# (1-min load over LOAD_HI, default = #cores/2) or aggregate python CPU is
# high — so the log stays empty on quiet nights and captures the spikes we
# currently have no record of (2026-07-12: intermittent ~2400%/proc bursts
# observed overnight, no logs). Records per-python-process %CPU + thread
# counts + owner so a spike can be attributed (the box is a SHARED 64-core
# login node — a spike may be another user, not our jobs).
#
# Usage (background):
#   nohup setsid bash compressed/cpu_watch.sh >> runs/cpu_watch_$(hostname -s).log 2>&1 &
set -u
CORES=$(nproc)
LOAD_HI="${LOAD_HI:-$(( CORES / 2 ))}"      # log when load1 exceeds this
CPU_HI="${CPU_HI:-1500}"                      # or when any one process >1500%
SAMPLE_S="${SAMPLE_S:-20}"

echo "[$(date '+%F %T')] cpu_watch start: cores=$CORES load_hi=$LOAD_HI cpu_hi=$CPU_HI sample=${SAMPLE_S}s"
while true; do
  load1=$(cut -d' ' -f1 /proc/loadavg | tr ',' '.')
  # peak single-process %CPU among python procs (instantaneous)
  peak=$(top -b -n1 2>/dev/null | awk '/python/ {gsub(",",".",$9); if($9>m)m=$9} END{printf "%.0f", m+0}')
  hot=$(awk -v a="$load1" -v b="$LOAD_HI" -v p="$peak" -v c="$CPU_HI" \
        'BEGIN{print (a>b || p>c) ? 1 : 0}')
  if [ "$hot" = "1" ]; then
    echo "[$(date '+%F %T')] SPIKE load1=$load1 peak_proc=${peak}%CPU cores=$CORES"
    # attribute: top python-ish procs by CPU with user + thread count
    ps -eo user,pid,%cpu,nlwp,comm --sort=-%cpu 2>/dev/null \
      | awk 'NR==1 || /python|pt_main|train|teacher/' | head -8 \
      | sed 's/^/    /'
  fi
  sleep "$SAMPLE_S"
done
