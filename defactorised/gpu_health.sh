#!/usr/bin/env bash
# AUTHORITATIVE GPU health (owner, 2026-07-18: "always check with
# nvidia-smi, it doesn't coincide with your usual analysis"). Ground truth
# is nvidia-smi compute-apps (what actually holds each GPU), NOT ps/grep.
# For each host: GPU index -> pid -> mem -> util -> which run, and a
# stray-context flag (a pid on more than one GPU). Usage:
#   defactorised/gpu_health.sh            # this host
#   defactorised/gpu_health.sh agpuh02    # a remote host (over ssh)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

report_local() {
  echo "### $(hostname -s)"
  # index -> uuid map
  declare -A IDX
  while IFS=',' read -r idx uuid; do IDX["$(echo "$uuid" | xargs)"]="$(echo "$idx" | xargs)"; done \
    < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
  # util per index
  declare -A UTIL
  while IFS=',' read -r idx u; do UTIL["$(echo "$idx" | xargs)"]="$(echo "$u" | xargs)"; done \
    < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader)
  # compute-apps: uuid,pid,mem
  local seen_pids=""
  local any=0
  while IFS=',' read -r uuid pid mem; do
    [ -z "$pid" ] && continue
    any=1
    uuid="$(echo "$uuid" | xargs)"; pid="$(echo "$pid" | xargs)"; mem="$(echo "$mem" | xargs)"
    local idx="${IDX[$uuid]:-?}"
    local run; run="$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | grep -oE 'h100_[a-z0-9_]+|ppp[0-9x_]+|v4_battery|dequant|benchmark_vllm' | head -1)"
    printf '  GPU%s  pid=%-8s mem=%-10s util=%s%%  %s\n' "$idx" "$pid" "$mem" "${UTIL[$idx]:-?}" "${run:-?}"
    case " $seen_pids " in *" $pid "*) echo "  !! STRAY: pid $pid on >1 GPU" ;; *) seen_pids="$seen_pids $pid" ;; esac
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
  [ "$any" = 0 ] && echo "  (all GPUs idle)"
}

if [ $# -ge 1 ] && [ "$1" != "$(hostname -s)" ] && [ "$1" != local ]; then
  ssh "$1" "cd '$ROOT' && bash defactorised/gpu_health.sh"
else
  report_local
fi
