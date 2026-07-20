#!/usr/bin/env bash
# Request cooperative checkpoint-and-stop from the reduced-score campaign.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for host in agpul04 agpul05; do
  echo "requesting cooperative stop on $host"
  ssh -o BatchMode=yes "$host" bash -s <<'SH'
mapfile -t pids < <(
  ps -eo pid=,args= | awk '
    /python .*defactorised\/train.py/ &&
    /qwen35_4b_v32r_score100_/ &&
    !/awk/ {print $1}'
)
if ((${#pids[@]} == 0)); then
  echo "no matching trainers"
  exit 0
fi
printf 'SIGTERM trainer pids:'
printf ' %s' "${pids[@]}"
printf '\n'
kill -TERM "${pids[@]}"
SH
done

echo "stop requested; launchers will generate reports after atomic checkpoints"
