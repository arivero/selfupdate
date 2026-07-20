#!/usr/bin/env bash
# Readable pipeline-v4 PPP independent-stage demo.
#
# Unlike PPn wavefront execution, these workers do not pass activations from
# stage to stage. Each independently updates its owned block shard and
# atomically publishes a shard manifest for the later merge step.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="$SCRIPT_DIR/ppp_independent_stage_demo.py"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/tmp/${USER:-selfupdate}/selfupdate-venv/bin/python" ]]; then
  PYTHON_BIN="/tmp/${USER:-selfupdate}/selfupdate-venv/bin/python"
elif [[ -x /opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3 ]]; then
  PYTHON_BIN=/opt/ohpc/pub/apps/anaconda/anaconda-2025/bin/python3
else
  PYTHON_BIN=python3
fi
stages=3
blocks_per_stage=2
fail_stage=-1
work_dir=""
keep_work_dir=0

usage() {
  cat <<'EOF'
Usage: ppp_demo.sh [options]
  --stages N            Independent stage workers (default: 3)
  --blocks-per-stage N  Blocks owned by each stage (default: 2)
  --fail-stage N        Inject failure in stage N; -1 disables (default: -1)
  --work-dir DIR        Use DIR instead of a fresh /dev/shm directory
  --keep-work-dir       Retain an automatically created work directory
  --python PATH         Python interpreter (default: project venv when present)
  -h, --help            Show this help
EOF
}

while (($#)); do
  case "$1" in
    --stages) stages="$2"; shift 2 ;;
    --blocks-per-stage) blocks_per_stage="$2"; shift 2 ;;
    --fail-stage) fail_stage="$2"; shift 2 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    --keep-work-dir) keep_work_dir=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$stages" =~ ^[1-9][0-9]*$ ]] || { echo "--stages must be positive" >&2; exit 2; }
[[ "$blocks_per_stage" =~ ^[1-9][0-9]*$ ]] || {
  echo "--blocks-per-stage must be positive" >&2; exit 2;
}
[[ -f "$WORKER" ]] || { echo "missing worker: $WORKER" >&2; exit 2; }

created_work_dir=0
if [[ -z "$work_dir" ]]; then
  demo_tmp_root="/dev/shm/${USER:-selfupdate}"
  if [[ ! -d /dev/shm || ! -w /dev/shm ]]; then
    demo_tmp_root="${TMPDIR:-/tmp}"
  else
    mkdir -p "$demo_tmp_root"
  fi
  work_dir="$(mktemp -d "$demo_tmp_root/selfupdate-ppp-demo.XXXXXX")"
  created_work_dir=1
else
  mkdir -p "$work_dir"
  work_dir="$(cd "$work_dir" && pwd)"
fi

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if ((created_work_dir && !keep_work_dir)); then rm -rf -- "$work_dir"; fi
}
trap cleanup EXIT INT TERM

launch_id="ppp-demo-$(date +%s)-$$"
echo "launch: $launch_id"
echo "work directory: $work_dir"
echo "ownership: $stages independent stages x $blocks_per_stage blocks"

for ((stage = 0; stage < stages; stage++)); do
  "$PYTHON_BIN" "$WORKER" \
    --stage "$stage" --stages "$stages" \
    --work-dir "$work_dir" --launch-id "$launch_id" \
    --blocks-per-stage "$blocks_per_stage" --fail-stage "$fail_stage" &
  pids+=("$!")
  echo "started independent stage $stage as pid ${pids[-1]}"
done

failed=0
for ((stage = 0; stage < stages; stage++)); do
  if ! wait "${pids[$stage]}"; then
    echo "stage $stage failed; stopping sibling workers" >&2
    failed=1
    break
  fi
  echo "stage $stage shard complete"
done
((failed == 0)) || exit 1
pids=()

echo "independent stage shards (merge inputs):"
for shard in "$work_dir"/shards/stage_*.json; do
  [[ -e "$shard" ]] || { echo "no shards produced" >&2; exit 1; }
  printf '  %s: ' "$(basename "$shard")"
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])))' "$shard"
done

if ((created_work_dir && !keep_work_dir)); then
  echo "temporary work directory will be removed"
else
  echo "artifacts retained in: $work_dir"
fi
