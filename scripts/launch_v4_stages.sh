#!/usr/bin/env bash
# Launch every pipeline-v4 layer-shard stage of one experiment, one detached
# process per stage. The stage count comes from the CONFIG
# (train.v4_stage_splits: N cuts -> N+1 stages), so this works for any
# number of GPUs — nothing here assumes four.
#
# Usage:
#   scripts/launch_v4_stages.sh <base.yaml> <experiment.yaml>
#
# Each stage k:
#   - runs `train.py --v4-stage k` (train.py pins model.device to
#     v4_stage_devices[k] — physical id, never renumbered; CUDA_VISIBLE_DEVICES
#     is deliberately left alone)
#   - writes runs/<run_name>/stage<k>/ (metrics, checkpoint shard)
#   - logs to runs/<run_name>_stage<k>.log
# Merge shards afterwards with scripts/merge_v4_adapters.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:?usage: launch_v4_stages.sh <base.yaml> <experiment.yaml>}"
EXP="${2:?usage: launch_v4_stages.sh <base.yaml> <experiment.yaml>}"
PY="${SELFUPDATE_VENV:-/tmp/$USER/selfupdate-venv}/bin/python"

read -r STAGES RUN_NAME <<EOF2
$("$PY" - "$BASE" "$EXP" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from selfupdate.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2])
if cfg.train.pipeline_version != 4:
    raise SystemExit("launch_v4_stages.sh requires pipeline_version=4")
print(len(cfg.train.v4_stage_splits or []) + 1, cfg.run_name)
PYEOF
)
EOF2

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
# Unbuffered stdout: fully-buffered stage logs read as 0 bytes for the
# whole run and misled crash triage three times on 2026-07-18.
export PYTHONUNBUFFERED=1
export SELFUPDATE_CPU_THREADS="${SELFUPDATE_CPU_THREADS:-8}"
# /home is a small shared NFS mount (seen 100% full 2026-07-18) — every
# compiler cache must live node-local, never under $HOME.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/selfupdate-triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/selfupdate-torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/$USER/selfupdate-vllm-cache}"
# Offline discipline (same as l40s_exec.sh / staged vLLM launches): snapshots
# and eval data are local — the RAM stage or the account cache for weights,
# vendored data/eval/*.json for standard damage. Without this every stage
# pings the Hub unauthenticated at startup; a genuinely cold cache must fail
# loudly instead. Override with HF_HUB_OFFLINE=0 for a first-time download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# Prefer a completed RAM stage of the model snapshots (the container-era
# convention): N stage processes cold-loading a 54 GB model from Lustre sit
# in D-state page faults; from /dev/shm they load in seconds. Stage once
# per node with: scripts/stage_hf_cache.sh --shm <org/model>
SHM_HF="/dev/shm/$USER/selfupdate-hf-cache"
if [[ -f "$SHM_HF/.selfupdate-hf-stage-ready" ]]; then
  export HF_HOME="$SHM_HF"
  echo "model snapshots: RAM stage $SHM_HF"
else
  echo "model snapshots: account cache (Lustre) — for big models stage first:"
  echo "  scripts/stage_hf_cache.sh --shm <org/model>"
fi

mkdir -p "$ROOT/runs"
# Launch lease: a second invocation for the same run must refuse while the
# first one's stages are alive (the 2026-07-17 double-launch near-miss:
# two watchers raced; both sets would have written the same run directory
# and OOMed every card). Stale leases from dead pids are reclaimed.
LEASE="$ROOT/runs/.v4-launch-$(echo "$RUN_NAME" | tr '/' '_').pids"
if [[ -f "$LEASE" ]]; then
  while read -r entry; do
    ohost="${entry%%:*}"; opid="${entry##*:}"
    if [[ "$entry" != *:* ]]; then ohost="$(hostname -s)"; opid="$entry"; fi
    if [[ "$ohost" == "$(hostname -s)" || "$ohost" == "local" ]]; then
      if kill -0 "$opid" 2>/dev/null; then
        echo "REFUSED: $RUN_NAME already launched (live pid $opid, lease $LEASE)" >&2
        exit 3
      fi
    else
      # A pid on another node cannot be probed from this namespace
      # (Multi-Node Conventions: never auto-reap remote leases). Verify
      # on that host and remove the lease by hand.
      echo "REFUSED: lease holds remote stage $entry; verify on $ohost" \
           "that scheduler and worker are dead, then rm $LEASE" >&2
      exit 3
    fi
  done < "$LEASE"
  rm -f "$LEASE" "$LEASE.local"
fi
# Cross-node stage map (plan B7, the InfiniBand jump): one host per stage,
# space-separated; empty/short entries and "local" mean this host. Example
# PPP8 over two nodes: SELFUPDATE_V4_STAGE_HOSTS="local local local local
# agpuh02 agpuh02 agpuh02 agpuh02". Remote stages launch over ssh with the
# same launch id; the postal envelopes already carry from_host, so
# mis-routed cross-host mail is refused by construction.
read -r -a STAGE_HOSTS <<< "${SELFUPDATE_V4_STAGE_HOSTS:-}"
SELF_HOST="$(hostname -s)"
MULTI_HOST=0
# NOTE (2026-07-20 fix): do NOT write this as "${STAGE_HOSTS[@]:-}" — on an
# array with ZERO elements (the normal single-node case, SELFUPDATE_V4_STAGE_HOSTS
# unset), bash's ${arr[@]:-default} gotcha substitutes ONE default empty-string
# element instead of iterating zero times. That silently turned a single-node
# launch into a length-1 "host map" downstream (exported as one hostname),
# which then failed validate.py's host/device equal-length check against the
# 4-entry v4_stage_devices list — every single-node multi-stage launch was
# broken from commit 60a268c (2026-07-19) until this fix. Bash 4.4+ (confirmed
# on this cluster, 4.4.20) iterates zero times over "${arr[@]}" even under
# `set -u` when the array is genuinely empty, so the plain form below is safe.
for h in "${STAGE_HOSTS[@]}"; do
  if [[ -n "$h" && "$h" != "local" && "$h" != "$SELF_HOST" ]]; then
    MULTI_HOST=1
  fi
done
# Resolve empty/"local" -> this launcher's short hostname so EVERY node reads
# the SAME stage->host map and routes boundary mail by CO-LOCATION: a
# same-node stage pair exchanges through /dev/shm (RAM), a cross-node pair
# over NCCL/IB. No comms path ever touches a disk filesystem (ssd/lustre/nfs)
# — owner 2026-07-19. The trainer's resolve_relay_transport reads this map.
STAGE_HOSTS_RESOLVED=()
for h in "${STAGE_HOSTS[@]}"; do
  if [[ -z "$h" || "$h" == "local" ]]; then
    STAGE_HOSTS_RESOLVED+=("$SELF_HOST")
  else
    STAGE_HOSTS_RESOLVED+=("$h")
  fi
done
# A genuinely empty STAGE_HOSTS (the normal single-node case) must resolve to
# an EMPTY SELFUPDATE_V4_STAGE_HOSTS, not a single-entry guess — validate.py
# (line ~140) deliberately skips its host/device length check only when this
# var is empty ("static config audit has no host assignment"). Do not
# backfill STAGE_HOSTS_RESOLVED from v4_stage_devices count here.
export SELFUPDATE_V4_STAGE_HOSTS="${STAGE_HOSTS_RESOLVED[*]:-}"

# One identity per coordinated launch: every relay/adapter file is stamped
# with it and stages refuse tensors from any other launch.
export SELFUPDATE_V4_LAUNCH_ID="v4-$(date +%Y%m%d%H%M%S)-$$"
if [[ "$MULTI_HOST" = "1" ]]; then
  # Cross-node mail goes NATIVE InfiniBand (owner, 2026-07-18): the
  # trainer resolves v4_relay_transport=auto -> nccl from this flag.
  # Rendezvous over the Ethernet management net (IPoIB node-to-node is
  # broken here); NCCL data path picks the HDR-200 HCAs itself.
  export SELFUPDATE_V4_CROSS_NODE=1
  # Stage 0's host is the rendezvous leader, NOT the launching shell's own
  # host — those differ whenever the orchestrator issues the launch from a
  # host that isn't itself one of the stages (measured 2026-07-19: launching
  # an all-agpuh02 run from an agpuh01 shell defaulted MASTER_ADDR=agpuh01,
  # a host with nothing listening; every stage ran ~9-10 min of real work
  # before the store-fill relay's TCPStore rendezvous failed cleanly —
  # looked like a stall from the outside, was actually a clean crash on a
  # dead address).
  export MASTER_ADDR="${MASTER_ADDR:-${STAGE_HOSTS_RESOLVED[0]}}"
  export MASTER_PORT="${MASTER_PORT:-29517}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno12419np2}"
  # gloo (the NcclBoundaryRelay launch-id all_gather subgroup) needs its OWN
  # interface hint or it auto-picks a non-routable NIC and spins forever in a
  # connect-retry sleep AFTER nccl init already succeeded (measured hang,
  # 2026-07-19). Pin it to the same management-net interface as NCCL's bootstrap.
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
fi
# The file relay is ALWAYS node-local /dev/shm (RAM tmpfs) — NEVER a disk
# filesystem (owner 2026-07-19: "standard file system ... decays to ssd/
# lustre/nfs, all bad options ... completely out of the code"). Cross-node
# mail does not travel through a shared mount at all: same-node stage pairs
# use this shm exchange, cross-node pairs go NCCL/IB (relay_nccl.py). Each
# node materializes its OWN shm root under the identical path.
export SELFUPDATE_V4_RELAY_ROOT="${SELFUPDATE_V4_RELAY_ROOT:-/dev/shm/$USER/selfupdate-v4-relay}"
case "$SELFUPDATE_V4_RELAY_ROOT" in
  /dev/shm/*) : ;;
  *) echo "REFUSED: relay root '$SELFUPDATE_V4_RELAY_ROOT' is not under /dev/shm; comms must be RAM-backed (owner 2026-07-19)" >&2; exit 3 ;;
esac
mkdir -p "$SELFUPDATE_V4_RELAY_ROOT"
# Wipe this run's relay exchange: we hold the lease, so no live stage of this
# run exists, and any files there are dead mail from a previous launch. The
# envelope check would refuse them fatally (2026-07-17 19:00 incident: a
# leftover e0001/stage1.st from a 17:31 launch killed stage 2 of the 17:53
# relaunch); prevention beats detection.
rm -rf "${SELFUPDATE_V4_RELAY_ROOT:?}/$RUN_NAME"
echo "launching $STAGES v4 stages of $RUN_NAME  (launch id $SELFUPDATE_V4_LAUNCH_ID)"
STAGE_ENV="PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF \
TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
SELFUPDATE_CPU_THREADS=$SELFUPDATE_CPU_THREADS \
HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
SELFUPDATE_V4_LAUNCH_ID=$SELFUPDATE_V4_LAUNCH_ID \
SELFUPDATE_V4_RELAY_ROOT=$SELFUPDATE_V4_RELAY_ROOT \
SELFUPDATE_V4_CROSS_NODE=${SELFUPDATE_V4_CROSS_NODE:-0} \
MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-} \
SELFUPDATE_V4_READY_PORT=${SELFUPDATE_V4_READY_PORT:-} \
NCCL_IB_HCA=${NCCL_IB_HCA:-} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-} \
NCCL_DEBUG=${NCCL_DEBUG:-} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-} \
SELFUPDATE_V4_STAGE_HOSTS='${SELFUPDATE_V4_STAGE_HOSTS:-}'"
pids=()
local_pids=()
for ((k = 0; k < STAGES; k++)); do
  logfile="$ROOT/runs/${RUN_NAME}_stage${k}.log"
  # Logs append across launches; without a separator, stale tracebacks
  # from a previous attempt masquerade as fresh failures (three
  # mis-triages on 2026-07-18).
  echo "==== launch $SELFUPDATE_V4_LAUNCH_ID stage $k $(date -Is) ====" >> "$logfile"
  host="${STAGE_HOSTS[$k]:-local}"
  if [[ -z "$host" || "$host" == "local" || "$host" == "$(hostname -s)" ]]; then
    nohup setsid "$PY" "$ROOT/scripts/train.py" \
      --config "$BASE" --experiment "$EXP" --v4-stage "$k" \
      >> "$logfile" 2>&1 &
    pids+=("$(hostname -s):$!")
    local_pids+=("$! $k")
    echo "  stage $k -> local pid $!  log $logfile"
  else
    # Remote stage: same tree over the shared filesystem, that node's own
    # /tmp venv (build it there first: scripts/venv_setup.sh). HF_HOME is
    # NOT forwarded — each node resolves its own stage or account cache.
    # NEVER capture the remote pid via command substitution: sshd holds
    # the session open for a healthy backgrounded stage (even with stdin
    # from /dev/null) and $() blocks forever — the PPP5 spawn loop froze
    # twice on 2026-07-18. Instead the remote writes its pid to a
    # shared-FS pidfile; we background the ssh client, poll the file, and
    # kill the client (the setsid'd stage survives it).
    pidfile="$ROOT/runs/.rpid_${RUN_NAME}_stage${k}"
    rm -f "$pidfile"
    ssh -o BatchMode=yes "$host" \
      "cd '$ROOT' && nohup setsid env $STAGE_ENV \
       /tmp/\$USER/selfupdate-venv/bin/python scripts/train.py \
       --config '$BASE' --experiment '$EXP' --v4-stage $k \
       < /dev/null >> '$logfile' 2>&1 & echo \$! > '$pidfile'" \
      < /dev/null > /dev/null 2>&1 &
    ssh_client=$!
    for _ in $(seq 1 45); do
      [ -s "$pidfile" ] && break
      sleep 1
    done
    kill "$ssh_client" 2>/dev/null
    rpid=$(cat "$pidfile" 2>/dev/null || echo unknown)
    if [[ "$rpid" == "unknown" ]]; then
      echo "  stage $k -> $host SPAWN FAILED (no pidfile within 45 s)" >&2
    fi
    pids+=("$host:$rpid")
    echo "  stage $k -> $host pid $rpid  log $logfile"
  fi
  # Small stagger so concurrent model loads do not thrash the snapshot cache.
  sleep 5
done
printf '%s\n' "${pids[@]}" > "$LEASE"
# Reaper: if one stage dies of a non-clean exit, terminate the siblings — a
# set with a dead stage cannot publish; do not burn the other cards. The
# reaper supervises LOCAL pids only (a pid on another node cannot be probed
# from this pid namespace — Multi-Node Conventions); remote stages rely on
# the relay timeout as their cross-host backstop, and each remote host can
# run its own reaper over its lease-file subset.
if [[ ${#local_pids[@]} -gt 0 ]]; then
  printf '%s\n' "${local_pids[@]}" > "$LEASE.local"
  nohup setsid "$ROOT/scripts/v4_stage_reaper.sh" "$LEASE.local" \
    "$ROOT/runs/${RUN_NAME}_stage" >> "$ROOT/runs/${RUN_NAME}_reaper.log" 2>&1 &
  echo "local reaper pid $! (watching ${#local_pids[@]} local stages)"
fi
echo "stage pids: ${pids[*]}"
echo "watch:  tail -f runs/${RUN_NAME}_stage*.log"
echo "merge:  $PY scripts/merge_v4_adapters.py runs/$RUN_NAME"
