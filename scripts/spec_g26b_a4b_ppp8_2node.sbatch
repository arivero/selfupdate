#!/usr/bin/env bash
#SBATCH --job-name=g26b_a4b_ppp8_2node
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=runs/sbatch_g26b_a4b_ppp8_2node_%j.out
#SBATCH --error=runs/sbatch_g26b_a4b_ppp8_2node_%j.err
#
# 2-node, 8-GPU PPP8 variant of spec_g26b_a4b_campaign.yaml: same model
# (google/gemma-4-26B-A4B-it, which does NOT need 8 GPUs -- this run exists
# to exercise cross-node stage coordination: rendezvous/master-node
# selection, NCCL boundary transport, per-node HF-cache staging) rather than
# to test a model that actually needs the capacity. Each pipeline stage is a
# hard gate: a failure stops the job rather than silently proceeding to burn
# the reservation on invalid work. See scripts/spec_g26b_a4b_campaign.sbatch
# for the single-node PPP4 version this was derived from, and its inline
# comments for bugs already found and fixed there (--out directory
# semantics, build-vs-train config separation, generation budget matching,
# generation_responses_path cache-identity patching) -- all apply here too.
#
# Account/partition/node are deliberately NOT #SBATCH directives: sbatch
# parses #SBATCH lines textually at submission time, before the script
# runs, so they cannot reference shell variables. Submit with:
#
#   source .campaign_env.sh
#   sbatch --account="$SLURM_CAMPAIGN_ACCOUNT" \
#          --partition="$SLURM_CAMPAIGN_PARTITION" \
#          scripts/spec_g26b_a4b_ppp8_2node.sbatch
#
# Deliberately NOT passing --nodelist here: with only 2 healthy H100 nodes
# in rasmia_hopper, --nodes=2 already forces both; letting Slurm pick keeps
# this script portable to a cluster with more nodes in the partition.
#
# See docs/h100_bringup.md "sbatch campaign templates" and
# .campaign_env.sh.example for the full contract.

set -uo pipefail
# NOT ${BASH_SOURCE[0]}-relative -- Slurm copies the script into a per-job
# spool dir before running it (caught live 2026-07-23, job 422588).
: "${SLURM_SUBMIT_DIR:?run sbatch from the repository root}"
ROOT="$SLURM_SUBMIT_DIR"
cd "$ROOT"

if [[ -f "$ROOT/.campaign_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.campaign_env.sh"
fi
: "${SELFUPDATE_SSL_CERT_FILE:?set in .campaign_env.sh (copy from .campaign_env.sh.example)}"
: "${SELFUPDATE_VLLM_PYTHON:?set in .campaign_env.sh (copy from .campaign_env.sh.example)}"

export SSL_CERT_FILE="$SELFUPDATE_SSL_CERT_FILE"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TRITON_CACHE_DIR="/tmp/$USER/selfupdate-triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/selfupdate-torchinductor"

fail() { echo "FATAL: $*" >&2; exit 1; }

# --- discover the 2 allocated nodes from Slurm's own allocation, never
# hardcoded, so this template is portable to any 2 nodes the scheduler picks.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -eq 2 ]] || fail "expected 2 nodes, got ${#NODES[@]}: ${NODES[*]-none}"
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"
SELF_HOST="$(hostname -s)"
[[ "$SELF_HOST" == "$NODE0" ]] || fail "batch host $SELF_HOST is not NODE0 ($NODE0) -- launch_v4_stages.sh's MASTER_ADDR assumption (stage 0's host is the rendezvous leader) requires the launching shell to BE stage 0's host (2026-07-19 gotcha: launching an all-remote run from a host with nothing listening looked like a stall, was a clean TCPStore rendezvous failure on a dead address)."
echo "==== nodes: NODE0=$NODE0 (this host, stages 0-3) NODE1=$NODE1 (stages 4-7) $(date -Is) ===="

echo "==== stage: venv setup, both nodes $(date -Is) ===="
scripts/venv_setup.sh || fail "venv_setup on $NODE0"
scripts/venv_check.sh || fail "venv_check on $NODE0"
ssh -o BatchMode=yes "$NODE1" "cd '$ROOT' && scripts/venv_setup.sh && scripts/venv_check.sh" \
  || fail "venv_setup/venv_check on $NODE1"
PY="/tmp/$USER/selfupdate-venv/bin/python"
VLLM_PY="$SELFUPDATE_VLLM_PYTHON"
[[ -x "$VLLM_PY" ]] || fail "vLLM interpreter not found at $VLLM_PY (SELFUPDATE_VLLM_PYTHON)"

echo "==== stage: HF model-weight cache -> /dev/shm, both nodes $(date -Is) ===="
# /dev/shm is node-local -- each host must publish its own copy (AGENTS.md
# Multi-Node Conventions); the shared Lustre cache under caches/ (below) is
# NOT this and does not need per-node duplication.
scripts/stage_hf_cache.sh --shm google/gemma-4-26B-A4B-it || fail "stage_hf_cache on $NODE0"
ssh -o BatchMode=yes "$NODE1" "cd '$ROOT' && scripts/stage_hf_cache.sh --shm google/gemma-4-26B-A4B-it" \
  || fail "stage_hf_cache on $NODE1"

echo "==== stage: fresh vLLM answer generation (2071 items), $NODE0 only $(date -Is) ===="
# Single-node generation is enough: 26B-A4B fits one card's TP=4 (measured
# 2026-07-17 and again 2026-07-23); the 2-node aspect is for TRAINING stage
# coordination, not generation throughput.
GEN_DIR="runs/vllm_h100/gemma4_26b_a4b_it_ppp8_$(date +%Y%m%d)"
mkdir -p "$GEN_DIR"
GEN_OUT="$GEN_DIR/responses_bs256.jsonl"
"$VLLM_PY" scripts/benchmark_vllm_generation.py \
  --model google/gemma-4-26B-A4B-it \
  --config configs/base.yaml \
  --experiment configs/teacher_references/teacher_cache_gemma4_26b_a4b_v5rs.yaml \
  --examples data/combined/examples_v5rs_window.jsonl \
  --batch-sizes 256 \
  --tensor-parallel-size 4 \
  --max-model-len 6144 \
  --prompt-format native \
  --generation-extra-tokens 96 \
  --reasoning-effort low \
  --out "$GEN_DIR" || fail "vLLM generation"
[[ -s "$GEN_OUT" ]] || fail "vLLM generation produced no output at $GEN_OUT"

echo "==== stage: teacher-hidden cache build, $NODE0 only $(date -Is) ===="
# Shared Lustre cache.root (base.yaml default) -- both nodes read the same
# built cache directly; no per-node cache replication needed (that's only
# for the node_epoch0/--coordinated-node-cache path, which this run doesn't
# use -- see docs/h100_bringup.md and AGENTS.md's "knob copied without its
# enabling context" lesson before ever turning that on for real capacity
# reasons).
"$PY" scripts/build_teacher_cache.py \
  --config configs/base.yaml \
  --experiment configs/teacher_references/teacher_cache_gemma4_26b_a4b_v5rs.yaml \
  --generation-responses "$GEN_OUT" || fail "build_teacher_cache"

# See scripts/spec_g26b_a4b_campaign.sbatch for why this patching step
# exists: build_teacher_cache.py's --generation-responses makes the cache
# directory hash depend on the response file's content, but nothing on the
# training path sets that field from a yaml, so a plain training config
# misses the cache unless patched to match.
patch_generation_responses() {
  local src="$1" dst="$2"
  "$PY" - "$src" "$dst" "$GEN_OUT" <<'PYEOF'
import sys, yaml
src, dst, gen_out = sys.argv[1:4]
with open(src) as f:
    d = yaml.safe_load(f) or {}
d.setdefault("cache", {})["generation_responses_path"] = gen_out
with open(dst, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)
PYEOF
}
PATCHED_NUMGATE_SP="runs/.patched_gemma4_26b_a4b_ppp8_numgate_sp.yaml"
PATCHED_NUMGATE_SHARD="runs/.patched_gemma4_26b_a4b_ppp8_numgate_shard.yaml"
PATCHED_CAMPAIGN="runs/.patched_gemma4_26b_a4b_ppp8_2node_campaign.yaml"
patch_generation_responses configs/experiments/gemma4_26b_a4b_ppp8_2node_numgate_sp.yaml "$PATCHED_NUMGATE_SP"
patch_generation_responses configs/experiments/gemma4_26b_a4b_ppp8_2node_numgate_shard.yaml "$PATCHED_NUMGATE_SHARD"
patch_generation_responses configs/experiments/gemma4_26b_a4b_ppp8_2node_campaign.yaml "$PATCHED_CAMPAIGN"

# Keep the durable-cache identity, but copy its immutable contents to each
# node's local /dev/shm before any PPP8 stage starts.  The launcher forwards
# SELFUPDATE_TEACHER_CACHE_ROOT to remote stages explicitly.
CACHE_DIR="$("$PY" - configs/base.yaml "$PATCHED_NUMGATE_SP" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from selfupdate.config import load_config
from selfupdate.teacher.cache import resolve_cache_dir
cfg = load_config(sys.argv[1], sys.argv[2])
root, _ = resolve_cache_dir(cfg)
print(root.resolve())
PYEOF
)"
CACHE_SOURCE_ROOT="$(dirname "$CACHE_DIR")"
CACHE_NAME="$(basename "$CACHE_DIR")"
export SELFUPDATE_TEACHER_CACHE_ROOT="/dev/shm/$USER/selfupdate-teacher-cache"
echo "==== stage: teacher-hidden cache -> /dev/shm, both nodes $(date -Is) ===="
SELFUPDATE_TEACHER_CACHE_SOURCE="$CACHE_SOURCE_ROOT" \
SELFUPDATE_TEACHER_CACHE_DEST="$SELFUPDATE_TEACHER_CACHE_ROOT" \
  scripts/stage_teacher_cache_shm.sh "$CACHE_NAME" \
  || fail "stage_teacher_cache_shm on $NODE0"
ssh -o BatchMode=yes "$NODE1" \
  "cd '$ROOT' && SELFUPDATE_TEACHER_CACHE_SOURCE='$CACHE_SOURCE_ROOT' SELFUPDATE_TEACHER_CACHE_DEST='$SELFUPDATE_TEACHER_CACHE_ROOT' scripts/stage_teacher_cache_shm.sh '$CACHE_NAME'" \
  || fail "stage_teacher_cache_shm on $NODE1"

# Waiting on a cross-node stage set needs to check REMOTE pids over ssh, not
# a bare local kill -0 (which silently reports "dead" for anything not on
# this host -- fine for the single-node PPP4 script, wrong here since half
# the stages are genuinely elsewhere).
wait_for_stages() {
  local lease="$1" max_iters="$2"
  for _ in $(seq 1 "$max_iters"); do
    [[ -f "$lease" ]] || return 0
    local alive=0
    while IFS= read -r entry; do
      local host="${entry%%:*}" pid="${entry##*:}"
      if [[ "$entry" != *:* ]]; then host="$SELF_HOST"; pid="$entry"; fi
      if [[ "$host" == "$SELF_HOST" ]]; then
        kill -0 "$pid" 2>/dev/null && alive=1
      else
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "kill -0 $pid" 2>/dev/null && alive=1
      fi
    done < "$lease"
    [[ "$alive" -eq 0 ]] && return 0
    sleep 5
  done
  return 1
}

echo "==== GATE 1: numerics -- PPP8 (cross-node) then single-process, 1 epoch each $(date -Is) ===="
# Both sides go through launch_v4_stages.sh, even the "single-process" one
# (a 1-stage config, SELFUPDATE_V4_STAGE_HOSTS still unset here so it stays
# purely local to NODE0): a bare `train.py --v4-stage 0` invocation is
# missing env vars the launcher sets unconditionally
# (SELFUPDATE_V4_LAUNCH_ID, SELFUPDATE_V4_RELAY_ROOT) -- the distributed
# battery's _verify_entry does a real torch.distributed all_gather even for
# a single stage and strictly requires SELFUPDATE_V4_LAUNCH_ID, then fails
# on TCPStore rendezvous without the rest of the launcher's setup -- caught
# live 2026-07-23, job 422604.
export SELFUPDATE_V4_STAGE_HOSTS="$NODE0 $NODE0 $NODE0 $NODE0 $NODE1 $NODE1 $NODE1 $NODE1"
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_NUMGATE_SHARD" || fail "numgate PPP8 launch"
wait_for_stages "runs/.v4-launch-g26b_a4b_ppp8_numgate_shard.pids" 1440 \
  || fail "numgate PPP8 stages did not finish within the 120m gate budget"

unset SELFUPDATE_V4_STAGE_HOSTS
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_NUMGATE_SP" || fail "numgate single-process launch"
wait_for_stages "runs/.v4-launch-g26b_a4b_ppp8_numgate_sp.pids" 1440 \
  || fail "numgate single-process stage did not finish within the 120m gate budget"
"$PY" scripts/compare_v4_shard_numerics.py \
  runs/g26b_a4b_ppp8_numgate_sp/stage0 runs/g26b_a4b_ppp8_numgate_shard --strict-current \
  || fail "NUMERICS GATE FAILED: single-process and PPP8 disagree -- either the sharding/cross-node transport is broken, or (as with the PPP4 gate) gemma's shared-KV path itself is wrong. Stopping before burning the reservation."
echo "==== GATE 1 PASSED $(date -Is) ===="

echo "==== GATE 2 (manual read, not auto-abort): epoch 1-3 learning signal ===="
echo "Check runs/g26b_a4b_ppp8_2node_campaign_stage*.log for teacher-argmax"
echo "acceptance / recall moving after the campaign launch below."

echo "==== stage: full 500-epoch PPP8 2-node campaign $(date -Is) ===="
export SELFUPDATE_V4_STAGE_HOSTS="$NODE0 $NODE0 $NODE0 $NODE0 $NODE1 $NODE1 $NODE1 $NODE1"
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_CAMPAIGN" || fail "campaign launch"
echo "launched; stage logs: runs/g26b_a4b_ppp8_2node_campaign_stage*.log"
echo "merge when done: $PY scripts/merge_v4_adapters.py runs/g26b_a4b_ppp8_2node_campaign"

wait_for_stages "runs/.v4-launch-g26b_a4b_ppp8_2node_campaign.pids" 100000 \
  && echo "==== campaign stages finished $(date -Is) ====" \
  || echo "==== wait_for_stages ceiling hit (24h wall time will end the job regardless) $(date -Is) ===="
