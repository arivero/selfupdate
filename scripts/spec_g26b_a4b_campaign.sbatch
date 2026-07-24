#!/usr/bin/env bash
#SBATCH --job-name=g26b_a4b_ppp4_campaign
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=runs/sbatch_g26b_a4b_campaign_%j.out
#SBATCH --error=runs/sbatch_g26b_a4b_campaign_%j.err
#
# Full v4.6 test campaign for google/gemma-4-26B-A4B-it: fresh vLLM
# generation -> teacher cache build -> numerics gate -> PPP4 campaign.
# Each stage is a hard gate: a failure stops the job rather than silently
# proceeding to burn the reservation on invalid work.
#
# Account/partition/node are deliberately NOT #SBATCH directives here:
# sbatch parses #SBATCH lines textually at submission time, before the
# script runs, so they cannot reference shell variables. Submit with:
#
#   source .campaign_env.sh
#   sbatch --account="$SLURM_CAMPAIGN_ACCOUNT" \
#          --partition="$SLURM_CAMPAIGN_PARTITION" \
#          ${SLURM_CAMPAIGN_NODELIST:+--nodelist="$SLURM_CAMPAIGN_NODELIST"} \
#          scripts/spec_g26b_a4b_campaign.sbatch
#
# See docs/h100_bringup.md "sbatch campaign templates" and
# .campaign_env.sh.example for the full contract.

set -uo pipefail
# NOT ${BASH_SOURCE[0]}-relative: Slurm copies the sbatch script into a
# per-job spool dir (/var/spool/slurmd/jobNNN/slurm_script) before running
# it, so that trick resolves to the spool dir, not the repo (caught live
# 2026-07-23, job 422588, instant "scripts/venv_setup.sh: No such file or
# directory"). SLURM_SUBMIT_DIR is the directory sbatch was invoked FROM --
# always submit from the repo root.
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

PPP4_ARM="${PPP4_ARM:-unscoped_b16}"
case "$PPP4_ARM" in
  unscoped_b16)
    NUMGATE_SP_CFG=configs/experiments/gemma4_26b_a4b_ppp4_numgate.yaml
    NUMGATE_SHARD_CFG=configs/experiments/gemma4_26b_a4b_ppp4_numgate_shard.yaml
    CAMPAIGN_CFG=configs/experiments/gemma4_26b_a4b_ppp4_campaign.yaml
    NUMGATE_SP_RUN=g26b_a4b_ppp4_unscoped_b16_numgate_sp
    NUMGATE_SHARD_RUN=g26b_a4b_ppp4_unscoped_b16_numgate_shard
    CAMPAIGN_RUN=g26b_a4b_ppp4_unscoped_b16_campaign ;;
  scoped_b32)
    NUMGATE_SP_CFG=configs/experiments/gemma4_26b_a4b_ppp4_scoped_b16_numgate_sp.yaml
    NUMGATE_SHARD_CFG=configs/experiments/gemma4_26b_a4b_ppp4_scoped_b16_numgate_shard.yaml
    CAMPAIGN_CFG=configs/experiments/gemma4_26b_a4b_ppp4_scoped_b32_campaign.yaml
    NUMGATE_SP_RUN=g26b_a4b_ppp4_scoped_b16_numgate_sp
    NUMGATE_SHARD_RUN=g26b_a4b_ppp4_scoped_b16_numgate_shard
    CAMPAIGN_RUN=g26b_a4b_ppp4_scoped_b32_campaign ;;
  *) fail "unknown PPP4_ARM=$PPP4_ARM" ;;
esac

echo "==== stage: venv setup $(date -Is) ===="
scripts/venv_setup.sh || fail "venv_setup"
scripts/venv_check.sh || fail "venv_check"
PY="/tmp/$USER/selfupdate-venv/bin/python"
VLLM_PY="$SELFUPDATE_VLLM_PYTHON"
[[ -x "$VLLM_PY" ]] || fail "vLLM interpreter not found at $VLLM_PY (SELFUPDATE_VLLM_PYTHON)"

echo "==== stage: HF model-weight cache -> /dev/shm $(date -Is) ===="
scripts/stage_hf_cache.sh --shm google/gemma-4-26B-A4B-it || fail "stage_hf_cache"

echo "==== stage: fresh vLLM answer generation (2071 items) $(date -Is) ===="
GEN_DIR="runs/vllm_h100/gemma4_26b_a4b_it_${PPP4_ARM}_${SLURM_JOB_ID}"
mkdir -p "$GEN_DIR"
# benchmark_vllm_generation.py treats --out as an OUTPUT DIRECTORY (line 294:
# out_dir = ROOT / args.out; line 327: response_path = out_dir /
# f"responses_bs{bs}.jsonl") -- pass the directory, not a file path, or you
# get a nested responses_bs256.jsonl/responses_bs256.jsonl (caught live
# 2026-07-23 on job 422578). If resubmitting after a failed attempt, remove
# any stale GEN_DIR first -- a leftover malformed path from a prior run
# breaks the os.replace() that finalizes the output file (job 422579).
GEN_OUT="$GEN_DIR/responses_bs256.jsonl"
# Deliberately NOT passing --generation-max-tokens: both this script and
# build_teacher_cache.py compute the same per-record proportional budget via
# _generation_budget() (2x expected answer length + generation_extra_tokens)
# when no flat override is set. Forcing a flat --generation-max-tokens here
# without also setting cache.generation_max_tokens in the cache-build yaml
# makes the two sides disagree per-record and build_teacher_cache.py rejects
# it (ValueError: "generation response ... used budget ..., expected ... from
# cache config" -- caught live 2026-07-23 on job 422587).
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

echo "==== stage: teacher-hidden cache build $(date -Is) ===="
# Cache generation and cache reading use DIFFERENT configs by design: the
# builder's masker always writes remove-view student metadata regardless of
# mask.compaction, and build_teacher_cache.py enforces this by rejecting a
# source_compaction/mask.compaction mismatch at build time (ValueError:
# "cache.source_compaction is a training-reader selector; cache generation
# must leave it empty or equal mask.compaction" -- caught live 2026-07-23 on
# job 422578, which used the campaign yaml here by mistake). Both configs
# share cache.root (base.yaml default), so training reads the same cache
# this build produces.
"$PY" scripts/build_teacher_cache.py \
  --config configs/base.yaml \
  --experiment configs/teacher_references/teacher_cache_gemma4_26b_a4b_v5rs.yaml \
  --generation-responses "$GEN_OUT" || fail "build_teacher_cache"

# resolve_cache_dir's hash includes a sha of cache.generation_responses_path's
# CONTENT when that field is set (teacher/cache.py line ~91-94). The build
# step above sets it via --generation-responses; nothing on the training
# path (train.py/online_v4.py/launch_v4_stages.sh) ever sets it from a yaml
# field, so a plain training config resolves to a DIFFERENT cache directory
# than the one just built -- confirmed empirically 2026-07-23 (job 422589
# would have failed at this exact gate: build hash
# .../gemma-4-26B-A4B-it-rag_system-remove-98a4886... vs train hash
# .../gemma-4-26B-A4B-it-rag_system-remove-c357992..., a clean mismatch, not
# a flaky one). load_config only merges base.yaml + ONE experiment file, so
# the fix is to materialize patched copies of the training-side yamls with
# cache.generation_responses_path injected, matching what the build step
# used, and point every downstream train.py/launch_v4_stages.sh call at the
# patched copies instead of the static ones.
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
PATCHED_NUMGATE_SP="runs/.patched_${NUMGATE_SP_RUN}.yaml"
PATCHED_NUMGATE_SHARD="runs/.patched_${NUMGATE_SHARD_RUN}.yaml"
PATCHED_CAMPAIGN="runs/.patched_${CAMPAIGN_RUN}.yaml"
patch_generation_responses "$NUMGATE_SP_CFG" "$PATCHED_NUMGATE_SP"
patch_generation_responses "$NUMGATE_SHARD_CFG" "$PATCHED_NUMGATE_SHARD"
patch_generation_responses "$CAMPAIGN_CFG" "$PATCHED_CAMPAIGN"

# Training uses the durable-cache identity, but its immutable contents must
# still be node-local for the hot layer/cohort walk.  The override preserves
# the cache hash while redirecting its root to this H100 node's /dev/shm.
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
echo "==== stage: teacher-hidden cache -> /dev/shm $(date -Is) ===="
SELFUPDATE_TEACHER_CACHE_SOURCE="$CACHE_SOURCE_ROOT" \
SELFUPDATE_TEACHER_CACHE_DEST="$SELFUPDATE_TEACHER_CACHE_ROOT" \
  scripts/stage_teacher_cache_shm.sh "$CACHE_NAME" \
  || fail "stage_teacher_cache_shm"

# Both sides of the gate go through launch_v4_stages.sh, even the
# "single-process" one (a 1-stage config): a bare `train.py --v4-stage 0`
# invocation is missing env vars the launcher sets unconditionally
# (SELFUPDATE_V4_LAUNCH_ID, SELFUPDATE_V4_RELAY_ROOT) -- the in-pipeline
# distributed battery's _verify_entry does a real torch.distributed
# all_gather even for a single stage and strictly requires
# SELFUPDATE_V4_LAUNCH_ID (RuntimeError: "distributed battery requires
# SELFUPDATE_V4_LAUNCH_ID"), then fails on TCPStore rendezvous without the
# rest of the launcher's setup -- caught live 2026-07-23, job 422604. Rather
# than hand-thread each env var the launcher would have set, just use the
# launcher for both sides; it handles 1-stage configs correctly (STAGES=1,
# loop runs once).
wait_local_stages() {
  local lease="$1" max_iters="$2"
  for _ in $(seq 1 "$max_iters"); do
    [[ -f "$lease" ]] || return 0
    local alive=0
    while read -r entry; do
      local pid="${entry##*:}"
      kill -0 "$pid" 2>/dev/null && alive=1
    done < "$lease"
    [[ "$alive" -eq 0 ]] && return 0
    sleep 5
  done
  return 1
}

echo "==== GATE 1: numerics -- PPP4 then single-process, 1 epoch each $(date -Is) ===="
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_NUMGATE_SHARD" || fail "numgate PPP4 launch"
# The reference epoch is intentionally full-corpus.  Give each gate side the
# same explicit 120-minute budget and never compare incomplete artifacts.
wait_local_stages "runs/.v4-launch-${NUMGATE_SHARD_RUN}.pids" 1440 \
  || fail "numgate PPP4 stages did not finish within the 120m gate budget"
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_NUMGATE_SP" || fail "numgate single-process launch"
wait_local_stages "runs/.v4-launch-${NUMGATE_SP_RUN}.pids" 1440 \
  || fail "numgate single-process stage did not finish within the 120m gate budget"
"$PY" scripts/compare_v4_shard_numerics.py \
  "runs/${NUMGATE_SP_RUN}/stage0" "runs/${NUMGATE_SHARD_RUN}" --strict-current \
  || fail "NUMERICS GATE FAILED: single-process and PPP4 disagree -- gemma shared-KV path is likely broken, campaign would be invalid. Stopping before burning the reservation."
echo "==== GATE 1 PASSED $(date -Is) ===="

echo "==== GATE 2 (manual read, not auto-abort): epoch 1-3 learning signal ===="
echo "Check runs/${CAMPAIGN_RUN}_stage*.log for teacher-argmax"
echo "acceptance / recall moving after the campaign launch below -- a"
echo "shared-KV bug can pass Gate 1 (both paths agree) while both are wrong."

echo "==== stage: full 500-epoch PPP4 campaign $(date -Is) ===="
scripts/launch_v4_stages.sh configs/base.yaml \
  "$PATCHED_CAMPAIGN" || fail "campaign launch"
echo "launched; stage logs: runs/${CAMPAIGN_RUN}_stage*.log"
echo "evaluation is emitted in-training; no adapter merge is part of this campaign"

# Keep the job (and the reservation) alive so the detached stage processes
# are not orphaned when this script would otherwise exit; the stages run to
# completion or to the 24h wall time, whichever is first.
REAPER_LEASE="runs/.v4-launch-${CAMPAIGN_RUN}.pids.local"
while [[ -f "$REAPER_LEASE" ]]; do
  alive=0
  while read -r _pid _k; do
    kill -0 "$_pid" 2>/dev/null && alive=1
  done < "$REAPER_LEASE"
  [[ "$alive" -eq 0 ]] && break
  sleep 60
done
echo "==== campaign stages finished (or reaper lease gone) $(date -Is) ===="
