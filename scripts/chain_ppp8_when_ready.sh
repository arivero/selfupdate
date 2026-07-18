#!/usr/bin/env bash
# Chain: wait for 122B staged on agpuh02 -> build its index-only teacher
# cache -> launch 122B PPP8 cross-node. Detached; the agent supervises via
# runs/chain_ppp8.log and the stage logs.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
LOG=runs/chain_ppp8.log
say(){ echo "$(date -Is) $*" >> "$LOG"; }
say "chain start"

# 1. Wait for agpuh02 122B stage-ready marker.
until ssh agpuh02 'test -f /dev/shm/arivero/selfupdate-hf-cache/.selfupdate-hf-stage-ready'; do
  sleep 20
done
say "agpuh02 122B staged"

# 2. Build the index-only node cache on agpuh02 (tokenizer only; answers
#    come from the shared-FS vLLM responses).
ssh agpuh02 "cd '$PWD' && CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error HF_HOME=/dev/shm/arivero/selfupdate-hf-cache HF_HUB_OFFLINE=1 /tmp/\$USER/selfupdate-venv/bin/python scripts/build_teacher_cache.py --config configs/experiments/h100_smoke/base_qwen35_122b_v4_full.yaml --experiment configs/experiments/h100_smoke/qwen35_122b_v4_ppp8_xnode.yaml --coordinated-node-cache --index-only" < /dev/null >> "$LOG" 2>&1
say "agpuh02 cache build rc=$?"

# 3. Launch PPP8 (agpuh01 head).
say "launching 122B PPP8"
scripts/launch_q122b_ppp8x.sh >> "$LOG" 2>&1
say "launcher exited rc=$?"
