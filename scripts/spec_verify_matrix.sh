#!/usr/bin/env bash
# Trainer-native vLLM-acceptance matrix (owner goal 2026-07-19): for each
# model with a usable vLLM responses file (prompt_token_ids + token_ids),
# build a 64-item subset + index-only cache and run the v4 PPP1 trainer;
# the goal metric is teacher_argmax_acceptance in its teacher_output_eval row.
#
#   scripts/spec_verify_matrix.sh 0p8b|27b|35b|26b|31b|122b [gpu]
#
# 0.8B rows PPP1/2/4 already measured bit-identical (runs/spec_verify/
# RESULTS.md). PPP8x: launch qwen35_0p8b_v4_spec_ppp8x.yaml with
# SELFUPDATE_V4_STAGE_HOSTS (see its header) after venv+HF-stage+cache exist
# on BOTH nodes. 397B needs the 2-stage relay verify (does not fit one node).
# DeepSeek first needs responses regenerated WITH prompt_token_ids.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/tmp/$USER/selfupdate-venv/bin/python
GPU="${2:-0}"

case "${1:?usage: spec_verify_matrix.sh 0p8b|27b|35b|26b|31b|122b [gpu]}" in
  0p8b) MODEL=Qwen/Qwen3.5-0.8B     RESP=runs/spec_verify/qwen35_0p8b_vllm/responses_bs64.jsonl ;;
  27b)  MODEL=Qwen/Qwen3.6-27B      RESP=runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl ;;
  35b)  MODEL=Qwen/Qwen3.6-35B-A3B  RESP=runs/vllm_h100/qwen36_35b_a3b/responses_bs256.jsonl ;;
  26b)  MODEL=google/gemma-4-26B-A4B-it RESP=runs/vllm_h100/gemma4_26b_a4b_it/responses_bs256.jsonl ;;
  31b)  MODEL=google/gemma-4-31B-it RESP=runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl ;;
  122b) MODEL=Qwen/Qwen3.5-122B-A10B RESP=runs/vllm_h100/qwen35_122b_a10b/responses_bs256.jsonl ;;
  *) echo "unknown model key $1" >&2; exit 2 ;;
esac
KEY="$1"; SHORT="${MODEL##*/}"

export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_ALLOC_CONF=expandable_segments:True SELFUPDATE_CPU_THREADS=8

# 1. 64-item subset of the responses' example ids (dataset order) + matching
#    subset responses file (build_teacher_cache requires exact coverage).
SUBEX="data/combined/examples_v5rs_window_spec64_${KEY}.jsonl"
SUBRESP="runs/spec_verify/${KEY}_responses64.jsonl"
$PY - "$RESP" "$SUBEX" "$SUBRESP" <<'PYEOF'
import json, sys
resp_path, subex, subresp = sys.argv[1:4]
rows = [json.loads(l) for l in open(resp_path) if l.strip()]
by_id = {r["example_id"]: r for r in rows if r.get("prompt_token_ids") and r.get("token_ids")}
kept, out_ex = [], open(subex, "w")
for l in open("data/combined/examples_v5rs_window.jsonl"):
    ex = json.loads(l)
    if ex["example_id"] in by_id and len(kept) < 64:
        out_ex.write(l); kept.append(ex["example_id"])
out_ex.close()
with open(subresp, "w") as f:
    for i in kept:
        f.write(json.dumps(by_id[i]) + "\n")
print(f"subset {len(kept)} items -> {subex} + {subresp}")
assert len(kept) == 64, "need 64 usable responses"
PYEOF

# 2. per-model base config from the 0.8B template (paths + names swapped).
BASE="configs/experiments/spec_verify/base_${KEY}_v4_spec.yaml"
EXP="configs/experiments/spec_verify/${KEY}_v4_spec_ppp1.yaml"
if [[ "$KEY" != 0p8b ]]; then
  sed -e "s|name: Qwen/Qwen3.5-0.8B|name: ${MODEL}|" \
      -e "s|examples_v5rs_window_spec64.jsonl|examples_v5rs_window_spec64_${KEY}.jsonl|" \
      -e "s|runs/spec_verify/qwen35_0p8b_vllm/responses_bs64.jsonl|${SUBRESP}|" \
      -e "s|spec_qwen35_0p8b_v4_base_never_train|spec_${KEY}_v4_base_never_train|" \
      -e "s|selfupdate-teacher-cache-v4-spec0p8b|selfupdate-teacher-cache-v4-spec${KEY}|" \
      -e "s|caches/spec_0p8b_v4|caches/spec_${KEY}_v4|" \
      -e "s|generation_max_tokens: 0|generation_max_tokens: 4096|" \
      configs/experiments/spec_verify/base_qwen35_0p8b_v4_spec.yaml > "$BASE"
  # historical h100 responses were generated with a FLAT 4096 budget
  # (generation_budget: 4096 per row); the 0.8B template derives per-record
  # budgets (generation_max_tokens: 0) — batch 420477 failed on this mismatch.
  # PHYSICAL device id in the config, never CUDA_VISIBLE_DEVICES renumbering
  # (CLAUDE.md multi-node law; the trainer's NVML stray-context guard reads
  # physical indices and aborts under a CVD remap — measured 2026-07-19).
  sed -e "s|spec_qwen35_0p8b_v4_ppp1_e1|spec_${KEY}_v4_ppp1_e1|" \
      -e "s|device: cuda:0|device: cuda:${GPU}|" \
      configs/experiments/spec_verify/qwen35_0p8b_v4_spec_ppp1.yaml > "$EXP"
fi

# 3. index-only cache (no model load) + PPP1 trainer run on one GPU.
$PY scripts/build_teacher_cache.py --config "$BASE" --experiment "$EXP" \
  --index-only --coordinated-node-cache
$PY scripts/train.py --config "$BASE" --experiment "$EXP"

# 4. extract the goal metric.
RUN="runs/spec_${KEY}_v4_ppp1_e1"
[[ "$KEY" == 0p8b ]] && RUN="runs/spec_qwen35_0p8b_v4_ppp1_e1"
echo "==== GOAL ROW ${SHORT} ===="
grep '"teacher_output_eval"' "$RUN/metrics.jsonl" | $PY -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    print({k: d[k] for k in ('epoch','teacher_argmax_acceptance',
          'student_argmax_acceptance','answer_token_count') if k in d})"
