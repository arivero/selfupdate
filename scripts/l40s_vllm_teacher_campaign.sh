#!/usr/bin/env bash
# V5RS L40S campaign: external vLLM answer generation followed by the
# in-repo exact-token hidden-state cache pass.  vLLM and the cache phase are
# deliberately separate processes and separate runtimes.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"
OUTROOT="$ROOT/runs/l40s_benchmark"
VLLMROOT="$OUTROOT/vllm"
CACHEROOT="$OUTROOT/cache_timings"
LOGROOT="$OUTROOT/logs"
mkdir -p "$VLLMROOT" "$CACHEROOT" "$LOGROOT"

export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SSL_CERT_FILE="/fs/agustina/arivero/supercomplex/.local/lib/python3.11/site-packages/certifi/cacert.pem"

run_isolated() {
  setsid "$@" &
  local leader=$! rc=0
  wait "$leader" || rc=$?
  kill -TERM -- "-$leader" 2>/dev/null || true
  sleep 0.2
  kill -KILL -- "-$leader" 2>/dev/null || true
  return "$rc"
}

run_one() {
  local tag="$1" model="$2" visible="$3" placement="$4" format="${5:-native}"
  local out="$VLLMROOT/$tag" log="$LOGROOT/$tag.log"
  local tmp="/tmp/arivero/l40s-cache-$tag"
  local -a vllm_parallel=() cache_parallel=()
  case "$placement" in
    single) ;;
    pp2) vllm_parallel=(--pipeline-parallel-size 2); cache_parallel=(--pipeline-split 18) ;;
    pp4) vllm_parallel=(--pipeline-parallel-size 4); cache_parallel=(--pipeline-splits 9 18 27) ;;
    *) echo "unknown placement $placement" >&2; return 2 ;;
  esac
  if [[ -f "$CACHEROOT/$tag/timings.json" ]]; then
    echo "SKIP $tag (timing already present)" >> "$OUTROOT/campaign.log"
    return 0
  fi
  echo "START $tag model=$model visible=$visible placement=$placement" >> "$OUTROOT/campaign.log"
  rm -rf "$tmp"
  mkdir -p "$out" "$CACHEROOT/$tag"
  if [[ ! -f "$out/responses_bs64.jsonl" ]]; then
    run_isolated env CUDA_VISIBLE_DEVICES="$visible" "$PY" \
      "$ROOT/scripts/benchmark_vllm_generation.py" \
      --model "$model" --batch-sizes 64 --max-num-seqs 64 \
      --gpu-memory-utilization 0.85 --max-model-len 4096 \
      --use-cudagraphs --progress --prompt-format "$format" \
      "${vllm_parallel[@]}" --out "$out" >> "$log" 2>&1
    local vrc=$?
    if [[ "$vrc" != 0 || ! -f "$out/responses_bs64.jsonl" ]]; then
      echo "VLLM_FAIL $tag rc=$vrc" >> "$OUTROOT/campaign.log"
      return "$vrc"
    fi
  else
    echo "REUSE_VLLM $tag" >> "$OUTROOT/campaign.log"
  fi
  rm -rf "$tmp"
  local response_rel="runs/l40s_benchmark/vllm/$tag/responses_bs64.jsonl"
  CUDA_VISIBLE_DEVICES="$visible" scripts/container_exec.sh python \
    scripts/build_teacher_cache.py \
    --config configs/teacher_references/teacher_cache_qwen36_35b_a3b_v5rs.yaml \
    --model "$model" --teacher-batch 64 --max-sequence-tokens 8192 \
    --hidden-dtype bfloat16 --cache-root "$tmp" \
    --generation-responses "$response_rel" \
    "${cache_parallel[@]}" >> "$log" 2>&1
  local crc=$?
  local timing
  timing="$(find "$tmp" -name timings.json -print -quit 2>/dev/null || true)"
  if [[ -n "$timing" ]]; then
    cp "$timing" "$CACHEROOT/$tag/timings.json"
    cp "$out/summary.json" "$CACHEROOT/$tag/vllm_summary.json" 2>/dev/null || true
    echo "DONE $tag cache_rc=$crc" >> "$OUTROOT/campaign.log"
  else
    echo "CACHE_FAIL $tag rc=$crc" >> "$OUTROOT/campaign.log"
  fi
  rm -rf "$tmp"
  return "$crc"
}

run_pair() {
  run_one "$1" "$2" "$3" single "${4:-native}" & local p0=$!
  run_one "$5" "$6" "$7" single "${8:-native}" & local p1=$!
  wait "$p0" || true
  wait "$p1" || true
}

echo "CAMPAIGN START commit=$(git -C "$ROOT" rev-parse HEAD) host=$(hostname -s)" > "$OUTROOT/campaign.log"

# One-card models, paired across the four L40S devices.
run_pair qwen3_0p6b Qwen/Qwen3-0.6B 0 native qwen35_0p8b Qwen/Qwen3.5-0.8B 1 native
run_pair qwen3_1p7b Qwen/Qwen3-1.7B 0 native qwen35_2b Qwen/Qwen3.5-2B 1 native
run_pair qwen3_4b Qwen/Qwen3-4B 0 native qwen35_9b Qwen/Qwen3.5-9B 1 native
run_pair qwen3_8b Qwen/Qwen3-8B 0 native llama31_8b meta-llama/Meta-Llama-3.1-8B-Instruct 1 native
run_pair qwen3_14b Qwen/Qwen3-14B 0 native phi4 microsoft/Phi-4 1 native
run_pair gpt_oss_20b openai/gpt-oss-20b 0 native nemotron_nano_9b nvidia/NVIDIA-Nemotron-Nano-9B-v2 1 native
run_one mistral_7b mistralai/Mistral-7B-Instruct-v0.1 0 single native || true

# Larger models: two-card PP, with a four-card PP run for the 120B point.
run_one gemma4_26b google/gemma-4-26B-A4B-it 0,1 pp2 native || true
run_one qwen36_35b Qwen/Qwen3.6-35B-A3B 0,1 pp2 native || true
run_one qwen36_27b Qwen/Qwen3.6-27B 0,1 pp2 native || true
run_one gemma4_31b google/gemma-4-31B-it 0,1 pp2 native || true
run_one qwen3_32b Qwen/Qwen3-32B 0,1 pp2 native || true
run_one alia_40b BSC-LT/ALIA-40b-fc-2606 0,1 pp2 native || true
run_one gpt_oss_120b openai/gpt-oss-120b 0,1,2,3 pp4 gpt_oss_harmony || true

echo "CAMPAIGN COMPLETE" >> "$OUTROOT/campaign.log"
