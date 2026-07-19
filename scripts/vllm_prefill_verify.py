"""vLLM's NATIVE teacher-forced verification: symmetric speed comparator.

Owner ask (2026-07-19): "vLLM prediction of the very last token of the
answers, with all the answer as input, compared with the same input in our
PPPn architecture." This is vLLM playing OUR verifier's role: input =
prompt + answer[:-1] (every token but the last), ask for exactly ONE
generated token (max_tokens=1) — vLLM's own prefill forward predicts the
final answer position. No autoregression on either side; both engines do
ONE forward pass over (almost) the full sequence. This is the fair
counterpart to our trainer's teacher_argmax_acceptance (which scores EVERY
answer position from one such forward, at no extra wall-clock cost in our
architecture — a batched forward doesn't get cheaper by wanting fewer output
rows, so "last token only" and "every position" cost the same on our side).

Reports: wall time for the whole batch (2071 items = one full epoch), and whether vLLM's own
prefill prediction matches its OWN earlier greedy answer (a self-consistency
sanity check: it must, since greedy decoding IS iterated argmax).

Run in the vLLM env:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python scripts/vllm_prefill_verify.py \
    --model Qwen/Qwen3.6-27B \
    --responses runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl \
    --out runs/spec_verify/27b_vllm_prefill_verify.json   # no --limit = full 2071-item epoch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

_node_tmp = Path(os.environ.get("SELFUPDATE_NODE_TMP",
                                f"/tmp/{os.environ.get('USER', 'selfupdate')}"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(_node_tmp / "selfupdate-vllm-cache"))
os.environ.setdefault("VLLM_CONFIG_ROOT", str(_node_tmp / "selfupdate-vllm-config"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(_node_tmp / "selfupdate-vllm-torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_node_tmp / "selfupdate-vllm-triton"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses", required=True,
                    help="vLLM responses jsonl (prompt_token_ids + token_ids)")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole file (2071 items = one full epoch)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.responses) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_token_ids") and r.get("token_ids"):
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} responses from {args.responses}", flush=True)

    from vllm import LLM, SamplingParams
    t_load = time.perf_counter()
    # enforce_eager=True (owner, 2026-07-19): skips torch.compile/CUDA-graph
    # capture entirely. Measured cause of multi-minute-plus hangs on TP4 for
    # 26B/31B (MoE + dense) — one case hit vLLM's OWN internal cancellation
    # at ~356s, another ran 1150s+ with a single thread pegged at ~100% CPU
    # (real Triton/Inductor codegen, not a stuck loop, but never converged in
    # reasonable time). A completed eager-mode measurement is more valuable
    # than a compiled-mode attempt that may never finish; eager will report
    # slower per-token throughput than compiled mode would — label results
    # accordingly, this is a real methodology difference, not a bug.
    llm_kw = dict(model=args.model, max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 tensor_parallel_size=args.tensor_parallel_size,
                 enforce_eager=True)
    if args.max_num_seqs:
        llm_kw["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kw)
    load_s = time.perf_counter() - t_load
    print(f"loaded in {load_s:.1f}s", flush=True)

    # input = prompt + answer[:-1]; ask for exactly the last answer token.
    prompts, golds, drop = [], [], 0
    for r in rows:
        p_ids = list(r["prompt_token_ids"])
        a_ids = list(r["token_ids"])
        if len(a_ids) < 1:
            drop += 1
            continue
        prompts.append({"prompt_token_ids": p_ids + a_ids[:-1]})
        golds.append(a_ids[-1])
    if drop:
        print(f"dropped {drop} empty-answer rows", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    match = sum(int(o.outputs[0].token_ids[0] == g)
                for o, g in zip(outs, golds))
    n = len(golds)
    seq_tokens = sum(len(p["prompt_token_ids"]) for p in prompts)
    summary = {
        "model": args.model,
        "responses": args.responses,
        "enforce_eager": True,
        "items": n,
        "self_consistency_match_rate": round(match / max(n, 1), 4),
        "seconds": round(dt, 3),
        "items_per_s": round(n / max(dt, 1e-9), 2),
        "total_context_tokens": seq_tokens,
        "context_tok_per_s": round(seq_tokens / max(dt, 1e-9), 1),
        "load_seconds": round(load_s, 2),
        "note": ("ONE forward pass per item (max_tokens=1), no autoregression "
                "— the symmetric counterpart to our trainer's single "
                "teacher-forced forward. self_consistency should be ~1.0: "
                "greedy decoding IS iterated argmax, so vLLM's own prefill "
                "here must reproduce its own earlier generated last token."),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
