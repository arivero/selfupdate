"""Fixed-length greedy vLLM drafts (ignore_eos) for the depth-decay probe.

Why: big models end answers at ~8-16 tokens by choice (EOS), so their verify
rows sample only SHALLOW positions; 0.8B's ~115-token answers sample deep
ones. To separate "small-model margins" from "depth accumulation" in the
transformers-vs-vLLM linear-attention divergence, force every draft to a
fixed length with ignore_eos=True and measure acceptance vs position. The
post-EOS tail is degenerate text, but per-position argmax agreement remains
well-defined — the draft is whatever vLLM emitted.

Run in the vLLM env:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python scripts/vllm_longdraft.py \
    --model Qwen/Qwen3.6-27B \
    --responses runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl \
    --max-tokens 128 --limit 64 --out runs/spec_verify/27b_longdraft128.jsonl
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
                    help="existing responses jsonl; only prompt_token_ids reused")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap concurrent sequences; hybrid/linear-attn "
                         "models size their Mamba cache to this (default "
                         "1024 can exceed available blocks on small runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.responses) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_token_ids"):
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} prompts from {args.responses}", flush=True)

    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm_kw = dict(model=args.model, max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 tensor_parallel_size=args.tensor_parallel_size)
    if args.max_num_seqs:
        llm_kw["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kw)
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0,
                        max_tokens=args.max_tokens, ignore_eos=True)
    t1 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": r["prompt_token_ids"]}
                         for r in rows], sp, use_tqdm=False)
    gen_s = time.perf_counter() - t1
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(outp, "w") as fh:
        for r, o in zip(rows, outs):
            tids = list(o.outputs[0].token_ids)
            n_tok += len(tids)
            fh.write(json.dumps({"example_id": r.get("example_id"),
                                 "prompt_token_ids": r["prompt_token_ids"],
                                 "token_ids": tids,
                                 "ignore_eos": True}) + "\n")
    print(json.dumps({"items": len(rows), "answer_tokens": n_tok,
                      "gen_seconds": round(gen_s, 2),
                      "tok_per_s": round(n_tok / max(gen_s, 1e-9), 1),
                      "out": str(outp)}), flush=True)


if __name__ == "__main__":
    main()
