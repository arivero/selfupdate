"""Round 2: pure-torch V5 generator with StaticCache — CPU and GPU.

Round-1 losses were bookkeeping, not math: DynamicCache re-copies the whole
KV cache every token, the attention mask was re-`cat`ed every step, and (on
GPU) the cuDNN SDPA backend rebuilt plans per step. Round 2 keeps the same
outer algorithm (length-sorted batches, greedy, per-record budgets, stop
token, retirement compaction) and changes the machinery:

1. `StaticCache`: KV written in place into a preallocated buffer.
2. Full-width attention mask allocated once per batch and updated in place —
   no per-step allocation, and constant tensor shapes.
3. Compaction rounds the surviving batch up to a power of two (padding with
   inert finished rows), so shapes only ever halve: bounded torch.compile
   recompiles, still most of the straggler win.
4. Optional `--compile`: decode steps run through `torch.compile(model)`;
   compile+warmup happens in the (discounted) load phase.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demos/generate_torch_v2.py \
      --device cuda --prompts demos/out/prompts_qwen3-0.6b_n64.jsonl
  CUDA_VISIBLE_DEVICES= .venv/bin/python demos/generate_torch_v2.py \
      --device cpu --threads 32 --cores 0-31 --prompts ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_prompts(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    meta = rows[0] if rows and rows[0].get("meta") else {}
    return meta, [x for x in rows if not x.get("meta")]


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--cores", default=None,
                    help="CPU affinity range like '0-31'; see round-1 lesson")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cores:
        import os
        lo, hi = args.cores.split("-")
        os.sched_setaffinity(0, range(int(lo), int(hi) + 1))

    import torch

    use_cuda = args.device == "cuda"
    if use_cuda:
        # Round-1 finding: cuDNN SDPA rebuilds CPU-side plans every decode
        # step (347ms/step self-CPU). Keep it out of dispatch.
        torch.backends.cuda.enable_cudnn_sdp(False)
    else:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    meta, prompts = load_prompts(Path(args.prompts))
    model_name = args.model or meta.get("model", "Qwen/Qwen3-0.6B")
    out_dir = Path(args.out) if args.out else HERE / "out" / f"torch_v2_{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    decode_model = model
    if args.compile:
        decode_model = torch.compile(model)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # In compile mode every batch is padded to ONE global shape pair so the
    # compiled decode graph is hit by every step of every batch; without
    # compile, per-batch shapes are tighter and cheaper.
    global_max_len = max(len(x["ids"]) for x in prompts)
    global_total_len = global_max_len + max(x["budget"] for x in prompts) + 1

    def run_batch(batch: list[dict], timing: dict | None) -> list[dict]:
        bsz = len(batch)
        if args.compile:
            max_len, total_len = global_max_len, global_total_len
        else:
            max_len = max(len(x["ids"]) for x in batch)
            total_len = max_len + max(x["budget"] for x in batch) + 1
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        full_mask = torch.zeros((bsz, total_len), dtype=torch.long)
        for row, item in enumerate(batch):
            ids = item["ids"]
            input_ids[row, max_len - len(ids):] = torch.tensor(ids)
            full_mask[row, max_len - len(ids):max_len] = 1
        input_ids, full_mask = input_ids.to(device), full_mask.to(device)
        position_ids = (full_mask[:, :max_len].cumsum(-1) - 1).clamp(min=0)

        cache = StaticCache(config=model.config, max_cache_len=total_len)
        t_pre = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids,
                        attention_mask=full_mask[:, :max_len],
                        position_ids=position_ids, past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.arange(max_len, device=device))
            next_tokens = out.logits[:, -1, :].argmax(-1)
        if use_cuda:
            torch.cuda.synchronize()
        if timing is not None:
            timing["prefill"] += time.perf_counter() - t_pre
        t_dec = time.perf_counter()

        stops = torch.tensor([x["stop_id"] for x in batch], device=device)
        budgets = torch.tensor([x["budget"] for x in batch], device=device)
        next_pos = position_ids[:, -1] + 1
        generated: list[list[int]] = [[] for _ in batch]
        row2orig = list(range(bsz))
        emitted = torch.zeros(bsz, dtype=torch.long, device=device)

        step = 0
        with torch.inference_mode():
            while True:
                toks = next_tokens.tolist()
                for row, token in enumerate(toks):
                    generated[row2orig[row]].append(token)
                emitted += 1
                done = (next_tokens == stops) | (emitted >= budgets)
                if bool(done.all()):
                    break
                n_active = int((~done).sum())
                new_size = next_pow2(n_active)
                if args.compile:
                    # Only B and B/2 are compile-warmed; deeper shrink would
                    # recompile inside the timed region for a tail of steps.
                    new_size = max(new_size, args.batch_size // 2)
                if new_size < len(row2orig):
                    # Shrink to the next power of two, padding the keep-set
                    # with inert finished rows so shapes only ever halve.
                    active = (~done).nonzero(as_tuple=True)[0]
                    finished = done.nonzero(as_tuple=True)[0]
                    keep = torch.cat([active, finished[:new_size - n_active]])
                    # StaticLayer has no batch_select_indices (transformers
                    # 5.12); row-select its preallocated buffers directly.
                    for layer in cache.layers:
                        if layer.keys is not None:
                            layer.keys = layer.keys.index_select(0, keep)
                            layer.values = layer.values.index_select(0, keep)
                    next_tokens = next_tokens[keep]
                    full_mask = full_mask[keep]
                    next_pos = next_pos[keep]
                    stops, budgets = stops[keep], budgets[keep]
                    emitted = emitted[keep]
                    row2orig = [row2orig[int(i)] for i in keep.tolist()]
                    done = done[keep]
                full_mask[:, max_len + step] = 1
                out = decode_model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=full_mask,
                    position_ids=next_pos.unsqueeze(1),
                    past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([max_len + step], device=device))
                next_tokens = out.logits[:, -1, :].argmax(-1)
                next_tokens = torch.where(done, stops, next_tokens)
                next_pos += 1
                step += 1
        if use_cuda:
            torch.cuda.synchronize()
        if timing is not None:
            timing["decode"] += time.perf_counter() - t_dec

        results = []
        for row, item in enumerate(batch):
            token_ids = generated[row]
            stop_id = item["stop_id"]
            if stop_id in token_ids:
                token_ids = token_ids[:token_ids.index(stop_id) + 1]
                hard_cut = False
            else:
                token_ids = token_ids + [stop_id]
                hard_cut = True
            results.append({
                "example_id": item["example_id"], "gen_tokens": len(token_ids),
                "token_ids": token_ids, "answer_tokens": len(token_ids) - 1,
                "hard_cut": hard_cut, "answer_text": tok.decode(token_ids[:-1]),
            })
        return results

    if args.compile:
        # Warm the compile caches during the discounted load phase with a
        # dummy batch of the shapes the real run will hit.
        warm = [{"example_id": "warm", "ids": [1] * 8, "budget": 8,
                 "stop_id": -1}] * args.batch_size
        run_batch(warm, None)
        run_batch(warm[:args.batch_size // 2], None)
    load_seconds = time.perf_counter() - t_load

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["ids"]))
    results: list[dict | None] = [None] * len(prompts)
    timing = {"prefill": 0.0, "decode": 0.0}
    t_gen = time.perf_counter()
    for start in range(0, len(order), args.batch_size):
        batch_idx = order[start:start + args.batch_size]
        for local, res in enumerate(run_batch([prompts[i] for i in batch_idx],
                                              timing)):
            results[batch_idx[local]] = res
    gen_seconds = time.perf_counter() - t_gen

    total_gen_tokens = sum(x["gen_tokens"] for x in results)
    (out_dir / "responses.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8")
    summary = {
        "engine": f"torch_v2_{args.device}" + ("_compiled" if args.compile else ""),
        "model": model_name, "dtype": args.dtype,
        "device": (torch.cuda.get_device_name(0) if use_cuda
                   else f"cpu[{args.threads}threads,{args.cores}]"),
        "batch_size": args.batch_size,
        "prompts": len(prompts), "load_seconds": round(load_seconds, 2),
        "generate_seconds": round(gen_seconds, 2),
        "prefill_seconds": round(timing["prefill"], 2),
        "decode_seconds": round(timing["decode"], 2),
        "gen_tokens": total_gen_tokens,
        "tokens_per_second": round(total_gen_tokens / gen_seconds, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
