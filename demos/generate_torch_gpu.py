"""Pure-torch GPU batched greedy generator for the V5 workload — demo 2.

Same algorithm as generate_torch_cpu.py (length-sorted static batches, KV
cached decode, retirement compaction) moved to one CUDA device, eager mode:
no CUDA graphs, no torch.compile, no vLLM. The race partner is standard vLLM
(graphs allowed) on a second GPU of the same kind.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demos/generate_torch_gpu.py \
      --prompts demos/out/prompts_qwen3-0.6b_n64.jsonl --batch-size 64
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--compact-threshold", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    meta, prompts = load_prompts(Path(args.prompts))
    model_name = args.model or meta.get("model", "Qwen/Qwen3-0.6B")
    out_dir = Path(args.out) if args.out else HERE / "out" / "torch_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t_load

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["ids"]))
    results: list[dict | None] = [None] * len(prompts)

    total_gen_tokens = 0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    t_gen = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            batch = [prompts[i] for i in batch_idx]
            bsz = len(batch)
            max_len = max(len(x["ids"]) for x in batch)
            input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
            mask = torch.zeros((bsz, max_len), dtype=torch.long)
            for row, item in enumerate(batch):
                ids = item["ids"]
                input_ids[row, max_len - len(ids):] = torch.tensor(ids)
                mask[row, max_len - len(ids):] = 1
            input_ids, mask = input_ids.to(device), mask.to(device)
            position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

            cache = DynamicCache()
            t_pre = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=mask,
                        position_ids=position_ids, past_key_values=cache,
                        use_cache=True)
            next_tokens = out.logits[:, -1, :].argmax(-1)
            torch.cuda.synchronize()
            prefill_seconds += time.perf_counter() - t_pre
            t_dec = time.perf_counter()

            stops = torch.tensor([x["stop_id"] for x in batch], device=device)
            budgets = torch.tensor([x["budget"] for x in batch], device=device)
            next_pos = position_ids[:, -1] + 1
            generated: list[list[int]] = [[] for _ in batch]
            row2orig = list(range(bsz))
            emitted = torch.zeros(bsz, dtype=torch.long, device=device)

            while True:
                # One .tolist() per step is the only host sync in the loop.
                toks = next_tokens.tolist()
                for row, token in enumerate(toks):
                    generated[row2orig[row]].append(token)
                emitted += 1
                done = (next_tokens == stops) | (emitted >= budgets)
                if bool(done.all()):
                    break
                if float(done.float().mean()) >= args.compact_threshold:
                    keep = (~done).nonzero(as_tuple=True)[0]
                    cache.batch_select_indices(keep)
                    next_tokens = next_tokens[keep]
                    mask = mask[keep]
                    next_pos = next_pos[keep]
                    stops, budgets, emitted = stops[keep], budgets[keep], emitted[keep]
                    row2orig = [row2orig[int(i)] for i in keep.tolist()]
                    done = done[keep]
                mask = torch.cat([mask, torch.ones((mask.shape[0], 1),
                                                   dtype=torch.long, device=device)], dim=1)
                out = model(input_ids=next_tokens.unsqueeze(1), attention_mask=mask,
                            position_ids=next_pos.unsqueeze(1), past_key_values=cache,
                            use_cache=True)
                next_tokens = out.logits[:, -1, :].argmax(-1)
                next_tokens = torch.where(done, stops, next_tokens)
                next_pos += 1
            torch.cuda.synchronize()
            decode_seconds += time.perf_counter() - t_dec

            for row, item in enumerate(batch):
                token_ids = generated[row]
                stop_id = item["stop_id"]
                if stop_id in token_ids:
                    token_ids = token_ids[:token_ids.index(stop_id) + 1]
                    hard_cut = False
                else:
                    token_ids = token_ids + [stop_id]
                    hard_cut = True
                text = tok.decode(token_ids[:-1])
                total_gen_tokens += len(token_ids)
                results[batch_idx[row]] = {
                    "example_id": item["example_id"], "gen_tokens": len(token_ids),
                    "token_ids": token_ids, "answer_tokens": len(token_ids) - 1,
                    "hard_cut": hard_cut, "answer_text": text,
                }

    gen_seconds = time.perf_counter() - t_gen
    (out_dir / "responses.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8")
    summary = {
        "engine": "torch_gpu_eager", "model": model_name, "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "compact_threshold": args.compact_threshold,
        "prompts": len(prompts), "load_seconds": round(load_seconds, 2),
        "generate_seconds": round(gen_seconds, 2),
        "prefill_seconds": round(prefill_seconds, 2),
        "decode_seconds": round(decode_seconds, 2),
        "gen_tokens": total_gen_tokens,
        "tokens_per_second": round(total_gen_tokens / gen_seconds, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
