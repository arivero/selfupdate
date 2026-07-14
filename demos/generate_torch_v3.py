"""Round 3: preallocated KV buffer + live-prefix views — CPU and GPU.

Round 2 exposed the dilemma exactly: DynamicCache attends over only the
real tokens but copies the whole cache every step; StaticCache writes in
place but attends over the full preallocated width. Round 3 refuses the
dilemma with a ~30-line custom cache layer:

  * KV buffers are preallocated once per batch (no copy per token),
  * `update()` writes in place at `cache_position` and returns a **sliced
    view** of the live prefix (no dead width, no allocation),
  * the attention mask is preallocated too and sliced per step (no cat).

This is PagedAttention degenerated to one contiguous page per sequence —
the part of the trick that IS reachable from HF building blocks. What
remains unreachable: continuous batching and CUDA graphs.

  CUDA_VISIBLE_DEVICES=0 .venv/bin/python demos/generate_torch_v3.py \
      --device cuda --prompts demos/out/prompts_qwen3-0.6b_n64.jsonl
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
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--cores", default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--compact-threshold", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cores:
        import os
        lo, hi = args.cores.split("-")
        os.sched_setaffinity(0, range(int(lo), int(hi) + 1))

    import torch

    use_cuda = args.device == "cuda"
    if use_cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)  # round-1 finding
    else:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from transformers.cache_utils import DynamicLayer

    class PreallocLayer(DynamicLayer):
        """In-place KV writes into a preallocated buffer; live-prefix views."""

        total_len: int = 0  # set by prealloc_cache()

        def lazy_initialization(self, key_states):
            b, h, _, d = key_states.shape
            self.keys = torch.zeros((b, h, self.total_len, d),
                                    dtype=key_states.dtype,
                                    device=key_states.device)
            self.values = torch.zeros_like(self.keys)
            self.cumulative_length = 0
            self.is_initialized = True

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states)
            new = key_states.shape[2]
            start = self.cumulative_length
            self.keys[:, :, start:start + new] = key_states
            self.values[:, :, start:start + new] = value_states
            self.cumulative_length = start + new
            end = self.cumulative_length
            return self.keys[:, :, :end], self.values[:, :, :end]

        def get_seq_length(self, cache_position=None) -> int:
            return self.cumulative_length

        def compact(self, keep) -> None:
            self.keys = self.keys.index_select(0, keep)
            self.values = self.values.index_select(0, keep)

    def prealloc_cache(total_len: int) -> DynamicCache:
        cache = DynamicCache()

        class _Layer(PreallocLayer):
            pass

        _Layer.total_len = total_len
        cache.layer_class_to_replicate = _Layer
        return cache

    meta, prompts = load_prompts(Path(args.prompts))
    model_name = args.model or meta.get("model", "Qwen/Qwen3-0.6B")
    out_dir = Path(args.out) if args.out else HERE / "out" / f"torch_v3_{args.device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    load_seconds = time.perf_counter() - t_load

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["ids"]))
    results: list[dict | None] = [None] * len(prompts)

    total_gen_tokens = 0
    timing = {"prefill": 0.0, "decode": 0.0}
    t_gen = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            batch = [prompts[i] for i in batch_idx]
            bsz = len(batch)
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

            cache = prealloc_cache(total_len)
            t_pre = time.perf_counter()
            out = model(input_ids=input_ids,
                        attention_mask=full_mask[:, :max_len],
                        position_ids=position_ids, past_key_values=cache,
                        use_cache=True)
            next_tokens = out.logits[:, -1, :].argmax(-1)
            if use_cuda:
                torch.cuda.synchronize()
            timing["prefill"] += time.perf_counter() - t_pre
            t_dec = time.perf_counter()

            stops = torch.tensor([x["stop_id"] for x in batch], device=device)
            budgets = torch.tensor([x["budget"] for x in batch], device=device)
            next_pos = position_ids[:, -1] + 1
            generated: list[list[int]] = [[] for _ in batch]
            row2orig = list(range(bsz))
            emitted = torch.zeros(bsz, dtype=torch.long, device=device)

            step = 0
            while True:
                toks = next_tokens.tolist()
                for row, token in enumerate(toks):
                    generated[row2orig[row]].append(token)
                emitted += 1
                done = (next_tokens == stops) | (emitted >= budgets)
                if bool(done.all()):
                    break
                if float(done.float().mean()) >= args.compact_threshold:
                    keep = (~done).nonzero(as_tuple=True)[0]
                    for layer in cache.layers:
                        layer.compact(keep)
                    next_tokens = next_tokens[keep]
                    full_mask = full_mask[keep]
                    next_pos = next_pos[keep]
                    stops, budgets = stops[keep], budgets[keep]
                    emitted = emitted[keep]
                    row2orig = [row2orig[int(i)] for i in keep.tolist()]
                    done = done[keep]
                full_mask[:, max_len + step] = 1
                out = model(input_ids=next_tokens.unsqueeze(1),
                            attention_mask=full_mask[:, :max_len + step + 1],
                            position_ids=next_pos.unsqueeze(1),
                            past_key_values=cache, use_cache=True)
                next_tokens = out.logits[:, -1, :].argmax(-1)
                next_tokens = torch.where(done, stops, next_tokens)
                next_pos += 1
                step += 1
            if use_cuda:
                torch.cuda.synchronize()
            timing["decode"] += time.perf_counter() - t_dec

            for row, item in enumerate(batch):
                token_ids = generated[row]
                stop_id = item["stop_id"]
                if stop_id in token_ids:
                    token_ids = token_ids[:token_ids.index(stop_id) + 1]
                    hard_cut = False
                else:
                    token_ids = token_ids + [stop_id]
                    hard_cut = True
                total_gen_tokens += len(token_ids)
                results[batch_idx[row]] = {
                    "example_id": item["example_id"], "gen_tokens": len(token_ids),
                    "token_ids": token_ids, "answer_tokens": len(token_ids) - 1,
                    "hard_cut": hard_cut, "answer_text": tok.decode(token_ids[:-1]),
                }

    gen_seconds = time.perf_counter() - t_gen
    (out_dir / "responses.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results),
        encoding="utf-8")
    summary = {
        "engine": f"torch_v3_{args.device}", "model": model_name,
        "dtype": args.dtype,
        "device": (torch.cuda.get_device_name(0) if use_cuda
                   else f"cpu[{args.threads}threads,{args.cores}]"),
        "batch_size": args.batch_size,
        "compact_threshold": args.compact_threshold,
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
