"""Authentic layerwise speed checks with minibatches.

This benchmark runs the real HF model through ``BlockStack`` and performs a
layerwise backward walk over every decoder block. It is synthetic only in the
targets: hidden targets are random tensors with the same shape/device pattern
as cached teacher targets. That keeps the check independent of data/cache/GPU
campaign state while exercising the expensive path that determines throughput.

It reports three variants per batch size:
  - gpu_targets: targets already resident on GPU
  - cpu_targets: each block target copied from CPU inside the block loop
  - sync_each_block: GPU targets, but scalar logging syncs after each block

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/speed_check.py --batches 1,2,4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.train.blocks import BlockStack  # noqa: E402
from selfupdate.train.losses import HiddenLoss  # noqa: E402


def _dtype(name: str):
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _make_targets(n_layers: int, batch: int, answer_len: int, hidden: int,
                  device: torch.device, dtype: torch.dtype, on_cpu: bool) -> dict[int, torch.Tensor]:
    tgt_device = torch.device("cpu") if on_cpu else device
    return {
        L: torch.randn(batch, answer_len, hidden, device=tgt_device, dtype=dtype)
        for L in range(1, n_layers + 1)
    }


def _pp_device_map(model_name: str, split: int) -> dict:
    if torch.cuda.device_count() < 2:
        raise ValueError("--pipeline-split needs at least two visible CUDA devices")
    mc = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(mc, "text_config", mc)
    n = text_cfg.num_hidden_layers
    if not 0 < split < n:
        raise ValueError(f"--pipeline-split {split} outside 1..{n - 1}")
    tied = getattr(mc, "tie_word_embeddings", getattr(text_cfg, "tie_word_embeddings", False))
    vocab_dev = 0 if tied else 1
    prefix = "model.language_model" if getattr(mc, "model_type", "") == "gemma4" else "model"
    dm = {
        f"{prefix}.embed_tokens": 0,
        f"{prefix}.norm": vocab_dev,
        "lm_head": vocab_dev,
    }
    if prefix != "model":
        dm["model.vision_tower"] = 0
        dm["model.embed_vision"] = 0
    dm[f"{prefix}.rotary_emb"] = 0
    for i in range(n):
        dm[f"{prefix}.layers.{i}"] = 0 if i < split else 1
    return dm


def _load_model(args, dtype: torch.dtype):
    if args.pipeline_split > 0 and args.device_map_auto:
        raise ValueError("--pipeline-split and --device-map-auto are mutually exclusive")
    if args.pipeline_split > 0:
        dm = _pp_device_map(args.model, args.pipeline_split)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=dm)
        return model, f"pp2_split_{args.pipeline_split}", dm
    if args.device_map_auto:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
        return model, "auto", getattr(model, "hf_device_map", {})
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    return model, str(args.device), {"all": str(args.device)}


def _one_walk(stack: BlockStack, loss_fn: HiddenLoss, ids: torch.Tensor,
              pos: torch.Tensor, targets: dict[int, torch.Tensor], s0: int,
              answer_len: int, sync_each_block: bool, optimizer=None) -> float:
    stack.model.zero_grad(set_to_none=True)
    h = stack.embed(ids)
    pos_emb = stack.rope(h, pos)
    total_for_log = 0.0
    for L in range(1, stack.n_layers + 1):
        h = h.detach()
        with torch.autocast(ids.device.type, dtype=torch.bfloat16,
                            enabled=ids.device.type == "cuda"):
            h_out = stack.run_block(L, h, pos_emb)
            pred = stack.loss_view(L, h_out)[:, s0: s0 + answer_len]
            tgt = targets[L].to(pred.device, non_blocking=True)
            loss = loss_fn(
                pred.reshape(-1, pred.shape[-1]),
                tgt.reshape(-1, tgt.shape[-1]),
                normed=(L == stack.n_layers),
            )
        loss.backward()
        if sync_each_block:
            total_for_log += float(loss.detach().cpu())
        h = h_out
    if optimizer is not None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_for_log


def _bench_variant(stack: BlockStack, batch: int, seq_len: int, answer_len: int,
                   iters: int, warmup: int, cpu_targets: bool,
                   sync_each_block: bool, optimizer_enabled: bool,
                   dtype: torch.dtype, seed: int) -> dict:
    device = stack.embed_tokens.weight.device
    vocab, hidden = stack.embed_tokens.weight.shape
    torch.manual_seed(seed + batch + int(cpu_targets) * 100 + int(sync_each_block) * 200)
    ids = torch.randint(8, min(vocab, 30000), (batch, seq_len), device=device)
    pos = torch.arange(seq_len, device=device)[None].expand(batch, -1)
    s0 = seq_len - answer_len
    targets = _make_targets(stack.n_layers, batch, answer_len, hidden,
                            device, dtype, on_cpu=cpu_targets)
    loss_fn = HiddenLoss("nmse", stack.final_norm, stack.lm_head)
    optimizer = None
    if optimizer_enabled:
        params = [p for L in range(1, stack.n_layers + 1)
                  for p in stack.block_params(L) if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=0.0)

    for _ in range(warmup):
        _one_walk(stack, loss_fn, ids, pos, targets, s0, answer_len,
                  sync_each_block, optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _one_walk(stack, loss_fn, ids, pos, targets, s0, answer_len,
                  sync_each_block, optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    items = batch * iters
    toks = batch * seq_len * iters
    out = {
        "batch": batch,
        "variant": "sync_each_block" if sync_each_block else ("cpu_targets" if cpu_targets else "gpu_targets"),
        "iters": iters,
        "seconds": seconds,
        "item_ms": 1000.0 * seconds / max(items, 1),
        "items_per_s": items / max(seconds, 1e-9),
        "tokens_per_s": toks / max(seconds, 1e-9),
    }
    if device.type == "cuda":
        out["vram_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
        out["vram_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map-auto", action="store_true",
                    help="load with Hugging Face device_map='auto' instead of .to(--device)")
    ap.add_argument("--pipeline-split", type=int, default=0,
                    help="two-card PP2 split layer index; reserves cross-GPU communication in the benchmark")
    ap.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--answer-len", type=int, default=64)
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--variants", default="gpu,cpu,sync",
                    help="comma list: gpu,cpu,sync")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-optimizer", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/speed_check_latest.json"))
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    model, placement, device_map = _load_model(args, dtype)
    model.train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()

    results = []
    variant_specs = {
        "gpu": (False, False),
        "cpu": (True, False),
        "sync": (False, True),
    }
    requested_variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    for v in requested_variants:
        if v not in variant_specs:
            raise ValueError(f"unknown --variants entry {v!r}")

    for batch in [int(x) for x in args.batches.split(",") if x.strip()]:
        for variant_key in requested_variants:
            cpu_targets, sync_each_block = variant_specs[variant_key]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                row = _bench_variant(
                    stack, batch, args.seq_len, args.answer_len, args.iters,
                    args.warmup, cpu_targets, sync_each_block,
                    optimizer_enabled=not args.no_optimizer,
                    dtype=dtype, seed=args.seed,
                )
            except torch.cuda.OutOfMemoryError as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                row = {
                    "batch": batch,
                    "variant": variant_key,
                    "iters": args.iters,
                    "error": "cuda_oom",
                    "message": str(e).split("\n", 1)[0],
                }
                if device.type == "cuda":
                    row["vram_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
                    row["vram_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            except Exception as e:  # noqa: BLE001 - failure is a benchmark result
                row = {
                    "batch": batch,
                    "variant": variant_key,
                    "iters": args.iters,
                    "error": type(e).__name__,
                    "message": str(e).split("\n", 1)[0],
                }
            results.append(row)
            if "error" in row:
                print(
                    f"batch={row['batch']:>2} {row['variant']:>15} "
                    f"ERROR {row['error']} vram={row.get('vram_reserved_gb', 0):.2f} GB",
                    flush=True,
                )
            else:
                print(
                    f"batch={row['batch']:>2} {row['variant']:>15} "
                    f"{row['item_ms']:8.1f} ms/item "
                    f"{row['items_per_s']:6.2f} items/s "
                    f"{row['tokens_per_s']:8.0f} tok/s "
                    f"vram={row.get('vram_reserved_gb', 0):.2f} GB",
                    flush=True,
                )

    payload = {
        "model": args.model,
        "device": str(device),
        "placement": placement,
        "device_map": {str(k): str(v) for k, v in dict(device_map).items()},
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "answer_len": args.answer_len,
        "optimizer": not args.no_optimizer,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
