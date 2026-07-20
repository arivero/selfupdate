"""Measure GPU-to-CPU egress for one cache token across every transformer layer.

No model weights are loaded.  A synthetic bf16 ``[layers, tokens, hidden]``
buffer has exactly the byte layout of the teacher hidden-state cache.  The
three paths distinguish the hardware bulk-copy limit from the cache writer's
current one-transfer-per-layer implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig


def elapsed(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def row(name: str, seconds: float, total_bytes: int, tokens: int) -> dict:
    return {
        "path": name,
        "seconds_per_copy": seconds,
        "microseconds_per_full_token": seconds * 1e6 / tokens,
        "effective_GiB_per_second": total_bytes / seconds / 2**30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", nargs="+", type=int, default=[1, 64, 512])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    layers, hidden = int(cfg.num_hidden_layers), int(cfg.hidden_size)
    element_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    bytes_per_token = layers * hidden * element_bytes
    results: list[dict] = []

    for tokens in args.tokens:
        shape = (layers, tokens, hidden)
        src = torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_()
        total_bytes = src.numel() * src.element_size()
        # Preallocation makes this the steady-state DMA lower bound.
        bulk_dst = torch.empty_like(src, device="cpu", pin_memory=True)
        layer_dst = [torch.empty((tokens, hidden), device="cpu", dtype=torch.bfloat16,
                                 pin_memory=True) for _ in range(layers)]
        # Warm every path before timing.
        bulk_dst.copy_(src, non_blocking=True)
        for layer, dst in enumerate(layer_dst):
            dst.copy_(src[layer], non_blocking=True)
        _ = [src[layer].contiguous().cpu() for layer in range(layers)]
        torch.cuda.synchronize()

        group = {"tokens": tokens, "bytes": total_bytes,
                 "KiB_per_full_token": bytes_per_token / 1024,
                 "measurements": []}
        group["measurements"].append(row(
            "bulk_to_preallocated_pinned_cpu",
            elapsed(lambda: bulk_dst.copy_(src, non_blocking=True), args.repeats),
            total_bytes, tokens))
        group["measurements"].append(row(
            "one_copy_per_layer_to_preallocated_pinned_cpu",
            elapsed(lambda: [dst.copy_(src[layer], non_blocking=True)
                             for layer, dst in enumerate(layer_dst)], args.repeats),
            total_bytes, tokens))
        group["measurements"].append(row(
            "current_like_per_layer_contiguous_cpu",
            elapsed(lambda: [src[layer].contiguous().cpu() for layer in range(layers)],
                    args.repeats), total_bytes, tokens))
        results.append(group)
        del src, bulk_dst, layer_dst

    output = {
        "model": args.model,
        "layers": layers,
        "hidden_size": hidden,
        "dtype": "bfloat16",
        "bytes_per_full_token": bytes_per_token,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
