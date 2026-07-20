#!/usr/bin/env python
"""Shard-streaming fp8/mxfp4 -> bf16 snapshot dequantizer (task #16).

The first version loaded the whole model through transformers and died in
`core_model_loading` ("Undefined Operation encountered!") after a 200 GB
RAM climb. This version never builds a model: it streams the checkpoint
shard by shard, dequantizes each weight against its sibling `.scale`
tensor, and writes a plain bf16 snapshot the ordinary (and stage-scoped)
loaders can mmap. Peak RAM = one output shard.

Measured layout (DeepSeek-V4-Flash, 2026-07-18):
- fp8 path:  `X.weight` F8_E4M3 [out, in] + `X.scale` F8_E8M0
  [ceil(out/128), ceil(in/128)]  (block-quant 128x128, ue8m0 = 2**(b-127))
- mxfp4 path: `X.weight` I8 [out, in/2] (two e2m1 nibbles per byte, low
  nibble first) + `X.scale` F8_E8M0 [out, in/32] (group-32 scales)
- everything else (norms, sinks, embeddings, biases) copies through.

Usage (CPU only, ~minutes of math + Lustre write time; run detached):
    python compressed/dequantize_snapshot.py deepseek-ai/DeepSeek-V4-Flash \
        --out /fs/.../snapshots/deepseek-v4-flash-bf16

The output dir is a local model path usable as model.name. The Qwen
397B-FP8 takes the same lane.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# fp4 e2m1: sign x [0, 0.5, 1, 1.5, 2, 3, 4, 6]
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)

SHARD_BYTES = 5 << 30


def _e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, scale.view(torch.uint8).to(torch.float32) - 127.0)


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor,
                       block: int = 128) -> torch.Tensor:
    """DeepSeek form: F8_E4M3 weight + F8_E8M0 block scale (power-of-two)."""
    out_dim, in_dim = weight.shape
    s = _e8m0_to_float(scale)
    s = s.repeat_interleave(block, 0)[:out_dim] \
         .repeat_interleave(block, 1)[:, :in_dim]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _dequant_fp8_inv(weight: torch.Tensor, scale_inv: torch.Tensor,
                     block: int = 128) -> torch.Tensor:
    """Qwen/DeepSeek-V3 form: F8_E4M3 weight + BF16 `weight_scale_inv`
    per-[128,128]-block multiplier (already the value to multiply, not a
    power-of-two exponent). Layout [ceil(out/128), ceil(in/128)]."""
    out_dim, in_dim = weight.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(block, 0)[:out_dim] \
         .repeat_interleave(block, 1)[:, :in_dim]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor,
                   group: int = 32) -> torch.Tensor:
    rows, packed = weight.shape
    b = weight.view(torch.uint8)
    lo = (b & 0x0F).long()
    hi = (b >> 4).long()
    vals = torch.empty((rows, packed * 2), dtype=torch.float32)
    vals[:, 0::2] = _E2M1[lo]
    vals[:, 1::2] = _E2M1[hi]
    s = _e8m0_to_float(scale).repeat_interleave(group, 1)[:, : packed * 2]
    return (vals * s).to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    snap = Path(args.model) if Path(args.model).is_dir() else Path(
        snapshot_download(args.model, local_files_only=True,
                          allow_patterns=["*.safetensors", "*.json",
                                          "*.jinja", "*.txt", "*.model"]))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    index = json.loads((snap / "model.safetensors.index.json").read_text())
    weight_map: dict = index["weight_map"]
    scales = {k for k in weight_map
              if k.endswith(".scale") or k.endswith(".weight_scale_inv")}

    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    handles: dict[str, object] = {}

    def read(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(str(snap / shard), framework="pt")
        return handles[shard].get_tensor(key)

    new_map: dict[str, str] = {}
    buffer: dict[str, torch.Tensor] = {}
    buffered = 0
    shard_no = 0
    total = 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_no
        if not buffer:
            return
        shard_no += 1
        name = f"model-{shard_no:05d}.safetensors"
        save_file(buffer, str(out / name), metadata={"format": "pt"})
        for key in buffer:
            new_map[key] = name
        print(f"wrote {name}: {len(buffer)} tensors, "
              f"{buffered / 2**30:.2f} GiB", flush=True)
        buffer, buffered = {}, 0

    for shard in sorted(by_shard):
        for key in sorted(by_shard[shard]):
            if key in scales:
                continue
            t = read(key)
            skey_dot = (key[: -len(".weight")] + ".scale"
                        if key.endswith(".weight") else None)
            skey_inv = (key + "_scale_inv"
                        if key.endswith(".weight") else None)
            if skey_inv in scales:  # Qwen/V3 fp8: BF16 per-block inv-scale
                t = _dequant_fp8_inv(t, read(skey_inv))
            elif skey_dot in scales:  # DeepSeek: E8M0 or mxfp4 scale
                s = read(skey_dot)
                if t.dtype == torch.int8:
                    t = _dequant_mxfp4(t, s)
                elif "float8" in str(t.dtype):
                    t = _dequant_fp8_block(t, s)
                else:
                    raise SystemExit(
                        f"{key}: has a scale but dtype {t.dtype} is not a "
                        "known quant storage")
            buffer[key] = t.contiguous()
            nbytes = t.numel() * t.element_size()
            buffered += nbytes
            total += nbytes
            if buffered >= SHARD_BYTES:
                flush()
        if len(handles) > 4:
            handles.clear()  # bound the open mmaps
    flush()

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": new_map}, indent=1))
    cfg = json.loads((snap / "config.json").read_text())
    cfg.pop("quantization_config", None)
    if cfg.get("expert_dtype"):
        cfg["expert_dtype"] = "bf16"
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for extra in snap.iterdir():
        if (extra.suffix in (".json", ".jinja", ".txt", ".model")
                and extra.name not in ("config.json",
                                       "model.safetensors.index.json")):
            shutil.copy2(extra, out / extra.name)
    print(f"DONE: {total / 2**30:.1f} GiB bf16 -> {out}", flush=True)


if __name__ == "__main__":
    main()
