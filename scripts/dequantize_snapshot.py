#!/usr/bin/env python
"""One-time offline dequantization of a quantized HF snapshot to bf16 (B8).

The stage-scoped loader (shard_load.py) and the frozen-teacher math want
plain bf16 tensors; DeepSeek-V4-Flash ships fp8 e4m3 (128x128 block scales)
with fp4 experts, and HF's fp8 quantizer is is_trainable=False — LoRA on an
fp8-resident base is outside its contract. This tool loads the checkpoint
ONCE on CPU with the quantizer's own dequantize path and saves a plain bf16
snapshot; every training lane then treats the model like any other.

Usage (hours of CPU + needs ~2x model RAM; run detached):
    python scripts/dequantize_snapshot.py deepseek-ai/DeepSeek-V4-Flash \
        --out /fs/.../snapshots/deepseek-v4-flash-bf16

The output dir is a local model path usable as model.name. The Qwen
397B-FP8 takes the same lane if shard_load's on-the-fly dequant is not
preferred. Verify a few tensors against the quantized load before training
(the script prints a checksum sample).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-size", default="4GB")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    acfg = AutoConfig.from_pretrained(args.model)
    qc = getattr(acfg, "quantization_config", None)
    kw = {}
    if qc is not None:
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(
            qc, "quant_method", None)
        if method == "fp8":
            from transformers import FineGrainedFP8Config

            kw["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        elif method == "mxfp4":
            from transformers import Mxfp4Config

            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        else:
            raise SystemExit(
                f"unhandled quant_method {method!r}; add its dequantize "
                "config here before trusting the output")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", **kw)
    sample = []
    with torch.no_grad():
        for name, p in list(model.named_parameters())[:3]:
            sample.append((name, float(p.float().abs().mean())))
    out = Path(args.out)
    model.save_pretrained(out, max_shard_size=args.max_shard_size,
                          safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    print(f"bf16 snapshot at {out}")
    for name, m in sample:
        print(f"  checksum |{name}|.abs().mean() = {m:.6e}")


if __name__ == "__main__":
    main()
