"""Memory-budget planner: measure block footprints WITHOUT loading the model,
then recommend micro-batch, window size, optimizer placement, and pipeline
partition points for a per-device VRAM budget.

The model is instantiated on the meta device (no weights, no VRAM); exact
per-module parameter counts come from there. ONE decoder block is then
materialized on the GPU to MEASURE the activation-graph bytes of a forward
+backward at each (micro_batch, seqlen) point — blocks are homogeneous, so
one block × window width predicts the connected-window graph. This is the
"plan before you load" tool for the hardware ladder: at 120B the plan must
exist before any weights move.

ADVISORY ONLY: it prints/writes recommendations and never edits configs —
config defaults are experiment variables (CLAUDE.md), a planner that silently
set knobs would fork arms the way the PP2 incident did.

Usage:
  python scripts/memory_plan.py --model Qwen/Qwen3-0.6B --seqlen 600 \
      --micro-batch 1 2 4 --window 1 4 8 --budget-gb 44 --devices 1 2 [--lora]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

GB = 2**30


def meta_model(name: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    mc = AutoConfig.from_pretrained(name)
    with torch.device("meta"):
        try:
            m = AutoModelForCausalLM.from_config(mc)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            m = AutoModelForImageTextToText.from_config(mc)
    return m, mc


def module_param_counts(model) -> dict:
    """Parameter counts of the pieces the trainer places: per decoder block,
    and the frozen vocabulary stack (embed + final norm + head; tied
    embeddings counted once)."""
    inner = model.model
    if not all(hasattr(inner, a) for a in ("embed_tokens", "layers", "norm")):
        inner = inner.language_model
    blocks = [sum(p.numel() for p in b.parameters()) for b in inner.layers]
    seen = set()
    vocab = 0
    for m in (inner.embed_tokens, inner.norm, model.lm_head):
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            vocab += p.numel()
    rotary = sum(p.numel() for p in getattr(inner, "rotary_emb", torch.nn.Identity()).parameters())
    return {"blocks": blocks, "vocab": vocab, "rotary": rotary,
            "hidden_size": getattr(inner.config, "hidden_size",
                                   model.config.hidden_size)}


def materialize_block(model, mc, device: str):
    """Move ONE decoder block (plus rotary) from meta to the GPU with sane
    random weights — values only need to be finite; the measurement is
    allocator behavior, which is shape-driven."""
    inner = model.model
    if not all(hasattr(inner, a) for a in ("embed_tokens", "layers", "norm")):
        inner = inner.language_model
    block = inner.layers[0].to_empty(device=device)
    with torch.no_grad():
        for p in block.parameters():
            p.normal_(0.0, 0.02)
        for b in block.buffers():
            if b.dtype.is_floating_point:
                b.normal_(0.0, 0.02)
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is not None:
        rotary = rotary.to_empty(device=device)
        # inv_freq buffers came back empty-garbage; recompute the standard
        # rope schedule so the forward stays finite
        if hasattr(rotary, "inv_freq"):
            dim = rotary.inv_freq.shape[0] * 2
            base = getattr(getattr(mc, "text_config", mc), "rope_theta", 1e6)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            rotary.inv_freq.copy_(inv)
    return block, rotary


def measure_block(block, rotary, hidden_size: int, B: int, T: int,
                  device: str, train_dtype: torch.dtype) -> dict:
    """Activation-graph + workspace bytes of one block fwd+bwd at (B, T)."""
    block = block.to(train_dtype)
    for p in block.parameters():
        p.requires_grad_(True)
        p.grad = None
    h = torch.randn(B, T, hidden_size, device=device, dtype=train_dtype,
                    requires_grad=True)
    pos = torch.arange(T, device=device)[None].expand(B, -1)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast(device, dtype=torch.bfloat16):
        pe = rotary(h, pos) if rotary is not None else None
        out = block(h, attention_mask=None, position_ids=pos,
                    position_embeddings=pe, use_cache=False)
        out = out[0] if isinstance(out, tuple) else out
    graph = torch.cuda.memory_allocated() - base  # retained fwd graph
    out.float().sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    for p in block.parameters():
        p.grad = None
    del h, out, pe
    torch.cuda.empty_cache()
    return {"graph_bytes": int(graph), "peak_bytes": int(peak)}


def optimizer_bytes_per_param(policy: str) -> int:
    """GPU-resident optimizer+master cost per TRAINED parameter, on top of
    the bf16/fp32 weight itself. AdamW: two fp32 moments (8B). full_resident
    trains fp32 masters (weights already counted at 4B) + fp32 grads (4B).
    full_offload keeps moments on pinned CPU: only grads stay."""
    return {"lora_fused": 12, "full_resident": 12, "full_offload": 4}[policy]


def plan(counts: dict, act: dict, args) -> list[dict]:
    """Enumerate feasible (devices, micro_batch, window, policy) cells and
    balance pipeline splits by per-device bytes."""
    n = len(counts["blocks"])
    weight_b = 4 if not args.lora else 2  # fp32 masters full-FT, bf16 LoRA base
    rows = []
    policies = (["lora_fused"] if args.lora
                else ["full_resident", "full_offload"])
    for ndev in args.devices:
        for B in args.micro_batch:
            for W in args.window:
                a = act[(B, args.seqlen)]
                for pol in policies:
                    per_block = counts["blocks"][0]
                    trained = per_block if not args.lora else per_block * args.lora_fraction
                    block_bytes = (per_block * weight_b
                                   + trained * optimizer_bytes_per_param(pol))
                    vocab_bytes = counts["vocab"] * (2 if args.lora else 4)
                    blocks_per_dev = -(-n // ndev)
                    dev_bytes = (blocks_per_dev * block_bytes + vocab_bytes
                                 + W * a["graph_bytes"]
                                 + (a["peak_bytes"] - a["graph_bytes"]))
                    # teacher targets resident for the active window only
                    tgt = (W + 1) * B * args.seqlen * counts["hidden_size"] * 2
                    total = dev_bytes + tgt
                    fits = total <= args.budget_gb * GB
                    splits = [round(i * n / ndev) for i in range(1, ndev)]
                    rows.append({
                        "devices": ndev, "micro_batch": B, "window": W,
                        "policy": pol,
                        "per_device_gb": round(total / GB, 2),
                        "fits": fits,
                        "pipeline_splits": splits,
                    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seqlen", type=int, default=600)
    ap.add_argument("--micro-batch", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--window", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--budget-gb", type=float, default=44.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-fraction", type=float, default=0.01,
                    help="trained fraction of block params under LoRA")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, mc = meta_model(args.model)
    counts = module_param_counts(model)
    n = len(counts["blocks"])
    print(f"{args.model}: {n} blocks x {counts['blocks'][0]/1e6:.1f}M params, "
          f"vocab stack {counts['vocab']/1e6:.1f}M, H={counts['hidden_size']}")

    train_dtype = torch.bfloat16 if args.lora else torch.float32
    block, rotary = materialize_block(model, mc, args.device)
    act = {}
    for B in sorted(set(args.micro_batch)):
        m = measure_block(block, rotary, counts["hidden_size"], B,
                          args.seqlen, args.device, train_dtype)
        act[(B, args.seqlen)] = m
        print(f"  measured B={B} T={args.seqlen}: graph "
              f"{m['graph_bytes']/GB:.3f} GB, fwd+bwd peak {m['peak_bytes']/GB:.3f} GB")

    rows = plan(counts, act, args)
    print(f"\nbudget {args.budget_gb} GB/device — feasible cells:")
    print("NOTE: excludes loss-head workspace (vocab-metric losses "
          "materialize [A, vocab] fp32 logits per layer eval) — measured "
          "+25% on 1.7B slide8 vocab_mse arms; plan with that margin.")
    print(f"{'dev':>3} {'B':>3} {'W':>3} {'policy':>14} {'GB/dev':>8} "
          f"{'fits':>5}  splits")
    for r in rows:
        print(f"{r['devices']:>3} {r['micro_batch']:>3} {r['window']:>3} "
              f"{r['policy']:>14} {r['per_device_gb']:>8.2f} "
              f"{str(r['fits']):>5}  {r['pipeline_splits']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "counts": {k: v for k, v in counts.items()},
             "seqlen": args.seqlen,
             "measured": {f"B{b}": m for (b, _), m in act.items()},
             "plan": rows}, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
