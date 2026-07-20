"""Measure teacher-context leakage in pipeline-v4's local objective.

This is a read-only diagnostic: it compares the input and block-local loss of
the production teacher-state/frozen-KV condition with a deployment-matched
flow-censored trajectory, while keeping the uncensored teacher h[L] target
fixed.  ``torch.autograd.grad`` measures adapter gradient norms without an
optimizer, parameter writes, or populating ``.grad``.

The probe intentionally supports only a single-process, fully resident,
dense transformer.  Stage-scoped, MoE, recurrent, compressed-attention, and
weight-rotation paths fail loudly instead of approximating their semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
import torch.nn.functional as F

from selfupdate.config import load_config
from selfupdate.data.dataset import DistillDataset
from selfupdate.train.losses import HiddenLoss
from selfupdate.train.moe import dequantize_overrides
from selfupdate.train.blocks import NO_PREPARED_ATTENTION_MASK
from selfupdate.train.online_v4 import (
    _FrozenKV,
    _V4Cohort,
    _online_teacher_capture,
    _student_ids,
)
from selfupdate.train.runtime import TrainingRuntime


def _git_provenance() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(run(
            "git", "status", "--porcelain", "--untracked-files=no")),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _parse_layers(text: str, n_layers: int) -> list[int]:
    try:
        layers = sorted(set(int(x) for x in text.split(",") if x.strip()))
    except ValueError as exc:
        raise SystemExit(f"--layers must be comma-separated integers: {exc}")
    if not layers or layers[0] < 1 or layers[-1] > n_layers:
        raise SystemExit(
            f"--layers must select blocks in 1..{n_layers}; got {layers}")
    return layers


def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    sq = torch.zeros((), dtype=torch.float64, device=loss.device)
    for grad in grads:
        if grad is not None:
            sq += grad.detach().double().pow(2).sum()
    return math.sqrt(float(sq.item()))


def _local_loss(loss_fn, stack, layer, output, target, valid, anchor):
    view = stack.loss_view(layer, output)
    flat_view = view[valid]
    flat_target = target.to(view.dtype)[valid]
    flat_anchor = anchor[valid] if loss_fn.kind == "delta_cosine" else None
    return loss_fn(
        flat_view, flat_target, normed=(layer == stack.n_layers),
        layer=layer, aligned_input=flat_anchor,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment")
    ap.add_argument("--layers", required=True,
                    help="comma-separated 1-based block numbers")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    cfg = load_config(args.config, args.experiment)
    if cfg.train.pipeline_version != 4 or cfg.mask.compaction != "flow_mask":
        raise SystemExit("probe requires pipeline-v4 with mask.compaction=flow_mask")
    if cfg.train.v4_stage_scoped or cfg.train.v4_stage >= 0:
        raise SystemExit("probe does not support staged/stage-scoped loading")
    if cfg.train.v4_weight_residency == "rotate":
        raise SystemExit("probe does not support rotating block weights")
    cfg.model.device = args.device

    rt = TrainingRuntime(cfg).load(
        dequantize_overrides(cfg.model.name, cfg.train.moe_mode))
    tok, stack = rt.tokenizer, rt.stack
    cache = rt.load_cache()
    layers = _parse_layers(args.layers, stack.n_layers)

    num_experts = int(
        getattr(stack.text_config, "num_experts", 0)
        or getattr(stack.text_config, "num_local_experts", 0) or 0)
    if num_experts:
        raise SystemExit(f"probe does not support MoE models (num_experts={num_experts})")
    unsupported = sorted({
        (stack.layer_types[layer - 1] if layer - 1 < len(stack.layer_types)
         else "full_attention")
        for layer in range(1, stack.n_layers + 1)
        if (stack.layer_types[layer - 1] if layer - 1 < len(stack.layer_types)
            else "full_attention") in {
                "linear_attention", "compressed_attention"
            }
    })
    if getattr(stack, "needs_deepseek_masks", False) or unsupported:
        raise SystemExit(
            f"probe does not support recurrent/compressed attention: {unsupported}")

    peft_model = rt.peft_model
    adapters_off = peft_model.disable_adapter if peft_model is not None else None
    if adapters_off is None:
        raise SystemExit("probe requires an attached LoRA adapter")
    params = {
        layer: [p for p in stack.block_params(layer) if p.requires_grad]
        for layer in layers
    }
    if any(not ps for ps in params.values()):
        raise SystemExit("every selected block must have trainable adapter parameters")

    ds = DistillDataset(
        cfg.data.examples_path, cache, tok, need_layers=[],
        with_teacher_ids=False, pad_random=False,
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        item_cache_items=cfg.cache.item_cache_items,
    )
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(ds)), min(args.limit, len(ds))))
    device = torch.device(args.device)
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    accum = {layer: {
        "cells": 0, "input_diff_sq": 0.0, "teacher_input_sq": 0.0,
        "input_cosine_sum": 0.0, "current_loss_sum": 0.0,
        "censored_loss_sum": 0.0, "current_grad_norms": [],
        "censored_grad_norms": [],
    } for layer in layers}
    started = time.perf_counter()

    for ordinal, index in enumerate(indices, 1):
        cohort = _V4Cohort(cfg, ds, [index], device)
        capture = _online_teacher_capture(
            cfg, stack, adapters_off, cohort, layers, device, stack.n_layers)
        ids = _student_ids(ds, cohort).to(device)
        pos = torch.arange(cohort.T, device=device)[None]
        keep = cohort.keep.to(device)
        with torch.no_grad():
            censored_h = stack.embed(ids)

        for layer in range(1, max(layers) + 1):
            if layer not in layers:
                with torch.no_grad():
                    censored_h = stack.run_block(
                        layer, censored_h, stack.rope(censored_h, pos),
                        position_ids=pos, flow_keep=keep,
                        causal_length=cohort.T)
                continue

            teacher_full = capture["inputs"][layer].to(device)
            target = capture["targets"][layer].to(device)
            teacher_q = cohort.gather_query_inputs(teacher_full).detach()
            censored_q = cohort.gather_query_inputs(censored_h).detach()
            valid = cohort.loss_valid_dev
            tq, cq = teacher_q[valid].float(), censored_q[valid].float()
            cells = int(valid.sum())
            row = accum[layer]
            row["cells"] += cells
            row["input_diff_sq"] += float((tq - cq).double().pow(2).sum().item())
            row["teacher_input_sq"] += float(tq.double().pow(2).sum().item())
            row["input_cosine_sum"] += float(
                F.cosine_similarity(tq, cq, dim=-1).double().sum().item())

            frozen = _FrozenKV()
            with torch.no_grad(), adapters_off():
                stack.run_block(
                    layer, teacher_full, stack.rope(teacher_full, pos),
                    position_ids=pos, past_key_values=frozen, use_cache=True,
                    prepared_attention_mask=NO_PREPARED_ATTENTION_MASK)
            frozen.recording = False
            layer_type = (
                stack.layer_types[layer - 1]
                if layer - 1 < len(stack.layer_types) else "full_attention")
            window = None
            if layer_type in ("sliding_attention", "chunked_attention"):
                window = (getattr(stack.text_config, "sliding_window", None)
                          or getattr(stack.text_config,
                                     "attention_chunk_size", None))
            current_out = stack.run_block(
                layer, teacher_q, stack.rope(teacher_q, cohort.qpos_dev),
                position_ids=cohort.qpos_dev, past_key_values=frozen,
                use_cache=False,
                prepared_attention_mask=cohort.additive_mask(
                    teacher_q.dtype, window=window))
            current_loss = _local_loss(
                loss_fn, stack, layer, current_out, target, valid, teacher_q)
            current_grad = _grad_norm(current_loss, params[layer])

            censored_out = stack.run_block(
                layer, censored_h, stack.rope(censored_h, pos),
                position_ids=pos, flow_keep=keep, causal_length=cohort.T)
            censored_rows = cohort.gather_query_inputs(censored_out)
            censored_loss = _local_loss(
                loss_fn, stack, layer, censored_rows, target, valid, censored_q)
            censored_grad = _grad_norm(censored_loss, params[layer])
            row["current_loss_sum"] += float(current_loss.detach()) * cells
            row["censored_loss_sum"] += float(censored_loss.detach()) * cells
            row["current_grad_norms"].append(current_grad)
            row["censored_grad_norms"].append(censored_grad)
            censored_h = censored_out.detach()
            del teacher_full, target, frozen, current_out, censored_out

        print(f"[{ordinal}/{len(indices)}] {cohort.example_ids[0]}", flush=True)

    results = []
    for layer in layers:
        row = accum[layer]
        cells = row["cells"]
        results.append({
            "layer": layer,
            "loss_cells": cells,
            "input_relative_l2": math.sqrt(
                row["input_diff_sq"] / max(row["teacher_input_sq"], 1e-30)),
            "input_cosine": row["input_cosine_sum"] / max(cells, 1),
            "current_teacher_local_loss": row["current_loss_sum"] / max(cells, 1),
            "fully_censored_local_loss": row["censored_loss_sum"] / max(cells, 1),
            "current_teacher_local_grad_norm_mean": sum(row["current_grad_norms"]) / len(indices),
            "fully_censored_grad_norm_mean": sum(row["censored_grad_norms"]) / len(indices),
            "gradient_aggregation": "mean_of_per_example_parameter_l2_norms",
        })

    output = {
        "schema": "v4_context_leak_probe_v1",
        "provenance": {
            **_git_provenance(), "config": args.config,
            "experiment": args.experiment, "model": cfg.model.name,
            "dataset": cfg.data.examples_path, "seed": args.seed,
            "selected_indices": indices, "selected_example_ids": [
                ds.pairs[i].example_id for i in indices],
        },
        "conditions": {
            "current_teacher_local": (
                "detached uncensored teacher h[L-1] query and frozen teacher "
                "KV; direct privileged key columns flow-masked"),
            "fully_censored": (
                "deployment-matched trajectory from embeddings through L, "
                "privileged rows/keys flow-masked at every block"),
            "target": "same uncensored adapters-off teacher h[L]",
            "parameter_updates": False,
        },
        "hidden_loss": cfg.train.hidden_loss,
        "items": len(indices), "layers": layers,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
