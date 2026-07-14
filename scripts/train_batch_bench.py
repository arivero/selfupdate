"""Benchmark the real summed trainer path without saving a checkpoint.

This is the speed-cert counterpart to ``scripts/speed_check.py``: it uses the
actual examples.jsonl + teacher cache and runs the same item or padded summed
step that training uses, but stops after a small number of optimizer steps and
writes timing/memory JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import (
    Batch,
    collate_padded_items,
    iter_batch_grid_tiles,
)
from selfupdate.teacher.cache import TeacherCache, resolve_cache_dir
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import (
    _extend_pending_from_batch,
    _loader,
    _make_dataset,
    _summed_batch,
)
from selfupdate.train.runtime import (
    OptimizerPlan,
    pp_device_map as _pp_device_map,
    uses_pipeline_map as _uses_pipeline_map,
)
from selfupdate.train.teacher_source import OnlineTeacherSource
from selfupdate.train.telemetry import _flush_train_log
from selfupdate.train.validate import validate_knob_schedule as _validate_knob_schedule
from selfupdate.train.losses import HiddenLoss


class NullLog:
    def log(self, **kwargs):
        pass


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def bench(args) -> dict:
    cfg = load_config(args.config, args.experiment)
    cfg.model.device = args.device
    if args.batching:
        cfg.train.batching = args.batching
    if args.micro_batch:
        cfg.train.micro_batch = args.micro_batch
        if not args.grad_accum:
            cfg.train.grad_accum = args.micro_batch
    if args.grad_accum:
        cfg.train.grad_accum = args.grad_accum
    cfg.train.max_steps = args.steps
    cfg.eval.every_epochs = 999999
    _validate_knob_schedule(cfg)
    if cfg.train.schedule != "summed":
        raise ValueError("train_batch_bench currently benchmarks summed only")

    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    full_ft_all_blocks = (not cfg.train.lora.enabled
                          and cfg.train.schedule != "sequential")
    base_dtype = torch.float32 if full_ft_all_blocks else torch.bfloat16
    t_load0 = time.perf_counter()
    pp_map = _pp_device_map(cfg) if _uses_pipeline_map(cfg) else None
    if pp_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=base_dtype, device_map=pp_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=base_dtype)
        model.to(args.device)
    t_load1 = time.perf_counter()
    peft_model = None
    if cfg.train.lora.enabled:
        from selfupdate.train.lora import attach_lora

        peft_model = attach_lora(model, cfg.train.lora)
        model = peft_model.get_base_model()
    model.train()

    stack = BlockStack(model, hook_free_walk=pp_map is not None)
    stack.freeze_non_blocks()
    if cfg.train.online_teacher and peft_model is None:
        raise ValueError("train.online_teacher requires train.lora.enabled")
    teacher = None
    if cfg.train.online_teacher:
        teacher = OnlineTeacherSource(stack, peft_model=peft_model)
    elif cfg.train.frozen_teacher_copy:
        # Mirror TrainingRuntime.load_teacher: a resident frozen bf16 copy
        # computes targets on the fly (the loss-grid arms' configuration).
        # Without this branch the bench silently fell to the disk-cache path
        # and measured a different workload than the arms it advises on.
        t_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16)
        t_model.to(args.device)
        t_model.eval().requires_grad_(False)
        teacher = OnlineTeacherSource(stack, frozen_stack=BlockStack(t_model))
    cache = None
    if teacher is None:
        cache_root, chash = resolve_cache_dir(cfg)
        cache = TeacherCache(cache_root, expect_hash=chash)
    n = stack.n_layers
    ds = _make_dataset(
        cfg, cache, tok,
        [] if teacher is not None else list(range(1, n + 1)),
        with_teacher_ids=teacher is not None,
    )
    loader = _loader(cfg, ds)
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head)
    plan = OptimizerPlan.build(stack, cfg)

    steps = []
    accum = step = 0
    next_step = cfg.train.grad_accum
    pending_losses = []
    for items in loader:
        if cfg.train.update_granularity == "grid":
            if not isinstance(items, Batch):
                raise ValueError("grid benchmark requires padded/bucketed batches")
            for tile in iter_batch_grid_tiles(
                    items, cfg.train.tokens_per_answer_update):
                batch = tile.batch
                _sync(args.device)
                t0 = time.perf_counter()
                layer_losses = _summed_batch(
                    cfg, stack, loss_fn, batch, batch.hidden, args.device)
                _extend_pending_from_batch(pending_losses, layer_losses)
                plan.step()
                _sync(args.device)
                elapsed = time.perf_counter() - t0
                token_count = int(batch.lengths.sum())
                aligned_count = tile.aligned_token_count
                steps.append({
                    "step": step,
                    "answer_visits": tile.answer_count,
                    "completed_answers": tile.completed_answer_count,
                    "tokens": token_count,
                    "aligned_tokens": aligned_count,
                    "layer_loss_cells": aligned_count * n,
                    "max_seq": int(batch.lengths.max()),
                    "pad_tokens": int(batch.student_ids.numel() - batch.lengths.sum()),
                    "seconds": elapsed,
                    "tiles_per_second": 1.0 / elapsed,
                    "answer_visits_per_second": tile.answer_count / elapsed,
                    "aligned_tokens_per_second": aligned_count / elapsed,
                    "grid_coordinates": tile.coordinate_ranges,
                })
                _flush_train_log(
                    NullLog(), epoch=0, step=step,
                    accum=sum(x["completed_answers"] for x in steps),
                    pending=pending_losses, n_layers=n,
                    update_granularity="grid",
                    update_reduction=cfg.train.update_reduction,
                )
                step += 1
                if step >= args.steps:
                    break
            if step >= args.steps:
                break
            continue
        _sync(args.device)
        t0 = time.perf_counter()
        if isinstance(items, Batch):
            targets = (teacher.aligned_targets_batch(items, args.device)
                       if teacher is not None else items.hidden)
            layer_losses = _summed_batch(cfg, stack, loss_fn, items, targets, args.device)
            accum += len(items.example_ids)
            item_count = len(items.example_ids)
            token_count = int(items.lengths.sum())
            aligned_count = int(items.A.sum())
            max_seq = int(items.lengths.max())
            pad_tokens = int(items.student_ids.numel() - items.lengths.sum())
            _extend_pending_from_batch(pending_losses, layer_losses)
        else:
            item_count = len(items)
            token_count = 0
            aligned_count = 0
            max_seq = 0
            pad_tokens = 0
            for it in items:
                b1 = collate_padded_items([it])
                targets = (teacher.aligned_targets_batch(b1, args.device)
                           if teacher is not None else b1.hidden)
                layer_losses = _summed_batch(cfg, stack, loss_fn, b1, targets,
                                             args.device)
                _extend_pending_from_batch(pending_losses, layer_losses)
                accum += 1
                token_count += len(it.student_ids)
                aligned_count += it.A
                max_seq = max(max_seq, len(it.student_ids))
        if accum >= next_step:
            plan.step()
            _sync(args.device)
            elapsed = time.perf_counter() - t0
            _flush_train_log(
                NullLog(), epoch=0, step=step, accum=accum,
                pending=pending_losses, n_layers=n,
            )
            steps.append({
                "step": step,
                "items": item_count,
                "tokens": token_count,
                "aligned_tokens": aligned_count,
                "max_seq": max_seq,
                "pad_tokens": pad_tokens,
                "seconds": elapsed,
                "items_per_second": item_count / elapsed,
                "tokens_per_second": token_count / elapsed,
                "aligned_tokens_per_second": aligned_count / elapsed,
            })
            step += 1
            next_step += cfg.train.grad_accum
            if step >= args.steps:
                break

    steady = steps[1:] if len(steps) > 1 else steps
    steady_seconds = sum(x["seconds"] for x in steady)
    steady_items = sum(x.get("items", x.get("answer_visits", 0))
                       for x in steady)
    steady_aligned = sum(x["aligned_tokens"] for x in steady)
    total_aligned = sum(pair.aligned_len for pair in ds.pairs)
    steady_tile_seconds = [x["seconds"] for x in steady]
    out = {
        "model": cfg.model.name,
        "runtime": {
            "python": platform.python_version(),
            "libc": list(platform.libc_ver()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "causal_conv_backend": os.environ.get(
                "SELFUPDATE_CAUSAL_CONV_BACKEND", "unspecified"),
            "causal_conv1d_importable": (
                importlib.util.find_spec("causal_conv1d") is not None),
        },
        "schedule": cfg.train.schedule,
        "batching": cfg.train.batching,
        "micro_batch": cfg.train.micro_batch,
        "grad_accum": cfg.train.grad_accum,
        "hidden_loss": cfg.train.hidden_loss,
        "pipeline_version": cfg.train.pipeline_version,
        "update_granularity": cfg.train.update_granularity,
        "answers_per_update": cfg.train.answers_per_update,
        "tokens_per_answer_update": cfg.train.tokens_per_answer_update,
        "update_reduction": cfg.train.update_reduction,
        "conn_window": cfg.train.conn_window,
        "final_logit_training": False,
        "load_seconds": t_load1 - t_load0,
        "steps": steps,
        "steady_state": {
            "warmup_steps_excluded": max(len(steps) - len(steady), 0),
            "items_per_second": steady_items / steady_seconds,
            "answer_visits_per_second": steady_items / steady_seconds,
            "aligned_tokens_per_second": steady_aligned / steady_seconds,
            "mean_tile_seconds": statistics.mean(steady_tile_seconds),
            "median_tile_seconds": statistics.median(steady_tile_seconds),
            "projected_epoch_examples": len(ds),
            "projected_epoch_aligned_tokens": total_aligned,
            "projected_epoch_seconds": (
                total_aligned * steady_seconds / steady_aligned
                if cfg.train.update_granularity == "grid"
                else len(ds) * steady_seconds / steady_items),
        },
    }
    if torch.device(args.device).type == "cuda":
        n_dev = torch.cuda.device_count()
        out["vram_reserved_gb"] = round(
            sum(torch.cuda.max_memory_reserved(d) for d in range(n_dev)) / 2**30, 3
        )
        out["vram_allocated_gb"] = round(
            sum(torch.cuda.max_memory_allocated(d) for d in range(n_dev)) / 2**30, 3
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batching", choices=["item", "padded", "bucketed"],
                    default=None)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = bench(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
