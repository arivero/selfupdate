"""Benchmark the real summed trainer path without saving a checkpoint.

This is the speed-cert counterpart to ``scripts/speed_check.py``: it uses the
actual examples.jsonl + teacher cache and runs the same item or padded summed
step that training uses, but stops after a small number of optimizer steps and
writes timing/memory JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config
from selfupdate.data.dataset import Batch
from selfupdate.teacher.cache import TeacherCache, resolve_cache_dir
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import (
    OnlineTeacherSource,
    _extend_pending_from_batch,
    _flush_train_log,
    _loader,
    _make_dataset,
    _move_opt_state,
    _pp_device_map,
    _summed_batch,
    _summed_item,
    _validate_knob_schedule,
)
from selfupdate.train.losses import HiddenLoss


class NullLog:
    def log(self, **kwargs):
        pass


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _step_opts(stack, opts, offload: bool, device: str) -> None:
    for L, opt in opts.items():
        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
        if offload:
            _move_opt_state(opt, device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if offload:
            _move_opt_state(opt, "cpu")


def bench(args) -> dict:
    cfg = load_config(args.config, args.experiment)
    cfg.model.device = args.device
    cfg.train.batching = args.batching
    cfg.train.micro_batch = args.micro_batch
    cfg.train.grad_accum = args.grad_accum or args.micro_batch
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
    pp_map = _pp_device_map(cfg) if cfg.model.pipeline_split > 0 else None
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

    stack = BlockStack(model)
    stack.freeze_non_blocks()
    if cfg.train.online_teacher and peft_model is None:
        raise ValueError("train.online_teacher requires train.lora.enabled")
    teacher = OnlineTeacherSource(stack, peft_model=peft_model) if cfg.train.online_teacher else None
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
    opts = {
        L: torch.optim.AdamW(
            [p for p in stack.block_params(L) if p.requires_grad], lr=cfg.train.lr
        )
        for L in range(1, n + 1)
    }

    steps = []
    accum = step = 0
    next_step = cfg.train.grad_accum
    pending_losses = []
    for items in loader:
        _sync(args.device)
        t0 = time.perf_counter()
        if isinstance(items, Batch):
            targets = (teacher.aligned_targets_batch(items, args.device)
                       if teacher is not None else items.hidden)
            layer_losses = _summed_batch(cfg, stack, loss_fn, items, targets, args.device)
            accum += len(items.example_ids)
            item_count = len(items.example_ids)
            token_count = int(items.lengths.sum())
            max_seq = int(items.lengths.max())
            pad_tokens = int(items.student_ids.numel() - items.lengths.sum())
            _extend_pending_from_batch(pending_losses, layer_losses)
        else:
            item_count = len(items)
            token_count = 0
            max_seq = 0
            pad_tokens = 0
            for it in items:
                targets = (teacher.aligned_targets(it, args.device)
                           if teacher is not None
                           else {L: it.hidden[L].to(args.device) for L in range(1, n + 1)})
                layer_losses = _summed_item(cfg, stack, loss_fn, it, targets, args.device)
                pending_losses.append(layer_losses)
                accum += 1
                token_count += len(it.student_ids)
                max_seq = max(max_seq, len(it.student_ids))
        if accum >= next_step:
            _step_opts(stack, opts, cfg.train.offload_adam, args.device)
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
                "max_seq": max_seq,
                "pad_tokens": pad_tokens,
                "seconds": elapsed,
                "items_per_second": item_count / elapsed,
                "tokens_per_second": token_count / elapsed,
            })
            step += 1
            next_step += cfg.train.grad_accum
            if step >= args.steps:
                break

    out = {
        "model": cfg.model.name,
        "schedule": cfg.train.schedule,
        "batching": cfg.train.batching,
        "micro_batch": cfg.train.micro_batch,
        "grad_accum": cfg.train.grad_accum,
        "hidden_loss": cfg.train.hidden_loss,
        "conn_window": cfg.train.conn_window,
        "readout_window_blocks": cfg.train.readout_window_blocks,
        "load_seconds": t_load1 - t_load0,
        "steps": steps,
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
                    default="bucketed")
    ap.add_argument("--micro-batch", type=int, default=2)
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
