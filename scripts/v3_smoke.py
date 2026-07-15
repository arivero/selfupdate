"""On-demand one-token GPU smoke for pipeline v3 (no checkpoint/report).

This is an execution probe, not a stored test or scientific run. It checks
the architecture adapter, flow-censorship invariance, one immediate local
write per block, cache graph detachment, and the frozen-vocabulary tripwire.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.data.dataset import DistillDataset  # noqa: E402
from selfupdate.train.layerwise import (  # noqa: E402
    TrainingRuntime,
    _validate_knob_schedule,
    dequantize_overrides,
)
from selfupdate.train.losses import HiddenLoss  # noqa: E402
from selfupdate.train.online_v3 import (  # noqa: E402
    _flow_keep,
    _prefill_student,
    _prefill_teacher,
    _prepared_cached_masks,
    answer_teacher_stale_windows_cached,
    answer_student_pipeline_lanes_cached,
    answer_teacher_layer_lanes_cached,
    answer_wavefront_cached,
    stage_answer_tensors,
    _token_cached,
    _token_recompute,
)
from selfupdate.utils.env import cap_cpu_threads  # noqa: E402
from selfupdate.utils.seeding import seed_everything  # noqa: E402


def _trainable_snapshot(stack):
    return {
        layer: [p.detach().float().cpu().clone()
                for p in stack.block_params(layer) if p.requires_grad]
        for layer in range(1, stack.n_layers + 1)
    }


def _changed_layers(stack, before) -> list[int]:
    changed = []
    for layer, refs in before.items():
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        delta = sum(float((p.detach().float().cpu() - ref).abs().sum())
                    for p, ref in zip(params, refs))
        if delta > 0:
            changed.append(layer)
    return changed


def _layer_delta_l2(stack, before) -> list[float]:
    out = []
    for layer, refs in before.items():
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        sq = sum(float((p.detach().float().cpu() - ref).square().sum())
                 for p, ref in zip(params, refs))
        out.append(sq ** 0.5)
    return out


def _write_trainable_deltas(stack, before, path: str) -> None:
    """Persist exact float32 parameter deltas for cross-geometry comparison."""
    tensors = {}
    for layer, refs in before.items():
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        for index, (param, ref) in enumerate(zip(params, refs)):
            tensors[f"layer{layer:03d}.param{index:03d}"] = (
                param.detach().float().cpu() - ref).contiguous()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cache_has_graph(cache) -> bool:
    def visit(value):
        if torch.is_tensor(value):
            return value.requires_grad or value.grad_fn is not None
        if isinstance(value, (list, tuple)):
            return any(visit(v) for v in value)
        if isinstance(value, dict):
            return any(visit(v) for v in value.values())
        return False

    return any(visit(vars(layer)) for layer in cache.layers)


@torch.no_grad()
def _flow_invariance(cfg, stack, it, device) -> float:
    ids_a = it.student_ids.to(device)[None]
    ids_b = ids_a.clone()
    for start, stop in it.t_priv or []:
        ids_b[:, start:stop] = (
            ids_b[:, start:stop] + 1) % stack.embed_tokens.num_embeddings
    pos = it.position_ids.to(device)[None]
    keep = _flow_keep(cfg, it, ids_a.shape[1], device)
    h_a = stack.embed(ids_a)
    h_b = stack.embed(ids_b)
    pe_a = stack.rope(h_a, pos)
    pe_b = stack.rope(h_b, pos)
    for layer in range(1, stack.n_layers + 1):
        h_a = stack.run_block(
            layer, h_a, pe_a, position_ids=pos, flow_keep=keep)
        h_b = stack.run_block(
            layer, h_b, pe_b, position_ids=pos, flow_keep=keep)
    return float((h_a[:, it.s0:] - h_b[:, it.s0:]).abs().max().cpu())


def main() -> None:
    cap_cpu_threads()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--longest", action="store_true",
                    help="select the dataset record with maximum aligned A")
    ap.add_argument(
        "--delta-out", default="",
        help="optional safetensors path for exact trainable-parameter deltas")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    seed_everything(cfg.train.seed)
    _validate_knob_schedule(cfg)
    if cfg.train.pipeline_version != 3:
        raise ValueError("v3_smoke requires pipeline_version=3")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    launched = time.perf_counter()
    load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    rt = TrainingRuntime(cfg).load(load_kw)
    _sync()
    model_load_s = time.perf_counter() - launched
    stack, tok, cache = rt.stack, rt.tokenizer, rt.load_cache()
    teacher_hidden = cfg.train.trajectory_source == "teacher_hidden"
    teacher = rt.load_teacher(load_kw) if teacher_hidden else None
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=list(range(1, stack.n_layers + 1)),
        with_teacher_ids=teacher_hidden,
        pad_random=(cfg.mask.compaction == "pad_random"),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
    )
    item_index = (max(range(len(ds)), key=lambda i: ds.pairs[i].aligned_len)
                  if args.longest else 0)
    it = ds[item_index]
    device = cfg.model.device
    flow_max_abs = None
    flow_cert_s = 0.0
    if cfg.mask.compaction == "flow_mask":
        started = time.perf_counter()
        flow_max_abs = _flow_invariance(cfg, stack, it, device)
        _sync()
        flow_cert_s = time.perf_counter() - started
    if flow_max_abs not in (None, 0.0):
        raise RuntimeError(
            f"flow censorship leaked privileged token identity: {flow_max_abs}")

    started = time.perf_counter()
    student_ids, position_ids, targets, staged_bytes = stage_answer_tensors(
        stack, it, device)
    _sync()
    answer_stage_s = time.perf_counter() - started
    before = _trainable_snapshot(stack)
    started = time.perf_counter()
    teacher_states = (
        teacher.full_inputs_resident(it, device) if teacher_hidden else None)
    _sync()
    teacher_states_s = time.perf_counter() - started
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    history = None
    prefill_s = 0.0
    if cfg.train.history_policy == "causal_frozen_history":
        history = DynamicCache(config=stack.text_config)
        started = time.perf_counter()
        if teacher_hidden:
            _prefill_teacher(
                cfg, stack, it, teacher_states, it.s0, history, device)
        else:
            _prefill_student(
                cfg, stack, it, it.s0, history, device,
                student_ids=student_ids, position_ids=position_ids)
        _sync()
        prefill_s = time.perf_counter() - started
    token_count = min(args.tokens, it.A)
    losses, grad_norms = [], []
    _sync()
    online_memory_baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    local_writes = 0
    physical_local_writes = 0
    use_teacher_window_path = (
        teacher_hidden and history is not None
        and cfg.train.backward_dispatch == "per_block"
        and cfg.train.online_write_dispatch == "after_backward")
    if use_teacher_window_path:
        (losses, grad_norms, local_writes,
         physical_local_writes) = answer_teacher_stale_windows_cached(
            cfg, stack, loss_fn, it, token_count, device, history,
            teacher_states=teacher_states, position_ids=position_ids,
            targets=targets)
    elif cfg.train.backward_dispatch == "teacher_layer_lanes":
        losses, grad_norms, local_writes = answer_teacher_layer_lanes_cached(
            cfg, stack, loss_fn, it, token_count, device, history,
            teacher_states=teacher_states, position_ids=position_ids,
            targets=targets)
    elif cfg.train.backward_dispatch == "answer_pipeline_lanes":
        losses, grad_norms, local_writes = answer_student_pipeline_lanes_cached(
            cfg, stack, loss_fn, it, token_count, device, history,
            student_ids=student_ids, position_ids=position_ids,
            targets=targets)
    elif cfg.train.backward_dispatch == "answer_wavefront_disconnected":
        losses, grad_norms, local_writes = answer_wavefront_cached(
            cfg, stack, loss_fn, it, token_count, device, history,
            student_ids=student_ids, position_ids=position_ids,
            targets=targets)
    else:
        prepared_masks = (_prepared_cached_masks(
            cfg, stack, it, position_ids, targets)
            if history is not None else None)
        for offset in range(token_count):
            if history is not None:
                step_losses, step_grads = _token_cached(
                    cfg, stack, loss_fn, it, offset, device, history,
                    teacher_states=teacher_states, student_ids=student_ids,
                    position_ids=position_ids, targets=targets,
                    prepared_masks=prepared_masks)
            else:
                step_losses, step_grads = _token_recompute(
                    cfg, stack, loss_fn, it, offset, device,
                    teacher_states=teacher_states, student_ids=student_ids,
                    position_ids=position_ids, targets=targets)
            losses.extend(step_losses)
            grad_norms.extend(step_grads)
            local_writes += stack.n_layers
    if not physical_local_writes:
        physical_local_writes = local_writes
    _sync()
    online_s = time.perf_counter() - started
    online_peak_memory = torch.cuda.max_memory_allocated()
    changed = _changed_layers(stack, before)
    layer_delta_l2 = _layer_delta_l2(stack, before)
    if args.delta_out:
        _write_trainable_deltas(stack, before, args.delta_out)
    expected = list(range(1, stack.n_layers + 1))
    if cfg.mask.compaction != "intact" and changed != expected:
        raise RuntimeError(
            f"online write changed layers {changed}, expected {expected}")
    if history is not None and _cache_has_graph(history):
        raise RuntimeError("causal_frozen_history retained an autograd graph")
    rt.check_vocab_frozen()
    print(json.dumps({
        "status": "passed",
        "model": cfg.model.name,
        "trajectory_source": cfg.train.trajectory_source,
        "history_policy": cfg.train.history_policy,
        "backward_dispatch": cfg.train.backward_dispatch,
        "online_write_dispatch": cfg.train.online_write_dispatch,
        "stale_gradient_window": cfg.train.stale_gradient_window,
        "censorship": cfg.mask.compaction,
        "layers": stack.n_layers,
        "token_events": token_count,
        "example_id": it.example_id,
        "aligned_tokens_available": it.A,
        "flow_max_abs": flow_max_abs,
        "changed_layers": changed,
        "layer_raw_parameter_delta_l2": layer_delta_l2,
        "parameter_delta_l2_max": max(layer_delta_l2),
        "parameter_delta_l2_total": sum(
            x * x for x in layer_delta_l2) ** 0.5,
        "delta_out": args.delta_out or None,
        "loss_min": min(float(x.cpu()) for x in losses),
        "loss_max": max(float(x.cpu()) for x in losses),
        "grad_norm_min": min(float(x.cpu()) for x in grad_norms),
        "grad_norm_max": max(float(x.cpu()) for x in grad_norms),
        "model_load_s": model_load_s,
        "flow_cert_s": flow_cert_s,
        "teacher_states_s": teacher_states_s,
        "answer_stage_s": answer_stage_s,
        "answer_stage_mib": staged_bytes / 2**20,
        "prefill_s": prefill_s,
        "online_s": online_s,
        "online_peak_allocated_mib": online_peak_memory / 2**20,
        "online_peak_increment_mib": (
            online_peak_memory - online_memory_baseline) / 2**20,
        "online_token_events_per_s": token_count / online_s,
        "online_local_writes_per_s": local_writes / online_s,
        "physical_local_writes": physical_local_writes,
        "online_physical_local_writes_per_s": (
            physical_local_writes / online_s),
        "with_prefill_token_events_per_s": (
            token_count / (prefill_s + online_s)),
        "cache_graph": False,
        "vocab_frozen": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
