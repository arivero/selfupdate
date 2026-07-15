"""One-tile v3.1 simultaneous-user B×K GPU benchmark.

B is independent serving parallelism across live conversations. K is
within-answer lookahead: K=1 is ordinary next-token online execution; K>1
requires prefetched teacher answers or speculative tokens. Gradients over all
valid B×K cells are summed, not averaged, before one block-local SGD write.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.data.dataset import DistillDataset, collate_padded_items  # noqa: E402
from selfupdate.train.layerwise import (  # noqa: E402
    TrainingRuntime,
    _validate_knob_schedule,
    dequantize_overrides,
)
from selfupdate.train.losses import HiddenLoss  # noqa: E402
from selfupdate.train.online_v3 import (  # noqa: E402
    _clear_block_grads,
    _detach_cache_layer,
    _immediate_sgd,
)
from selfupdate.utils.env import cap_cpu_threads  # noqa: E402
from selfupdate.utils.seeding import seed_everything  # noqa: E402


def _sync() -> None:
    torch.cuda.synchronize()


def _snapshot(stack):
    return {
        layer: [p.detach().float().cpu().clone()
                for p in stack.block_params(layer) if p.requires_grad]
        for layer in range(1, stack.n_layers + 1)
    }


def _delta_l2(stack, before):
    values = []
    for layer, refs in before.items():
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        sq = sum(float((p.detach().float().cpu() - ref).square().sum())
                 for p, ref in zip(params, refs))
        values.append(sq ** 0.5)
    return values


def _tight_median_bucket(ds, width: int) -> list[int]:
    if width > len(ds):
        raise ValueError(f"B={width} exceeds dataset size {len(ds)}")
    ordered = sorted(
        range(len(ds)), key=lambda index: len(ds.pairs[index].student_ids))
    start = max(0, len(ordered) // 2 - width // 2)
    return ordered[start:start + width]


def _prefix_layout(items, device):
    lengths = torch.tensor([it.s0 for it in items], device=device)
    maximum = int(lengths.max())
    timeline = torch.arange(maximum, device=device)[None]
    left = maximum - lengths[:, None]
    valid = timeline >= left
    source = (timeline - left).clamp_min(0)
    keep = valid.clone()
    for row, it in enumerate(items):
        shift = maximum - it.s0
        for start, stop in it.t_priv or []:
            keep[row, shift + start:shift + min(stop, it.s0)] = False
    return maximum, source.long(), valid, keep


def _gather_hidden(value, index):
    return value.gather(
        1, index[:, :, None].expand(-1, -1, value.shape[-1]))


def main() -> None:
    cap_cpu_threads()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, args.experiment)
    seed_everything(cfg.train.seed)
    _validate_knob_schedule(cfg)
    if cfg.train.history_policy != "causal_bk_probe":
        raise ValueError("v31_bk_smoke requires history_policy=causal_bk_probe")
    B = cfg.train.micro_batch
    K = cfg.train.stale_gradient_window
    load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    rt = TrainingRuntime(cfg).load(load_kw)
    stack, tok, cache = rt.stack, rt.tokenizer, rt.load_cache()
    if stack.block_devices is not None or any(
            kind not in (None, "full_attention") for kind in stack.layer_types):
        raise NotImplementedError(
            "first v3.1 B×K probe requires one-device full attention")
    teacher = rt.load_teacher(load_kw)
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=list(range(1, stack.n_layers + 1)),
        with_teacher_ids=True,
        pad_random=(cfg.mask.compaction == "pad_random"),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
    )
    indices = _tight_median_bucket(ds, B)
    items = [ds[index] for index in indices]
    batch = collate_padded_items(items)
    device = torch.device(cfg.model.device)
    if min(it.A for it in items) < K:
        raise ValueError("selected B bucket has an answer shorter than K")

    _sync()
    teacher_started = time.perf_counter()
    teacher_inputs = teacher.full_inputs_resident_batch(batch, device)
    targets = {
        layer: batch.hidden[layer][:, :K].to(device)
        for layer in range(1, stack.n_layers + 1)
    }
    _sync()
    teacher_s = time.perf_counter() - teacher_started

    prompt_length, prefix_index, prefix_valid, prefix_keep = _prefix_layout(
        items, device)
    prefix_positions = prefix_index
    q_offset = torch.arange(K, device=device)[None]
    s0 = torch.tensor([it.s0 for it in items], device=device)[:, None]
    query_index = s0 + q_offset
    query_positions = query_index
    query_valid = q_offset < torch.tensor(
        [it.A for it in items], device=device)[:, None]
    token_events = int(query_valid.sum())
    key_keep = torch.cat((prefix_keep, query_valid), dim=1)

    q_timeline = torch.arange(prompt_length, device=device)[:, None]
    k_timeline = torch.arange(prompt_length, device=device)[None]
    prefill_allowed = ((k_timeline <= q_timeline)[None]
                       & prefix_keep[:, None, :])
    mask_dtype = teacher_inputs[0].dtype
    prefill_mask = torch.zeros(
        (B, 1, prompt_length, prompt_length),
        dtype=mask_dtype, device=device)
    prefill_mask.masked_fill_(
        ~prefill_allowed[:, None], torch.finfo(mask_dtype).min)
    query_timeline = torch.arange(
        prompt_length, prompt_length + K, device=device)[:, None]
    all_keys = torch.arange(prompt_length + K, device=device)[None]
    query_allowed = ((all_keys <= query_timeline)[None]
                     & key_keep[:, None, :])
    query_mask = torch.zeros(
        (B, 1, K, prompt_length + K), dtype=mask_dtype, device=device)
    query_mask.masked_fill_(
        ~query_allowed[:, None], torch.finfo(mask_dtype).min)

    history = DynamicCache(config=stack.text_config)
    _sync()
    prefill_started = time.perf_counter()
    first_prefix = _gather_hidden(teacher_inputs[0], prefix_index)
    prefix_rope = stack.rope(first_prefix, prefix_positions)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for layer in range(1, stack.n_layers + 1):
            h = _gather_hidden(teacher_inputs[layer - 1], prefix_index)
            stack.run_block(
                layer, h, prefix_rope, position_ids=prefix_positions,
                flow_keep=prefix_keep, past_key_values=history,
                use_cache=True, causal_length=prompt_length,
                prepared_attention_mask=prefill_mask)
            _detach_cache_layer(history, layer - 1)
    _sync()
    prefill_s = time.perf_counter() - prefill_started

    before = _snapshot(stack)
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    first_query = _gather_hidden(teacher_inputs[0], query_index)
    query_rope = stack.rope(first_query, query_positions)
    losses = []
    grads = []
    online_memory_baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    _sync()
    tile_started = time.perf_counter()
    for layer in range(1, stack.n_layers + 1):
        params = _clear_block_grads(stack, layer)
        h_in = _gather_hidden(teacher_inputs[layer - 1], query_index).detach()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_out = stack.run_block(
                layer, h_in, query_rope, position_ids=query_positions,
                flow_keep=key_keep, past_key_values=history,
                use_cache=True, causal_length=prompt_length + K,
                prepared_attention_mask=query_mask)
            view = stack.loss_view(layer, h_out)[query_valid]
            target = targets[layer][query_valid]
            mean_loss = loss_fn(
                view, target, normed=(layer == stack.n_layers), layer=layer)
            summed_loss = mean_loss * token_events
        summed_loss.backward()
        grad = _immediate_sgd(params, cfg.train.lr)
        _detach_cache_layer(history, layer - 1)
        losses.append(mean_loss.detach())
        grads.append(grad.detach() / token_events)
    _sync()
    tile_s = time.perf_counter() - tile_started
    peak = torch.cuda.max_memory_allocated(device)
    deltas = _delta_l2(stack, before)
    rt.check_vocab_frozen()
    print(json.dumps({
        "status": "passed",
        "pipeline_revision": "3.1",
        "B_simultaneous_users": B,
        "K_context_tokens": K,
        "lookahead_contract": (
            "next_token_online" if K == 1 else
            "teacher_prefetched_or_speculative_confirmed_tokens"),
        "gradient_aggregation": "unaveraged_sum_over_valid_BxK_cells",
        "example_bucket": "tight_median_by_student_sequence_length",
        "sequence_length_min": min(len(it.student_ids) for it in items),
        "sequence_length_max": max(len(it.student_ids) for it in items),
        "prompt_length_padded": prompt_length,
        "token_events": token_events,
        "conceptual_block_local_writes": token_events * stack.n_layers,
        "physical_block_writes": stack.n_layers,
        "teacher_batch_s": teacher_s,
        "prefill_s": prefill_s,
        "tile_s": tile_s,
        "tile_token_events_per_s": token_events / tile_s,
        "end_to_end_token_events_per_s": token_events / (
            teacher_s + prefill_s + tile_s),
        "peak_allocated_mib": peak / 2**20,
        "peak_increment_mib": (peak - online_memory_baseline) / 2**20,
        "loss_min": min(float(value.cpu()) for value in losses),
        "loss_max": max(float(value.cpu()) for value in losses),
        "normalized_window_grad_min": min(float(value.cpu()) for value in grads),
        "normalized_window_grad_max": max(float(value.cpu()) for value in grads),
        "parameter_delta_l2_total": sum(v * v for v in deltas) ** 0.5,
        "parameter_delta_l2_by_layer": deltas,
        "cache_graph": False,
        "vocab_frozen": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
