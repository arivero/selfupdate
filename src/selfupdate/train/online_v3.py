"""Pipeline-v3 online block-local learning.

The atomic event is one aligned token.  Within that event blocks are visited
in forward order and each block receives one local backward followed
immediately by a state-free SGD write.  No gradient is averaged across
answers, tokens, or layers.  Earlier causal state is either recomputed from
the current weights or retained as an immutable per-answer cache; both are
discarded and rebuilt for the next answer (and therefore the next epoch).
"""

from __future__ import annotations

import random
import time

import torch
from transformers import DynamicCache

from ..data.dataset import DistillDataset
from .losses import HiddenLoss
from .telemetry import (
    ParameterDeltaTracker,
    _epoch_end_telemetry,
    _epoch_zero_telemetry,
    _flush_train_log,
)


def _flow_keep(cfg, it, length: int, device) -> torch.Tensor | None:
    """Return the full-history keep mask, or None for non-flow controls."""
    if cfg.mask.compaction != "flow_mask":
        return None
    keep = torch.ones((1, length), dtype=torch.bool, device=device)
    for start, stop in it.t_priv or []:
        if start < length:
            keep[:, start:min(stop, length)] = False
    return keep


def stage_answer_tensors(stack, it, device):
    """Stage one answer's immutable IDs, positions, and target rows once.

    Without this answer-local buffer, B=1/K=1 performs one tiny pageable-host
    transfer for every token/block cell. The largest v5 answer remains small
    relative to model weights, while removing that latency from the measured
    online loop. Buffers are discarded at the answer boundary.
    """
    ids = it.student_ids.to(device)
    positions = it.position_ids.to(device)
    targets = {}
    bytes_staged = ids.numel() * ids.element_size()
    bytes_staged += positions.numel() * positions.element_size()
    for layer in range(1, stack.n_layers + 1):
        if layer == stack.n_layers:
            param = next(stack.final_norm.parameters(), None)
        else:
            param = next(iter(stack.block_params(layer)), None)
        target_device = param.device if param is not None else torch.device(device)
        targets[layer] = it.hidden[layer].to(target_device)
        bytes_staged += targets[layer].numel() * targets[layer].element_size()
    return ids, positions, targets, bytes_staged


def _clear_block_grads(stack, layer: int) -> list[torch.nn.Parameter]:
    params = [p for p in stack.block_params(layer) if p.requires_grad]
    for param in params:
        param.grad = None
    return params


@torch.no_grad()
def _immediate_sgd(params: list[torch.nn.Parameter], lr: float) -> torch.Tensor:
    """Apply one state-free local write and release its gradient immediately.

    The online law requires one write per block, not one CUDA launch per
    parameter tensor.  Multi-tensor foreach kernels preserve the independent
    ``p -= lr*g`` updates while avoiding dozens of tiny update/reduction
    launches for a LoRA-instrumented block.
    """
    if not params:
        raise RuntimeError("pipeline-v3 reached a block with no trainable parameters")
    groups = {}
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        key = (param.device, param.dtype, grad.dtype)
        group = groups.setdefault(key, ([], []))
        group[0].append(param)
        group[1].append(grad.detach())
    if not groups:
        raise RuntimeError("pipeline-v3 local loss produced no block gradient")
    dev = params[0].device
    grad_sq = torch.zeros((), dtype=torch.float32, device=dev)
    for (group_device, _, _), (group_params, grads) in groups.items():
        norms = torch._foreach_norm(grads, 2)
        norm_vector = torch.stack(norms).float()
        group_sq = norm_vector.square().sum()
        grad_sq.add_(group_sq.to(dev))
        torch._foreach_add_(group_params, grads, alpha=-lr)
    for param in params:
        param.grad = None
    return grad_sq.sqrt()


@torch.no_grad()
def _immediate_sgd_token(params_by_layer: list[list[torch.nn.Parameter]],
                         lr: float) -> list[torch.Tensor]:
    """One multi-tensor write for a token, retaining per-layer grad norms."""
    groups = {}
    for layer_index, params in enumerate(params_by_layer):
        for param in params:
            grad = param.grad
            if grad is None:
                continue
            key = (param.device, param.dtype, grad.dtype)
            group = groups.setdefault(key, ([], [], []))
            group[0].append(param)
            group[1].append(grad.detach())
            group[2].append(layer_index)
    if not groups:
        raise RuntimeError("pipeline-v3 token loss produced no block gradients")
    layer_sq: list[torch.Tensor | None] = [None] * len(params_by_layer)
    for (device, _, _), (params, grads, layer_indices) in groups.items():
        norm_vector = torch.stack(torch._foreach_norm(grads, 2)).float()
        ids = torch.tensor(layer_indices, dtype=torch.long, device=device)
        group_layer_sq = torch.zeros(
            len(params_by_layer), dtype=torch.float32, device=device)
        group_layer_sq.scatter_add_(0, ids, norm_vector.square())
        for layer_index in set(layer_indices):
            value = group_layer_sq[layer_index]
            previous = layer_sq[layer_index]
            layer_sq[layer_index] = (
                value if previous is None else previous + value.to(previous.device))
        torch._foreach_add_(params, grads, alpha=-lr)
    for params in params_by_layer:
        for param in params:
            param.grad = None
    if any(value is None for value in layer_sq):
        missing = [i + 1 for i, value in enumerate(layer_sq) if value is None]
        raise RuntimeError(
            f"pipeline-v3 token loss produced no gradient for blocks {missing}")
    return [value.sqrt() for value in layer_sq]


@torch.no_grad()
def _foreach_accumulate(sums: list[torch.Tensor],
                        values: list[torch.Tensor]) -> None:
    """Accumulate one token's per-layer scalars with one launch per device."""
    groups = {}
    for total, value in zip(sums, values):
        key = (total.device, total.dtype, value.dtype)
        pair = groups.setdefault(key, ([], []))
        pair[0].append(total)
        pair[1].append(value)
    for totals, additions in groups.values():
        torch._foreach_add_(totals, additions)


def _local_forward(cfg, stack, loss_fn, layer: int, h_in: torch.Tensor,
                   pos_emb, position_ids: torch.Tensor, target: torch.Tensor,
                   row: int, *, flow_keep=None, cache=None,
                   causal_length=None):
    """Build one isolated block-local objective and return its parameters."""
    params = _clear_block_grads(stack, layer)
    h_in = h_in.detach()
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16,
                        enabled=h_in.device.type == "cuda"):
        h_out = stack.run_block(
            layer, h_in, pos_emb, position_ids=position_ids,
            flow_keep=flow_keep, past_key_values=cache,
            use_cache=cache is not None, causal_length=causal_length,
        )
        view = stack.loss_view(layer, h_out)[0, row]
        loss = loss_fn(
            view, target.to(view.device), normed=(layer == stack.n_layers),
            layer=layer,
        )
    return loss, h_out, params


def _local_update(cfg, stack, loss_fn, layer: int, h_in: torch.Tensor,
                  pos_emb, position_ids: torch.Tensor, target: torch.Tensor,
                  row: int, *, flow_keep=None, cache=None,
                  causal_length=None):
    """One block forward, one local backward, one immediate parameter write."""
    loss, h_out, params = _local_forward(
        cfg, stack, loss_fn, layer, h_in, pos_emb, position_ids, target, row,
        flow_keep=flow_keep, cache=cache, causal_length=causal_length)
    loss.backward()
    grad_norm = _immediate_sgd(params, cfg.train.lr)
    if cache is not None:
        _detach_cache_layer(cache, layer - 1)
    # The forward value belongs to the pre-write model and is the causal
    # input to the next block for this token.  It never carries a graph.
    return loss.detach(), grad_norm.detach(), h_out.detach()


def _finish_disconnected_token(cfg, losses, params_by_layer):
    """Backward disconnected local roots once, then write before next token.

    Summing roots is only an autograd dispatch device. Every root has a
    detached block input and a disjoint parameter set, so no gradient is
    averaged, shared, or propagated between layers.
    """
    # A list of scalar roots enters the autograd engine once without even a
    # synthetic sum kernel. Multi-device roots are supported by the engine;
    # their graphs remain disjoint because every block input was detached.
    torch.autograd.backward(losses)
    grad_norms = [value.detach() for value in
                  _immediate_sgd_token(params_by_layer, cfg.train.lr)]
    return [loss.detach() for loss in losses], grad_norms


def _detach_value(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, list):
        return [_detach_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_value(v) for k, v in value.items()}
    return value


def _detach_cache_layer(cache: DynamicCache, index: int) -> None:
    """Make history a constant after the current token's backward.

    Dynamic attention caches concatenate tensors and would otherwise retain
    the just-completed autograd graph. Linear/recurrent caches use in-place
    state buffers; detaching their tensor attributes makes the same frozen-
    history law explicit.
    """
    layer = cache.layers[index]
    for name, value in vars(layer).items():
        detached = _detach_value(value)
        if detached is not value:
            setattr(layer, name, detached)


@torch.no_grad()
def _prefill_student(cfg, stack, it, stop: int, cache: DynamicCache, device,
                     *, student_ids=None, position_ids=None):
    if stop <= 0:
        return
    ids = (student_ids[:stop] if student_ids is not None
           else it.student_ids[:stop].to(device))[None]
    pos = (position_ids[:stop] if position_ids is not None
           else it.position_ids[:stop].to(device))[None]
    h = stack.embed(ids)
    pos_emb = stack.rope(h, pos)
    keep = _flow_keep(cfg, it, stop, h.device)
    for layer in range(1, stack.n_layers + 1):
        h = stack.run_block(
            layer, h, pos_emb, position_ids=pos, flow_keep=keep,
            past_key_values=cache, use_cache=True, causal_length=stop,
        ).detach()
        _detach_cache_layer(cache, layer - 1)


@torch.no_grad()
def _prefill_teacher(cfg, stack, it, teacher_states, stop: int,
                     cache: DynamicCache, device):
    if stop <= 0:
        return
    pos = it.position_ids[:stop].to(device)[None]
    for layer in range(1, stack.n_layers + 1):
        h_in = teacher_states[layer - 1][:, :stop].to(device).detach()
        pos_emb = stack.rope(h_in, pos)
        keep = _flow_keep(cfg, it, stop, h_in.device)
        stack.run_block(
            layer, h_in, pos_emb, position_ids=pos, flow_keep=keep,
            past_key_values=cache, use_cache=True, causal_length=stop,
        )
        _detach_cache_layer(cache, layer - 1)


def _token_recompute(cfg, stack, loss_fn, it, offset: int, device,
                     teacher_states=None, *, student_ids=None,
                     position_ids=None, targets=None):
    pos_index = it.s0 + offset
    stop = pos_index + 1
    pos = (position_ids[:stop] if position_ids is not None
           else it.position_ids[:stop].to(device))[None]
    losses, grad_norms, deferred_params = [], [], []
    deferred = cfg.train.backward_dispatch == "per_token_disconnected"
    if teacher_states is None:
        ids = (student_ids[:stop] if student_ids is not None
               else it.student_ids[:stop].to(device))[None]
        h = stack.embed(ids).detach()
        pos_emb = stack.rope(h, pos)
        keep = _flow_keep(cfg, it, stop, h.device)
        for layer in range(1, stack.n_layers + 1):
            if deferred:
                loss, h_out, params = _local_forward(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], pos_index,
                    flow_keep=keep)
                losses.append(loss)
                deferred_params.append(params)
                h = h_out.detach()
            else:
                loss, grad, h = _local_update(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], pos_index,
                    flow_keep=keep)
                losses.append(loss)
                grad_norms.append(grad)
    else:
        for layer in range(1, stack.n_layers + 1):
            h_in = teacher_states[layer - 1][:, :stop].to(device).detach()
            pos_emb = stack.rope(h_in, pos)
            keep = _flow_keep(cfg, it, stop, h_in.device)
            if deferred:
                loss, _, params = _local_forward(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], pos_index,
                    flow_keep=keep)
                losses.append(loss)
                deferred_params.append(params)
            else:
                loss, grad, _ = _local_update(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], pos_index,
                    flow_keep=keep)
                losses.append(loss)
                grad_norms.append(grad)
    if deferred:
        losses, grad_norms = _finish_disconnected_token(
            cfg, losses, deferred_params)
    return losses, grad_norms


def _token_cached(cfg, stack, loss_fn, it, offset: int, device,
                  cache: DynamicCache, teacher_states=None, *,
                  student_ids=None, position_ids=None, targets=None):
    pos_index = it.s0 + offset
    pos = (position_ids[pos_index:pos_index + 1]
           if position_ids is not None else
           it.position_ids[pos_index:pos_index + 1].to(device))[None]
    full_keep = _flow_keep(cfg, it, pos_index + 1, device)
    losses, grad_norms, deferred_params = [], [], []
    deferred = cfg.train.backward_dispatch == "per_token_disconnected"
    if teacher_states is None:
        ids = (student_ids[pos_index:pos_index + 1]
               if student_ids is not None else
               it.student_ids[pos_index:pos_index + 1].to(device))[None]
        h = stack.embed(ids).detach()
        for layer in range(1, stack.n_layers + 1):
            pos_emb = stack.rope(h, pos)
            if deferred:
                loss, h_out, params = _local_forward(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1)
                losses.append(loss)
                deferred_params.append(params)
                h = h_out.detach()
            else:
                loss, grad, h = _local_update(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1)
                losses.append(loss)
                grad_norms.append(grad)
    else:
        for layer in range(1, stack.n_layers + 1):
            h_in = teacher_states[layer - 1][
                :, pos_index:pos_index + 1].to(device).detach()
            pos_emb = stack.rope(h_in, pos)
            if deferred:
                loss, _, params = _local_forward(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1)
                losses.append(loss)
                deferred_params.append(params)
            else:
                loss, grad, _ = _local_update(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1)
                losses.append(loss)
                grad_norms.append(grad)
    if deferred:
        losses, grad_norms = _finish_disconnected_token(
            cfg, losses, deferred_params)
        for layer in range(stack.n_layers):
            _detach_cache_layer(cache, layer)
    return losses, grad_norms


def train_online_v3(cfg, stack, tok, log, cache, teacher=None) -> None:
    """Run the pipeline-v3 online walk; checkpoint publication stays in runtime."""
    device = cfg.model.device
    n = stack.n_layers
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    teacher_hidden = cfg.train.trajectory_source == "teacher_hidden"
    ds = DistillDataset(
        cfg.data.examples_path,
        cache,
        tok,
        need_layers=list(range(1, n + 1)),
        with_teacher_ids=teacher_hidden,
        pad_random=(cfg.mask.compaction == "pad_random"),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
    )
    delta_tracker = ParameterDeltaTracker(stack)
    started = time.time()
    token_events = optimizer_updates = answers_seen = 0
    max_answer_stage_bytes = 0
    done = False
    standard_baseline = _epoch_zero_telemetry(
        cfg, stack, tok, log, started)
    delta_tracker.log(log, epoch=0, phase="epoch0", started_at=started)
    log.log(
        kind="pipeline_v3_contract",
        atomic_event="one_aligned_token",
        block_order="forward",
        updates_per_token=n,
        optimizer="state_free_immediate_sgd",
        backward_dispatch=cfg.train.backward_dispatch,
        gradient_aggregation="none",
        history_policy=cfg.train.history_policy,
        history_lifetime="current_answer_only",
        trajectory_source=cfg.train.trajectory_source,
        teacher_hidden_identity="uncensored_teacher_h[L-1]",
    )

    for epoch in range(cfg.train.epochs):
        epoch_started = time.time()
        epoch_token_start = token_events
        epoch_write_start = optimizer_updates
        progress_token_start = token_events
        progress_started = epoch_started
        pending_losses: list[list[torch.Tensor]] = []
        pending_grad_norms: list[list[torch.Tensor]] = []
        order = list(range(len(ds)))
        random.Random(cfg.train.seed + epoch).shuffle(order)
        for index in order:
            it = ds[index]
            student_ids, position_ids, targets, staged_bytes = (
                stage_answer_tensors(stack, it, device))
            max_answer_stage_bytes = max(max_answer_stage_bytes, staged_bytes)
            teacher_states = (
                teacher.full_states_cpu(it, device) if teacher_hidden else None)
            history = None
            if cfg.train.history_policy == "causal_frozen_history":
                history = DynamicCache(config=stack.text_config)
                if teacher_hidden:
                    _prefill_teacher(
                        cfg, stack, it, teacher_states, it.s0, history, device)
                else:
                    _prefill_student(
                        cfg, stack, it, it.s0, history, device,
                        student_ids=student_ids, position_ids=position_ids)
            answer_loss_sums = None
            answer_grad_sums = None
            answer_tokens = 0
            for offset in range(it.A):
                if history is None:
                    losses, grads = _token_recompute(
                        cfg, stack, loss_fn, it, offset, device,
                        teacher_states=teacher_states,
                        student_ids=student_ids, position_ids=position_ids,
                        targets=targets)
                else:
                    losses, grads = _token_cached(
                        cfg, stack, loss_fn, it, offset, device, history,
                        teacher_states=teacher_states,
                        student_ids=student_ids, position_ids=position_ids,
                        targets=targets)
                if answer_loss_sums is None:
                    answer_loss_sums = losses
                    answer_grad_sums = grads
                else:
                    _foreach_accumulate(answer_loss_sums, losses)
                    _foreach_accumulate(answer_grad_sums, grads)
                answer_tokens += 1
                token_events += 1
                optimizer_updates += n
                if cfg.train.max_steps and token_events >= cfg.train.max_steps:
                    done = True
                    break
            if answer_tokens:
                pending_losses.append([x / answer_tokens
                                       for x in answer_loss_sums])
                pending_grad_norms.append([x / answer_tokens
                                           for x in answer_grad_sums])
            answers_seen += 1
            del teacher_states
            del history
            del student_ids
            del position_ids
            del targets
            if token_events - progress_token_start >= 1000:
                progress_seconds = time.time() - progress_started
                progress_tokens = token_events - progress_token_start
                log.log(
                    kind="v3_progress",
                    epoch=epoch + 1,
                    token_events_seen=token_events,
                    optimizer_updates_seen=optimizer_updates,
                    answers_seen=answers_seen,
                    interval_token_events=progress_tokens,
                    interval_seconds=progress_seconds,
                    interval_token_events_per_s=(
                        progress_tokens / progress_seconds),
                    backward_dispatch=cfg.train.backward_dispatch,
                )
                print(
                    f"v3 epoch {epoch + 1} tokens {token_events} "
                    f"rate {progress_tokens / progress_seconds:.2f}/s "
                    f"dispatch {cfg.train.backward_dispatch}",
                    flush=True,
                )
                progress_token_start = token_events
                progress_started = time.time()
            if done:
                break

        _flush_train_log(
            log, epoch=epoch, step=token_events, accum=answers_seen,
            pending=pending_losses, n_layers=n, partial=done,
            pipeline_version=3, update_granularity="online",
            update_reduction="none", trajectory_source=cfg.train.trajectory_source,
            history_policy=cfg.train.history_policy,
            token_events_seen=token_events,
            optimizer_updates_seen=optimizer_updates,
            optimizer_updates_per_token=n,
            gradient_aggregation="none",
            completed_epochs=(epoch if done else epoch + 1),
            partial_epoch=done,
            max_answer_stage_mib=max_answer_stage_bytes / 2**20,
        )
        if done:
            log.log(
                kind="v3_partial_boundary",
                token_events_seen=token_events,
                optimizer_updates_seen=optimizer_updates,
                completed_epochs=epoch,
                partial_epoch_index=epoch + 1,
                meaning=(
                    "budget checkpoint inside a dataset traversal; not a "
                    "completed epoch"),
            )
        if pending_grad_norms:
            by_layer = []
            for layer in range(n):
                vals = [row[layer].to(pending_grad_norms[0][layer].device)
                        for row in pending_grad_norms]
                by_layer.append(torch.stack(vals).mean().detach().cpu())
            log.log(
                kind="v3_gradient_norm",
                epoch=epoch + 1,
                per_layer_mean=[float(x) for x in by_layer],
                token_events_seen=token_events,
                optimizer_updates_seen=optimizer_updates,
            )
        epoch_seconds = time.time() - epoch_started
        epoch_tokens = token_events - epoch_token_start
        epoch_writes = optimizer_updates - epoch_write_start
        log.log(
            kind="v3_throughput",
            epoch=epoch + 1,
            seconds=epoch_seconds,
            token_events=epoch_tokens,
            local_writes=epoch_writes,
            token_events_per_s=(epoch_tokens / epoch_seconds),
            local_writes_per_s=(epoch_writes / epoch_seconds),
            includes="dataset_cache_io_plus_prompt_prefill_plus_online_writes",
            history_policy=cfg.train.history_policy,
            trajectory_source=cfg.train.trajectory_source,
            completed_epochs=(epoch if done else epoch + 1),
            partial_epoch=done,
        )
        standard_baseline = _epoch_end_telemetry(
            cfg, stack, tok, log, epoch=epoch, baseline=standard_baseline,
            started_at=started)
        delta_tracker.log(
            log, epoch=epoch + 1, phase=f"after_epoch_{epoch + 1}",
            started_at=started)
        if done:
            break
