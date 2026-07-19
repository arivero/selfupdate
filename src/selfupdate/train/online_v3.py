"""Pipeline-v3 online block-local learning.

The atomic event is one aligned token. Within that event blocks are visited
in forward order and every block receives its own detached local objective.
The minimum-memory dispatch backpropagates and writes after each block; the
LoRA dispatch may enter autograd once over all disconnected roots and perform
the same writes at the token boundary. No gradient is averaged across
answers, tokens, or layers, and every write lands before the next token.
Earlier causal state is either recomputed from the current weights or retained
as an immutable per-answer cache; both are discarded and rebuilt for the next
answer (and therefore the next epoch).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
import random
import threading
import time

import torch
from transformers import DynamicCache, StaticCache
from transformers.cache_utils import LinearAttentionLayer

from ..data.dataset import DistillDataset, collate_padded_items
from ..eval.teacher_output import teacher_output_eval_sums
from .blocks import NO_PREPARED_ATTENTION_MASK
from .losses import HiddenLoss
from .stop import requested_signal, stop_requested
from .telemetry import (
    ParameterDeltaTracker,
    _epoch_end_telemetry,
    _epoch_zero_telemetry,
    _flush_train_log,
)
from .ppn import (PPnExecutor, PPnPartition, StageResult, Tile,
                  boundary_volume_bytes, partition_from_config)


def _flow_keep(cfg, it, length: int, device) -> torch.Tensor | None:
    """Return the full-history keep mask, or None for non-flow controls."""
    if cfg.mask.compaction != "flow_mask":
        return None
    keep = torch.ones((1, length), dtype=torch.bool, device=device)
    for start, stop in it.t_priv or []:
        if start < length:
            keep[:, start:min(stop, length)] = False
    return keep


class _PreparedIntactCausal:
    """Mask-free q=1 views plus shared explicit masks for K>1 windows."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self._window_key = None
        self._window_mask = None

    def window(self, start: int, stop: int):
        if stop - start == 1:
            # A single cached query has no future keys in its prefix, so no
            # mask is numerically required and SDPA can take its fast path.
            return NO_PREPARED_ATTENTION_MASK
        key = (start, stop)
        mask = self._window_mask if key == self._window_key else None
        if mask is None:
            q_pos = torch.arange(start, stop, device=self.device)[:, None]
            k_pos = torch.arange(stop, device=self.device)[None, :]
            allowed = k_pos <= q_pos
            mask = torch.zeros(
                (1, 1, stop - start, stop), dtype=self.dtype,
                device=self.device)
            mask.masked_fill_(~allowed[None, None],
                              torch.finfo(self.dtype).min)
            # Layer calls for one window share this tensor. Retain only the
            # most recent window so a long answer stays O(K*T), not O(T^2).
            self._window_key = key
            self._window_mask = mask
        return mask


def _prepared_cached_masks(cfg, stack, it, position_ids, targets):
    """Build fixed answer-wide causal masks once, then return per-layer views."""
    length = position_ids.numel()
    masks = {}
    cache = {}
    intact_specs = {}
    for layer in range(1, stack.n_layers + 1):
        target = targets[layer]
        device = target.device
        dtype = target.dtype
        layer_type = (
            stack.layer_types[layer - 1]
            if layer - 1 < len(stack.layer_types) else None)
        if layer_type in ("sliding_attention", "chunked_attention"):
            raise NotImplementedError(
                f"pipeline-v3 prepared masks do not approximate {layer_type}; "
                "add the model-authoritative mask adapter first")
        full_keep = _flow_keep(cfg, it, length, device)
        if layer_type == "linear_attention":
            masks[layer] = full_keep
            continue
        if full_keep is None:
            # q=1 intact/pad-random cells need no explicit mask. K>1 stale
            # windows do: without one, cached chunk attention can see future
            # tokens within the chunk. The shared spec materializes only the
            # required K×prefix mask, not an answer-wide T² tensor.
            key = (device, dtype)
            if key not in intact_specs:
                intact_specs[key] = _PreparedIntactCausal(device, dtype)
            masks[layer] = intact_specs[key]
            continue
        key = (device, dtype, layer_type)
        if key not in cache:
            q_pos = torch.arange(length, device=device)[:, None]
            k_pos = torch.arange(length, device=device)[None, :]
            allowed = k_pos <= q_pos
            allowed = allowed[None]
            if full_keep is not None:
                allowed &= full_keep[:, None, :]
            additive = torch.zeros(
                (1, 1, length, length), dtype=dtype, device=device)
            additive.masked_fill_(
                ~allowed[:, None], torch.finfo(dtype).min)
            cache[key] = additive
        masks[layer] = cache[key]
    return masks


def _prepared_mask_row(mask, pos_index: int):
    if isinstance(mask, _PreparedIntactCausal):
        return NO_PREPARED_ATTENTION_MASK
    if mask is None or mask is NO_PREPARED_ATTENTION_MASK:
        return mask
    if mask.ndim == 2:
        return mask[:, :pos_index + 1]
    return mask[:, :, pos_index:pos_index + 1, :pos_index + 1]


def _prepared_mask_window(mask, start: int, stop: int):
    """Select query rows ``[start, stop)`` and their causal key prefix."""
    if isinstance(mask, _PreparedIntactCausal):
        return mask.window(start, stop)
    if mask is None or mask is NO_PREPARED_ATTENTION_MASK:
        return mask
    if mask.ndim == 2:
        return mask[:, :stop]
    return mask[:, :, start:stop, :stop]


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
    cache = getattr(stack, "_v3_trainable_block_params", None)
    if cache is None:
        cache = [
            [p for p in stack.block_params(index) if p.requires_grad]
            for index in range(1, stack.n_layers + 1)
        ]
        stack._v3_trainable_block_params = cache
    params = cache[layer - 1]
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


_LAYER_INDEX_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def _layer_index_tensor(device, indices: list[int], n_layers: int):
    """Cache immutable telemetry grouping metadata outside the token loop."""
    key = (str(device), tuple(indices), n_layers)
    value = _LAYER_INDEX_TENSOR_CACHE.get(key)
    if value is None:
        value = torch.tensor(indices, dtype=torch.long, device=device)
        _LAYER_INDEX_TENSOR_CACHE[key] = value
    return value


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
        ids = _layer_index_tensor(device, layer_indices, len(params_by_layer))
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


_GRAD_READY_STATE = "_selfupdate_v3_grad_ready_state"
_GRAD_READY_INSTALLED = "_selfupdate_v3_grad_ready_installed"


def _grad_ready_hook(param: torch.nn.Parameter) -> None:
    """Write one parameter and release its gradient at AccumulateGrad."""
    state = getattr(param, _GRAD_READY_STATE, None)
    if state is None:
        return
    accumulator, lr = state
    grad = param.grad
    if grad is None:
        raise RuntimeError("grad-ready hook fired without an accumulated gradient")
    with torch.no_grad():
        accumulator.add_(grad.detach().float().square().sum())
        param.add_(grad, alpha=-lr)
        param.grad = None
    setattr(param, _GRAD_READY_STATE, None)


def _arm_grad_ready(params: list[torch.nn.Parameter], lr: float) -> torch.Tensor:
    """Arm persistent post-accumulation hooks for one local backward."""
    if not params:
        raise RuntimeError("pipeline-v3 reached a block with no trainable parameters")
    accumulator = torch.zeros(
        (), dtype=torch.float32, device=params[0].device)
    for param in params:
        if getattr(param, _GRAD_READY_STATE, None) is not None:
            raise RuntimeError("grad-ready parameter was armed twice")
        if not getattr(param, _GRAD_READY_INSTALLED, False):
            param.register_post_accumulate_grad_hook(_grad_ready_hook)
            setattr(param, _GRAD_READY_INSTALLED, True)
        setattr(param, _GRAD_READY_STATE, (accumulator, lr))
    return accumulator


def _finish_grad_ready(params: list[torch.nn.Parameter],
                       accumulator: torch.Tensor) -> torch.Tensor:
    missing = [
        index for index, param in enumerate(params)
        if getattr(param, _GRAD_READY_STATE, None) is not None
    ]
    if missing:
        for param in params:
            setattr(param, _GRAD_READY_STATE, None)
        raise RuntimeError(
            f"grad-ready backward produced no gradient for parameter slots {missing}")
    return accumulator.sqrt().detach()


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
                   causal_length=None, prepared_attention_mask=None):
    """Build one isolated block-local objective and return its parameters."""
    params = _clear_block_grads(stack, layer)
    h_in = h_in.detach()
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16,
                        enabled=h_in.device.type == "cuda"):
        h_out = stack.run_block(
            layer, h_in, pos_emb, position_ids=position_ids,
            flow_keep=flow_keep, past_key_values=cache,
            use_cache=cache is not None, causal_length=causal_length,
            prepared_attention_mask=prepared_attention_mask,
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
                  causal_length=None, prepared_attention_mask=None):
    """One block forward, one local backward, one immediate parameter write."""
    loss, h_out, params = _local_forward(
        cfg, stack, loss_fn, layer, h_in, pos_emb, position_ids, target, row,
        flow_keep=flow_keep, cache=cache, causal_length=causal_length,
        prepared_attention_mask=prepared_attention_mask)
    if cfg.train.online_write_dispatch == "grad_ready":
        accumulator = _arm_grad_ready(params, cfg.train.lr)
        loss.backward()
        grad_norm = _finish_grad_ready(params, accumulator)
    else:
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
    if cfg.train.online_write_dispatch == "grad_ready":
        accumulators = [
            _arm_grad_ready(params, cfg.train.lr) for params in params_by_layer
        ]
        torch.autograd.backward(losses)
        grad_norms = [
            _finish_grad_ready(params, accumulator)
            for params, accumulator in zip(params_by_layer, accumulators)
        ]
    else:
        torch.autograd.backward(losses)
        grad_norms = [value.detach() for value in
                      _immediate_sgd_token(params_by_layer, cfg.train.lr)]
    return [loss.detach() for loss in losses], grad_norms


def _assert_independent_block_state(stack, dispatch: str) -> None:
    """Reject schedules whose dependency graph omits cross-block KV edges."""
    shared = [
        layer for layer, accepts in enumerate(
            stack._accepts_shared_kv_states, start=1) if accepts
    ]
    if shared:
        raise NotImplementedError(
            f"{dispatch} assumes block-private causal state, but blocks "
            f"{shared} expose shared_kv_states; use per_block/token-major "
            "until the model's KV-sharing DAG has a dependency-aware schedule")


def answer_wavefront_cached(cfg, stack, loss_fn, it, token_count: int, device,
                            cache: DynamicCache, *, student_ids,
                            position_ids, targets):
    """Exact online walk over the layer × known-answer anti-diagonals.

    Cell ``(L,t)`` depends only on ``(L-1,t)`` for its detached student input
    and ``(L,t-1)`` for that layer's cache/update history. Both live on the
    preceding anti-diagonal. A diagonal therefore contains at most one cell
    per block and can enter autograd as disconnected roots without stale
    weights, gradient averaging, or cross-block credit.
    """
    _assert_independent_block_state(stack, "answer_wavefront_disconnected")
    n = stack.n_layers
    if token_count <= 0:
        return [], [], 0
    frontier = {}
    full_keep = _flow_keep(cfg, it, position_ids.numel(), position_ids.device)
    prepared_masks = _prepared_cached_masks(
        cfg, stack, it, position_ids, targets)
    losses_by_layer = [[] for _ in range(n)]
    grads_by_layer = [[] for _ in range(n)]
    for diagonal in range(token_count + n - 1):
        diagonal_losses = []
        diagonal_params = []
        diagonal_cells = []
        for layer in range(1, n + 1):
            offset = diagonal - (layer - 1)
            if offset < 0 or offset >= token_count:
                continue
            pos_index = it.s0 + offset
            pos = position_ids[pos_index:pos_index + 1][None]
            if layer == 1:
                ids = student_ids[pos_index:pos_index + 1][None]
                h_in = stack.embed(ids).detach()
            else:
                h_in = frontier.pop((layer - 1, offset))
            pos_emb = stack.rope(h_in, pos)
            keep = (full_keep[:, :pos_index + 1]
                    if full_keep is not None else None)
            loss, h_out, params = _local_forward(
                cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                targets[layer][offset], 0,
                flow_keep=keep, cache=cache,
                causal_length=pos_index + 1,
                prepared_attention_mask=_prepared_mask_row(
                    prepared_masks[layer], pos_index))
            if layer < n:
                frontier[(layer, offset)] = h_out.detach()
            diagonal_losses.append(loss)
            diagonal_params.append(params)
            diagonal_cells.append((layer, offset))
        detached_losses, diagonal_grads = _finish_disconnected_token(
            cfg, diagonal_losses, diagonal_params)
        for layer, _, loss, grad in zip(
                (cell[0] for cell in diagonal_cells),
                (cell[1] for cell in diagonal_cells),
                detached_losses, diagonal_grads):
            losses_by_layer[layer - 1].append(loss)
            grads_by_layer[layer - 1].append(grad)
            _detach_cache_layer(cache, layer - 1)
    if frontier:
        raise RuntimeError(
            f"wavefront ended with {len(frontier)} unconsumed hidden states")
    mean_losses = [torch.stack(values).mean() for values in losses_by_layer]
    mean_grads = [torch.stack(values).mean() for values in grads_by_layer]
    return mean_losses, mean_grads, token_count * n


def answer_teacher_stale_windows_cached(
        cfg, stack, loss_fn, it, token_count: int, device,
        cache: DynamicCache, *, teacher_states, position_ids, targets):
    """Vectorize known-answer tokens under explicit stale weight snapshots.

    A window of K token losses is evaluated at one block-weight snapshot.
    Backward uses their unaveraged sum. Under state-free SGD, one fused
    ``W -= lr * sum(g_t)`` write is exactly the final value obtained by
    replaying those already-computed gradients one at a time; the only
    approximation is that later tokens in the window do not recompute their
    gradients after earlier writes. K=1 is exact online learning.

    Teacher ``h[L-1]`` inputs remove same-token cross-block dependencies, so
    each block can process the token window as one ordinary causal forward.
    Inputs, targets, and block credit remain strictly local and detached.
    """
    _assert_independent_block_state(stack, "teacher_stale_gradient_window")
    if any(kind in ("sliding_attention", "chunked_attention")
           for kind in stack.layer_types):
        raise NotImplementedError(
            "stale-gradient windows require authoritative non-full attention "
            "mask semantics; use K=1 until the Gemma/chunk adapter lands")
    if token_count <= 0:
        return [], [], 0, 0
    configured = cfg.train.stale_gradient_window
    width = token_count if configured == 0 else configured
    prepared_masks = _prepared_cached_masks(
        cfg, stack, it, position_ids, targets)
    position_cache = {}
    keep_cache = {}
    for layer in range(1, stack.n_layers + 1):
        h_in = teacher_states[layer - 1]
        if h_in.device not in position_cache:
            position_cache[h_in.device] = position_ids.to(h_in.device)
            keep_cache[h_in.device] = _flow_keep(
                cfg, it, position_ids.numel(), h_in.device)
    loss_sums = [None] * stack.n_layers
    # This is the norm of each window-summed gradient, normalized by its K
    # before answer averaging. It is deliberately not mislabelled as the
    # unavailable mean of K individual gradient norms.
    normalized_grad_sums = [None] * stack.n_layers
    physical_writes = 0

    for offset_start in range(0, token_count, width):
        offset_stop = min(offset_start + width, token_count)
        window_tokens = offset_stop - offset_start
        pos_start = it.s0 + offset_start
        pos_stop = it.s0 + offset_stop
        rope_cache = {}
        for layer in range(1, stack.n_layers + 1):
            params = _clear_block_grads(stack, layer)
            h_in = teacher_states[layer - 1][
                :, pos_start:pos_stop].detach()
            pos = position_cache[h_in.device][pos_start:pos_stop][None]
            rope_key = (h_in.device, h_in.dtype)
            if stack.rotary_needs_layer_type:
                pos_emb = stack.rope(h_in, pos)
            else:
                pos_emb = rope_cache.get(rope_key)
                if pos_emb is None:
                    pos_emb = stack.rope(h_in, pos)
                    rope_cache[rope_key] = pos_emb
            full_keep = keep_cache[h_in.device]
            keep = (full_keep[:, :pos_stop]
                    if full_keep is not None else None)
            with torch.autocast(
                    h_in.device.type, dtype=torch.bfloat16,
                    enabled=h_in.device.type == "cuda"):
                h_out = stack.run_block(
                    layer, h_in, pos_emb, position_ids=pos,
                    flow_keep=keep, past_key_values=cache, use_cache=True,
                    causal_length=pos_stop,
                    prepared_attention_mask=_prepared_mask_window(
                        prepared_masks[layer], pos_start, pos_stop),
                )
                view = stack.loss_view(layer, h_out)[0]
                mean_loss = loss_fn(
                    view, targets[layer][offset_start:offset_stop],
                    normed=(layer == stack.n_layers), layer=layer)
                summed_loss = mean_loss * window_tokens
            summed_loss.backward()
            summed_grad_norm = _immediate_sgd(params, cfg.train.lr)
            _detach_cache_layer(cache, layer - 1)

            detached_loss_sum = mean_loss.detach() * window_tokens
            normalized_grad_sum = summed_grad_norm.detach()
            index = layer - 1
            if loss_sums[index] is None:
                loss_sums[index] = detached_loss_sum
                normalized_grad_sums[index] = normalized_grad_sum
            else:
                loss_sums[index].add_(detached_loss_sum)
                normalized_grad_sums[index].add_(normalized_grad_sum)
            physical_writes += 1

    mean_losses = [value / token_count for value in loss_sums]
    normalized_grad_norms = [
        value / token_count for value in normalized_grad_sums]
    conceptual_writes = token_count * stack.n_layers
    return (mean_losses, normalized_grad_norms,
            conceptual_writes, physical_writes)


def answer_teacher_layer_lanes_cached(
        cfg, stack, loss_fn, it, token_count: int, device,
        cache: DynamicCache, *, teacher_states, position_ids, targets):
    """Run independent teacher-forced block lanes concurrently.

    Uncensored teacher ``h[L-1,t]`` removes the same-token dependency between
    student blocks.  A lane therefore owns one block and advances its answer
    tokens causally, writing that block after every token.  CUDA stream order
    preserves each lane's online law while host threads expose all lanes to
    the device without constructing a whole-answer graph.
    """
    _assert_independent_block_state(stack, "teacher_layer_lanes")
    if token_count <= 0:
        return [], [], 0

    layer_positions = {}
    layer_keeps = {}
    prepared_masks = _prepared_cached_masks(
        cfg, stack, it, position_ids, targets)
    for layer in range(1, stack.n_layers + 1):
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        if not params:
            raise RuntimeError(f"teacher lane {layer} has no trainable parameters")
        layer_positions[layer] = position_ids.to(params[0].device)
        layer_keeps[layer] = _flow_keep(
            cfg, it, position_ids.numel(), params[0].device)

    def run_lane(layer: int):
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        if not params:
            raise RuntimeError(f"teacher lane {layer} has no trainable parameters")
        layer_device = params[0].device
        stream = (torch.cuda.Stream(device=layer_device)
                  if layer_device.type == "cuda" else None)

        def walk():
            losses = []
            grads = []
            for offset in range(token_count):
                pos_index = it.s0 + offset
                pos = layer_positions[layer][pos_index:pos_index + 1][None]
                h_in = teacher_states[layer - 1][
                    :, pos_index:pos_index + 1].detach()
                pos_emb = stack.rope(h_in, pos)
                full_keep = layer_keeps[layer]
                keep = (full_keep[:, :pos_index + 1]
                        if full_keep is not None else None)
                loss, grad, _ = _local_update(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                    targets[layer][offset], 0, flow_keep=keep, cache=cache,
                    causal_length=pos_index + 1,
                    prepared_attention_mask=_prepared_mask_row(
                        prepared_masks[layer], pos_index))
                losses.append(loss)
                grads.append(grad)
            # Do not launch scalar accumulation kernels inside the hot loop.
            # These detached scalars are answer-local and released here.
            return torch.stack(losses).mean(), torch.stack(grads).mean()

        if stream is None:
            return walk()
        with torch.cuda.device(layer_device), torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.default_stream(layer_device))
            result = walk()
        stream.synchronize()
        return result

    # A CPU fallback remains deterministic and avoids oversubscribing host
    # kernels; the scientific campaign path is CUDA.
    if not torch.cuda.is_available():
        rows = [run_lane(layer) for layer in range(1, stack.n_layers + 1)]
    else:
        with ThreadPoolExecutor(max_workers=stack.n_layers) as pool:
            rows = list(pool.map(run_lane, range(1, stack.n_layers + 1)))
    return ([row[0] for row in rows], [row[1] for row in rows],
            token_count * stack.n_layers)


def answer_student_pipeline_lanes_cached(
        cfg, stack, loss_fn, it, token_count: int, device,
        cache: DynamicCache, *, student_ids, position_ids, targets):
    """Execute the student layer×token wavefront as bounded CUDA lanes.

    Each lane owns one block and therefore preserves its causal token/update
    order. A depth-one queue carries the detached pre-write hidden value and
    a CUDA event to the next block. The queues expose answer-wide pipeline
    concurrency without retaining answer-wide activations.
    """
    _assert_independent_block_state(stack, "answer_pipeline_lanes")
    if token_count <= 0:
        return [], [], 0
    if not torch.cuda.is_available():
        return answer_wavefront_cached(
            cfg, stack, loss_fn, it, token_count, device, cache,
            student_ids=student_ids, position_ids=position_ids,
            targets=targets)

    frontiers = [Queue(maxsize=1) for _ in range(stack.n_layers - 1)]
    cancelled = threading.Event()
    prepared_masks = _prepared_cached_masks(
        cfg, stack, it, position_ids, targets)
    layer_keeps = {}
    for layer in range(1, stack.n_layers + 1):
        params = [p for p in stack.block_params(layer) if p.requires_grad]
        if not params:
            raise RuntimeError(
                f"student pipeline lane {layer} has no trainable parameters")
        layer_keeps[layer] = _flow_keep(
            cfg, it, position_ids.numel(), params[0].device)

    def queue_put(q, value):
        while not cancelled.is_set():
            try:
                q.put(value, timeout=0.1)
                return
            except Full:
                pass
        raise RuntimeError("student pipeline cancelled while publishing state")

    def queue_get(q):
        while not cancelled.is_set():
            try:
                return q.get(timeout=0.1)
            except Empty:
                pass
        raise RuntimeError("student pipeline cancelled while awaiting state")

    def run_lane(layer: int):
        try:
            params = [p for p in stack.block_params(layer) if p.requires_grad]
            if not params:
                raise RuntimeError(
                    f"student pipeline lane {layer} has no trainable parameters")
            layer_device = params[0].device
            stream = torch.cuda.Stream(device=layer_device)
            losses = []
            grads = []
            with torch.cuda.device(layer_device), torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream(layer_device))
                for offset in range(token_count):
                    pos_index = it.s0 + offset
                    pos = position_ids[pos_index:pos_index + 1].to(
                        layer_device)[None]
                    if layer == 1:
                        ids = student_ids[pos_index:pos_index + 1].to(
                            layer_device)[None]
                        h_in = stack.embed(ids).detach()
                    else:
                        h_in, ready = queue_get(frontiers[layer - 2])
                        stream.wait_event(ready)
                        h_in.record_stream(stream)
                    pos_emb = stack.rope(h_in, pos)
                    full_keep = layer_keeps[layer]
                    keep = (full_keep[:, :pos_index + 1]
                            if full_keep is not None else None)
                    loss, grad, h_out = _local_update(
                        cfg, stack, loss_fn, layer, h_in, pos_emb, pos,
                        targets[layer][offset], 0, flow_keep=keep, cache=cache,
                        causal_length=pos_index + 1,
                        prepared_attention_mask=_prepared_mask_row(
                            prepared_masks[layer], pos_index))
                    losses.append(loss)
                    grads.append(grad)
                    if layer < stack.n_layers:
                        ready = torch.cuda.Event()
                        ready.record(stream)
                        queue_put(frontiers[layer - 1], (h_out, ready))
            stream.synchronize()
            return torch.stack(losses).mean(), torch.stack(grads).mean()
        except BaseException:
            cancelled.set()
            raise

    with ThreadPoolExecutor(max_workers=stack.n_layers) as pool:
        rows = list(pool.map(run_lane, range(1, stack.n_layers + 1)))
    return ([row[0] for row in rows], [row[1] for row in rows],
            token_count * stack.n_layers)


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


def _to_device_value(value, device):
    """Move an immutable PP boundary value to its owning stage device."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_to_device_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device_value(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to_device_value(v, device) for k, v in value.items()}
    return value


def _detach_cache_layer(cache, index: int) -> None:
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


class _TrainingLinearAttentionLayer(LinearAttentionLayer):
    """Out-of-place recurrent cache for a forward followed by backward.

    Transformers' inference cache preserves static tensor addresses with
    ``copy_``.  Qwen3.5's FLA backward saves the predecessor recurrent state,
    so overwriting that tensor with the successor before backward violates
    autograd's version contract.  Training needs the opposite ownership law:
    keep the predecessor alive for backward and publish a detached successor
    tensor for the next causal tile.
    """

    def update_conv_state(self, conv_states: torch.Tensor, **kwargs):
        # Qwen3.5 supplies the complete next convolution window on every
        # multi-token call.  Retain it without its producer graph.  Single-
        # token inference updates bypass this method; the BxK trainer avoids
        # that inference-only path for trainable tail tiles below.
        self.dtype, self.device = conv_states.dtype, conv_states.device
        self.max_batch_size = conv_states.shape[0]
        self.conv_kernel_size = conv_states.shape[-1]
        self.conv_states = conv_states.detach()
        self.is_conv_states_initialized = True
        self.has_previous_state = True
        return self.conv_states

    def update_recurrent_state(self, recurrent_states: torch.Tensor,
                               **kwargs):
        self.recurrent_states = recurrent_states.detach()
        self.is_recurrent_states_initialized = True
        return self.recurrent_states


def _training_static_cache(config, max_cache_len: int) -> StaticCache:
    """Construct StaticCache with autograd-safe Qwen3.5 recurrent layers."""
    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    for index, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionLayer):
            cache.layers[index] = _TrainingLinearAttentionLayer()
    return cache


def _bk_layer_type(stack, layer: int) -> str:
    if layer - 1 < len(stack.layer_types):
        return stack.layer_types[layer - 1]
    block = stack.blocks[layer - 1]
    return (getattr(block, "layer_type", None)
            or getattr(getattr(block, "self_attn", None), "layer_type", None)
            or "full_attention")


def _bk_gather_hidden(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return value.gather(
        1, index[:, :, None].expand(-1, -1, value.shape[-1]))


def _bk_pinned_gather_hidden(value: torch.Tensor, index: torch.Tensor,
                             rows: torch.Tensor | None = None) -> torch.Tensor:
    """Gather a host tensor directly into a reusable-transfer-friendly buffer."""
    if value.device.type != "cpu":
        raise ValueError("pinned gather source must be on CPU")
    index = index.to("cpu")
    if rows is not None:
        value = value.index_select(0, rows.to("cpu"))
    out = torch.empty(
        (*index.shape, value.shape[-1]), dtype=value.dtype,
        device="cpu", pin_memory=True)
    torch.gather(
        value, 1, index[:, :, None].expand(-1, -1, value.shape[-1]),
        out=out)
    return out


def _bk_teacher_gather_hidden(value: torch.Tensor, index: torch.Tensor,
                              rows: torch.Tensor | None = None) -> torch.Tensor:
    """Gather from either pinned-host or explicitly resident teacher input."""
    if value.device.type == "cpu":
        return _bk_pinned_gather_hidden(value, index, rows=rows)
    if rows is not None:
        value = value.index_select(0, rows.to(value.device))
    index = index.to(value.device)
    return _bk_gather_hidden(value, index)


def _bk_pack_teacher_input(items, layer: int, width: int) -> torch.Tensor:
    """Pack one full teacher block input into transient pinned storage.

    Cached teacher-hidden execution needs full-prefix ``iL=h[L-1]`` only
    while constructing block L's causal state.  Packing one layer at a time
    prevents activation sharding from accidentally retaining
    ``n_layers * B * sequence`` teacher values for the whole cohort.
    """
    values = [item.teacher_inputs[layer] for item in items]
    staged = torch.zeros(
        (len(items), width, values[0].shape[-1]), dtype=values[0].dtype,
        device="cpu", pin_memory=True)
    for row, value in enumerate(values):
        staged[row, :value.shape[0]].copy_(value)
    return staged


def _bk_certify_cached_teacher_chain(item, n_layers: int) -> dict:
    """Fail before training unless cached h[L] is exactly cached i[L+1]."""
    compared_values = 0
    for layer in range(1, n_layers):
        target = item.hidden[layer]
        next_input = item.teacher_inputs[layer + 1][
            item.t0:item.t0 + item.A]
        if target.shape != next_input.shape or not torch.equal(
                target, next_input):
            raise RuntimeError(
                f"{item.example_id}: teacher cache chain mismatch at h{layer} "
                f"-> i{layer + 1}; rebuild the full-input cache")
        compared_values += target.numel()
    return {
        "passed": True,
        "example_id": item.example_id,
        "compared_boundaries": max(0, n_layers - 1),
        "compared_values": compared_values,
        "identity": "h[L]_aligned_exactly_equals_i[L+1]_aligned",
    }


def _bk_slice_sequence(value, start: int, stop: int):
    if torch.is_tensor(value):
        return value[:, start:stop]
    if isinstance(value, tuple):
        return tuple(_bk_slice_sequence(v, start, stop) for v in value)
    if isinstance(value, list):
        return [_bk_slice_sequence(v, start, stop) for v in value]
    return value


def _bk_prefix_layout(cfg, items, batch, device):
    """Left-pad one fixed cohort without changing any user's coordinates."""
    maximum = int(batch.s0.max())
    lengths = batch.s0.to(device)
    timeline = torch.arange(maximum, device=device)[None]
    left = maximum - lengths[:, None]
    valid = timeline >= left
    source = (timeline - left).clamp_min(0).long()
    keep = valid.clone()
    if cfg.mask.compaction == "flow_mask":
        for row, it in enumerate(items):
            shift = maximum - it.s0
            for start, stop in it.t_priv or []:
                if start < it.s0:
                    keep[row, shift + start:shift + min(stop, it.s0)] = False
    positions = batch.position_ids.to(device).gather(1, source)
    return maximum, source, positions, valid, keep


def _bk_teacher_prefix_layout(batch, device):
    """Left-pad the uncensored teacher prefix in teacher coordinates."""
    lengths = batch.t0.to(device)
    maximum = int(lengths.max())
    timeline = torch.arange(maximum, device=device)[None]
    left = maximum - lengths[:, None]
    valid = timeline >= left
    source = (timeline - left).clamp_min(0).long()
    return maximum, source, source, valid, valid.clone()



def _bk_static_additive_mask(keep: torch.Tensor, query_valid: torch.Tensor,
                             query_start: int, query_stop: int,
                             dtype: torch.dtype) -> torch.Tensor:
    """Additive mask against a preallocated, full-length static cache."""
    q_pos = torch.arange(
        query_start, query_stop, device=keep.device)[:, None]
    k_pos = torch.arange(keep.shape[1], device=keep.device)[None]
    allowed = (k_pos <= q_pos)[None] & keep[:, None, :]
    allowed[:, :, 0] |= ~query_valid
    mask = torch.zeros(
        (keep.shape[0], 1, query_stop - query_start, keep.shape[1]),
        dtype=dtype, device=keep.device)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return mask


def _bk_compact_cache_rows(cache: StaticCache, live: torch.Tensor,
                           old_batch: int) -> None:
    """Drop completed independent rows from every initialized cache state."""
    for layer in cache.layers:
        for name, value in list(vars(layer).items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == old_batch:
                setattr(layer, name, value.index_select(0, live.to(value.device)))
        if hasattr(layer, "max_batch_size"):
            layer.max_batch_size = int(live.numel())


def _bk_compact_finished_rows(shard, start: int) -> None:
    """Release rows whose aligned answer ended before this tile."""
    if start == 0 or not shard["lengths_cpu"].numel():
        return
    live_cpu = torch.nonzero(
        shard["lengths_cpu"] > start, as_tuple=False).flatten()
    old_batch = int(shard["lengths_cpu"].numel())
    if live_cpu.numel() == old_batch:
        return
    live = live_cpu.to(shard["lengths"].device)
    _bk_compact_cache_rows(shard["history"], live, old_batch)
    for name in (
        "student_ids", "batch_positions", "lengths", "answer_keep",
        "source_s0", "full_keep",
    ):
        if shard[name] is not None:
            shard[name] = shard[name].index_select(0, live)
    shard["lengths_cpu"] = shard["lengths_cpu"].index_select(0, live_cpu)
    shard["target_items"] = [shard["target_items"][int(i)] for i in live_cpu]
    shard["active_rows_cpu"] = shard["active_rows_cpu"].index_select(
        0, live_cpu)
    shard["b_now"] = int(live.numel())


def _bk_bucketed_cohorts(ds, width: int, seed: int):
    """Length-tight fixed cohorts; shuffle cohort order, never refill lanes."""
    ordered = sorted(
        range(len(ds)), key=lambda index: len(ds.pairs[index].student_ids))
    cohorts = [ordered[start:start + width]
               for start in range(0, len(ordered), width)]
    rng = random.Random(seed)
    for cohort in cohorts:
        rng.shuffle(cohort)
    rng.shuffle(cohorts)
    return cohorts


def _bk_footprint_guard(cfg, stack, ds, device, layer_types, shard_users,
                        window_width):
    """Refuse a deterministic worst tile that cannot fit after model load."""
    teacher_coordinates = cfg.train.trajectory_source == "teacher_hidden"
    max_prompt = max(
        pair.t_aligned.start if teacher_coordinates else pair.s_aligned.start
        for pair in ds.pairs)
    max_answer = max(pair.aligned_len for pair in ds.pairs)
    # One masked sentinel cache slot lets a final logical q=1 tile execute as
    # a two-query training chunk instead of entering Qwen3.5's inference-only
    # in-place recurrent kernels.
    max_total = max_prompt + max_answer + 1
    text = stack.text_config
    hidden = int(getattr(text, "hidden_size"))
    heads = int(getattr(text, "num_attention_heads", 1))
    kv_heads = int(getattr(text, "num_key_value_heads", heads))
    head_dim = int(getattr(text, "head_dim", hidden // heads))
    full_layers = sum(kind == "full_attention" for kind in layer_types)
    execution_bytes = 2 if cfg.train.lora.enabled else 4
    target_bytes_per_value = 2
    kv_bytes = (
        shard_users * full_layers * max_total * 2 * kv_heads * head_dim
        * execution_bytes)
    target_bytes = (
        stack.n_layers * shard_users * window_width * hidden
        * target_bytes_per_value)
    mask_bytes = (
        shard_users * min(cfg.train.prefill_query_chunk, max_prompt)
        * max_total * execution_bytes)
    # Conservative current-block backward workspace. Only one local block's
    # graph is alive, so this does not scale with depth.
    activation_bytes = (
        shard_users * window_width * hidden * execution_bytes * 20)
    # Evaluation-only CE/KL decodes student and teacher final states in
    # bounded 256-row chunks. Conservatively account for logits plus fp32
    # log-probabilities for both sides.
    vocab = int(stack.lm_head.weight.shape[0])
    output_eval_rows = min(256, cfg.train.micro_batch * window_width)
    output_eval_bytes = output_eval_rows * vocab * 4 * 4
    # Full-weight immediate SGD materializes gradients only for the current
    # block. LoRA makes this negligible; fp32 full-FT does not. Account for
    # the largest trainable block explicitly after model load so the launch
    # guard sees the real remaining margin.
    current_block_grad_bytes = max(
        (sum(param.numel() * param.element_size()
             for param in stack.block_params(layer)
             if param.requires_grad)
         for layer in range(1, stack.n_layers + 1)),
        default=0,
    )
    required = (kv_bytes + target_bytes + mask_bytes + activation_bytes
                + output_eval_bytes + current_block_grad_bytes)
    free, total = torch.cuda.mem_get_info(device)
    allowed = int(free * 0.80)
    evidence = {
        "max_prompt": max_prompt,
        "prompt_coordinate_source": (
            "teacher_t0" if teacher_coordinates else "student_s0"),
        "max_answer": max_answer,
        "max_total": max_total,
        "full_attention_layers": full_layers,
        "execution_bytes_per_value": execution_bytes,
        "estimated_kv_bytes": kv_bytes,
        "estimated_target_bytes": target_bytes,
        "estimated_mask_bytes": mask_bytes,
        "estimated_activation_bytes": activation_bytes,
        "estimated_output_eval_bytes": output_eval_bytes,
        "estimated_current_block_grad_bytes": current_block_grad_bytes,
        "estimated_peak_increment_bytes": required,
        "cuda_free_bytes_after_model_load": free,
        "cuda_total_bytes": total,
        "guard_fraction": 0.80,
    }
    if required > allowed:
        raise MemoryError(
            "causal_bk deterministic footprint guard rejected launch: "
            f"estimated incremental {required / 2**30:.2f} GiB exceeds "
            f"80% of current free VRAM ({free / 2**30:.2f} GiB); reduce "
            "activation_shard_users/prefill_query_chunk/K")
    return evidence


def _bk_prepare_cohort_shards(cfg, stack, items, device, teacher,
                              teacher_hidden, layer_types, shard_users,
                              window_width, execution_dtype):
    """Build fixed B-user causal state in bounded activation shards.

    The shard boundary is purely an execution/memory boundary.  A later tile
    visits all shards at the same pre-write block weights, accumulates their
    unaveraged gradients, and writes that block once.  Thus the logical
    B×K update remains intact even though no backward graph spans all B rows.
    """
    n = stack.n_layers
    shards = []
    for first in range(0, len(items), shard_users):
        shard_items = items[first:first + shard_users]
        # v3.2 collates token metadata only. Full n-layer targets remain in
        # their mmap-backed Items and are copied into a pinned K-window below.
        batch = collate_padded_items(shard_items, include_hidden=False)
        b_now = len(shard_items)
        # Teacher-hidden tiles never consume the censored student token stream.
        # Keep those tensors off stage 0; answer-eval ids remain in target_items
        # on host and are staged only for the final output metric.
        student_ids = (
            None if teacher_hidden else batch.student_ids.to(device))
        batch_positions = (
            None if teacher_hidden else batch.position_ids.to(device))
        teacher_input = None
        online_teacher_inputs = None
        if teacher_hidden:
            if teacher is not None:
                online_teacher_inputs = teacher.full_inputs_pinned_batch(
                    batch, device)
                teacher_width = online_teacher_inputs[0].shape[1]
            else:
                # Explicit full-prefix cache: each layer is packed only while
                # its prompt state is built below.  Padding follows every real
                # teacher row and is never indexed by valid prefix/tile maps.
                teacher_width = max(
                    next(iter(it.teacher_inputs.values())).shape[0]
                    for it in shard_items)
        (prompt_length, prefix_index, prefix_positions, prefix_valid,
         prefix_keep) = (
            _bk_teacher_prefix_layout(batch, device) if teacher_hidden else
            _bk_prefix_layout(cfg, shard_items, batch, device))
        lengths_cpu = batch.A.clone()
        lengths = lengths_cpu.to(device)
        max_answer = int(lengths_cpu.max())
        timeline = torch.arange(max_answer, device=device)[None]
        answer_keep = timeline < lengths[:, None]
        sentinel_keep = torch.zeros(
            (b_now, 1), dtype=torch.bool, device=device)
        full_keep = torch.cat(
            (prefix_keep, answer_keep, sentinel_keep), dim=1)
        max_cache_len = full_keep.shape[1]
        history = _training_static_cache(
            config=stack.text_config, max_cache_len=max_cache_len)
        # Pin full-attention cache storage to the execution dtype before any
        # update. Linear-attention layers own a different recurrent state
        # shape and retain their model-authoritative lazy initialization.
        text = stack.text_config
        kv_heads = int(getattr(
            text, "num_key_value_heads",
            getattr(text, "num_attention_heads", 1)))
        heads = int(getattr(text, "num_attention_heads", kv_heads))
        head_dim = int(getattr(
            text, "head_dim", int(getattr(text, "hidden_size")) // heads))
        for layer_index, (cache_layer, layer_type) in enumerate(
                zip(history.layers, layer_types), start=1):
            if layer_type == "full_attention" and not cache_layer.is_initialized:
                block_param = next(stack.blocks[layer_index - 1].parameters(), None)
                block_device = (block_param.device if block_param is not None
                                else torch.device(device))
                empty_kv = torch.empty(
                    (b_now, kv_heads, 0, head_dim), dtype=execution_dtype,
                    device=block_device)
                cache_layer.lazy_initialization(empty_kv, empty_kv)
        if not teacher_hidden:
            prefix_ids = student_ids.gather(1, prefix_index)
            prefix_hidden = stack.embed(prefix_ids)
            prefix_rope = stack.rope(prefix_hidden, prefix_positions)
        with torch.no_grad(), torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=cfg.train.lora.enabled):
            h = prefix_hidden if not teacher_hidden else None
            for layer in range(1, n + 1):
                if teacher_hidden:
                    layer_teacher_input = (
                        online_teacher_inputs[layer - 1]
                        if online_teacher_inputs is not None else
                        _bk_pack_teacher_input(
                            shard_items, layer, teacher_width))
                    # i1=h0 is the only full teacher input needed after
                    # prefill.  Later answer inputs are exactly the preceding
                    # block's already-staged local-loss target h[L-1].
                    if layer == 1:
                        teacher_input = layer_teacher_input
                    if online_teacher_inputs is not None:
                        # Keep i1 through the answer walk via teacher_input;
                        # release every other online full input as soon as its
                        # block-local prompt history has been constructed.
                        online_teacher_inputs[layer - 1] = None
                    block_device = next(
                        stack.blocks[layer - 1].parameters()).device
                    layer_positions = prefix_positions.to(
                        block_device, non_blocking=True)
                    h_in = _bk_teacher_gather_hidden(
                        layer_teacher_input, prefix_index
                    ).to(block_device, non_blocking=True)
                    layer_rope = stack.rope(h_in, layer_positions)
                    layer_valid = prefix_valid.to(block_device)
                    layer_keep_full = full_keep.to(block_device)
                else:
                    h_in = h
                    layer_positions = prefix_positions
                    layer_rope = prefix_rope
                    layer_valid = prefix_valid
                    layer_keep_full = full_keep
                pieces = [] if not teacher_hidden else None
                for q0 in range(0, prompt_length, cfg.train.prefill_query_chunk):
                    q1 = min(q0 + cfg.train.prefill_query_chunk, prompt_length)
                    query_valid = layer_valid[:, q0:q1]
                    query_keep = layer_keep_full[:, q0:q1]
                    layer_mask = (
                        query_keep
                        if layer_types[layer - 1] == "linear_attention" else
                        _bk_static_additive_mask(
                            layer_keep_full, query_valid, q0, q1,
                            execution_dtype))
                    piece = stack.run_block(
                        layer, h_in[:, q0:q1],
                        _bk_slice_sequence(layer_rope, q0, q1),
                        position_ids=layer_positions[:, q0:q1],
                        flow_keep=query_keep,
                        past_key_values=history, use_cache=True,
                        causal_length=max_cache_len,
                        prepared_attention_mask=layer_mask)
                    if teacher_hidden:
                        # The immutable cache supplies the next block's input;
                        # only this block's detached causal history survives.
                        del piece
                    else:
                        pieces.append(piece)
                _detach_cache_layer(history, layer - 1)
                if teacher_hidden:
                    if layer > 1:
                        del layer_teacher_input
                else:
                    h_out = torch.cat(pieces, dim=1)
                    h = h_out.detach()
        if online_teacher_inputs is not None:
            del online_teacher_inputs
        target_h = shard_items[0].hidden[1].shape[-1]
        target_dtype = shard_items[0].hidden[1].dtype
        shards.append({
            "b_now": b_now,
            "student_ids": student_ids,
            "batch_positions": batch_positions,
            "teacher_input": teacher_input,
            "prompt_length": prompt_length,
            "full_keep": full_keep,
            "history": history,
            "lengths": lengths,
            "lengths_cpu": lengths_cpu,
            "answer_keep": answer_keep,
            "source_s0": (
                batch.t0 if teacher_hidden else batch.s0
            ).to(device)[:, None],
            "source_width": (
                teacher_input.shape[1] if teacher_hidden else
                batch_positions.shape[1]),
            "max_answer": max_answer,
            "target_items": list(shard_items),
            "active_rows_cpu": torch.arange(b_now),
            "target_staging": torch.empty(
                (n, b_now, window_width, target_h), dtype=target_dtype,
                device="cpu", pin_memory=True),
            "teacher_staging": (
                torch.empty(
                    (b_now, window_width, target_h),
                    dtype=teacher_input.dtype, device="cpu",
                    pin_memory=True)
                if teacher_hidden else None),
        })
        # These are prefill-only tensors. Persistent causal state is held in
        # history; retaining the masks would turn the shard mechanism into an
        # answer-length activation cache.
        del prefix_index, prefix_positions, prefix_valid
        if not teacher_hidden:
            del prefix_rope
        del h_in, pieces
        if not teacher_hidden:
            del h_out, prefix_ids, prefix_hidden, h
    return shards


def _bk_prepare_shard_tile(shard, start, width, n, device, stack,
                           teacher_hidden, execution_dtype, *,
                           compact_finished=True, isolated_target=False):
    """Materialize one shard's current K-window state, or return None."""
    if compact_finished:
        _bk_compact_finished_rows(shard, start)
    if not shard["b_now"] or start >= shard["max_answer"]:
        return None
    logical_stop = min(start + width, shard["max_answer"])
    stop = logical_stop
    # Qwen3.5 dispatches q=1 through fused inference kernels that mutate
    # recurrent state in place.  A final one-token training tile therefore
    # carries one masked, loss-free dummy query through the causal chunk
    # kernel.  Causality leaves the real first output unchanged, and no later
    # tile consumes the dummy successor state.
    if logical_stop - start == 1 and width > 1:
        stop += 1
    tile_width = stop - start
    offsets = torch.arange(start, stop, device=device)[None]
    query_valid = offsets < shard["lengths"][:, None]
    valid_widths = (shard["lengths_cpu"] - start).clamp(
        min=0, max=tile_width)
    cells = int(valid_widths.sum())
    if not cells:
        return None
    # Target staging remains bounded by the current K window. The shard
    # boundary exists for backward activations, not for target averaging.
    if isolated_target:
        # A wavefront stage may still consume tile t while stage 0 admits
        # tile t+1.  Do not recycle the serial target buffer until the final
        # stage has drained the tile.
        staging = torch.empty(
            (n, shard["b_now"], tile_width, shard["target_staging"].shape[-1]),
            dtype=shard["target_staging"].dtype, device="cpu", pin_memory=True)
    else:
        staging = shard["target_staging"][
            :, :shard["b_now"], :tile_width]
    staging.zero_()
    counts = valid_widths.tolist()
    if all(count == tile_width for count in counts):
        # Almost every bucket tile is rectangular. A layer-wise stack writes
        # the same bytes with n C++ calls instead of B*n tiny Python copy_
        # dispatches, so the single tile producer can keep PP3+ fed.
        for layer in range(1, n + 1):
            torch.stack([
                it.hidden[layer][start:start + tile_width]
                for it in shard["target_items"]
            ], dim=0, out=staging[layer - 1])
    else:
        # Finished and final-tile rows have at most K distinct valid widths.
        # Group them so the ragged path is bounded by K*n C++ copies rather
        # than falling back to B*n Python dispatches as users complete.
        rows_by_width = {}
        for row, count in enumerate(counts):
            if count:
                rows_by_width.setdefault(count, []).append(row)
        for count, rows in rows_by_width.items():
            row_index = torch.tensor(rows, dtype=torch.long)
            for layer in range(1, n + 1):
                packed = torch.stack([
                    shard["target_items"][row].hidden[layer][
                        start:start + count]
                    for row in rows
                ])
                staging[layer - 1, :, :count].index_copy_(
                    0, row_index, packed)
    # Wavefront tiles may remain live until the final stage drains them. Keep
    # their full-depth targets in pinned host storage and transfer only the
    # current owned layer in the stage callback; otherwise every in-flight
    # tile replicates n layers of targets on stage 0 and fragments its VRAM.
    window_targets = (
        staging if isolated_target else staging.to(device, non_blocking=True))
    source_index = (shard["source_s0"] + offsets).clamp_max(
        shard["source_width"] - 1)
    query_positions = (
        source_index.clone() if teacher_hidden else
        shard["batch_positions"].gather(1, source_index))
    if stop != logical_stop:
        # The sentinel has a real, in-range cache position even though its
        # token embedding is an ignored duplicate of the final source token.
        query_positions[:, -1] = query_positions[:, -2] + 1
    full_mask = _bk_static_additive_mask(
        shard["full_keep"], query_valid,
        shard["prompt_length"] + start,
        shard["prompt_length"] + stop, execution_dtype)
    valid_index_cpu = torch.cat([
        row * tile_width + torch.arange(int(count))
        for row, count in enumerate(valid_widths)
        if int(count)
    ])
    valid_index = valid_index_cpu.to(device, non_blocking=True)
    eval_index_cpu, eval_target_ids_cpu = _bk_answer_eval_coordinates(
        shard, start, stop)
    eval_index = eval_index_cpu.to(device, non_blocking=True)
    eval_target_ids = eval_target_ids_cpu.to(device, non_blocking=True)
    if teacher_hidden:
        # Preserve the architecture adapter's exact RoPE call: some adapters
        # use only shape/dtype, but the contract does not require rotary input
        # to be value-independent. Keep this same staged i1 tensor for block 1
        # so it is neither gathered nor transferred twice.
        teacher_staging = (
            torch.empty(
                (shard["b_now"], tile_width,
                 shard["teacher_staging"].shape[-1]),
                dtype=shard["teacher_staging"].dtype,
                device="cpu", pin_memory=True)
            if isolated_target else shard["teacher_staging"])
        h = _bk_stage_first_teacher_tile(
            shard, source_index, device, teacher_staging).detach()
        query_rope = stack.rope(h, query_positions)
    else:
        teacher_staging = None
        query_ids = shard["student_ids"].gather(1, source_index)
        h = stack.embed(query_ids)
        query_rope = stack.rope(h, query_positions)
    return {
        "shard": shard,
        "stop": stop,
        "cells": cells,
        "window_targets": window_targets,
        "source_index": source_index,
        "query_positions": query_positions,
        "query_valid": query_valid,
        "valid_index": valid_index,
        "eval_index": eval_index,
        "eval_target_ids": eval_target_ids,
        "full_mask": full_mask,
        "query_rope": query_rope,
        "h": h,
        "teacher_staging": teacher_staging,
    }


def _bk_answer_eval_coordinates(shard, start: int, stop: int):
    """Return flattened tile rows and next-token ids for the whole answer.

    The aligned cache span is ``shared_mid + answer``.  Therefore aligned
    offset ``answer_offset - 1`` predicts the first answer token, while the
    last aligned row has no next token and is excluded.  Prompt/shared-mid
    targets, finished-row padding, and final padding never enter the metric.
    """
    tile_width = stop - start
    flat_rows = []
    target_ids = []
    for row, item in enumerate(shard["target_items"]):
        answer_offset = item.ans0 - item.s0
        if not 1 <= answer_offset <= item.A:
            raise RuntimeError(
                f"{item.example_id}: full-answer output evaluation requires "
                "a non-empty shared_mid predictor before the answer")
        first_q = max(start, answer_offset - 1)
        last_q_exclusive = min(stop, item.A - 1)
        if first_q >= last_q_exclusive:
            continue
        q = torch.arange(first_q, last_q_exclusive, dtype=torch.long)
        flat_rows.append(row * tile_width + (q - start))
        target_positions = item.s0 + q + 1
        target_ids.append(item.student_ids.index_select(0, target_positions))
    if not flat_rows:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone()
    return torch.cat(flat_rows), torch.cat(target_ids)


def _bk_stage_first_teacher_tile(shard, source_index: torch.Tensor,
                                 device: torch.device,
                                 staging: torch.Tensor | None = None
                                 ) -> torch.Tensor:
    """Gather i1=h0; deeper inputs reuse the preceding loss target."""
    value = shard["teacher_input"]
    source_cpu = source_index.to("cpu")
    source = value.index_select(0, shard["active_rows_cpu"])
    buffer = staging if staging is not None else shard["teacher_staging"]
    out = buffer[
        :shard["b_now"], :source_cpu.shape[1]]
    torch.gather(
        source, 1,
        source_cpu[:, :, None].expand(-1, -1, source.shape[-1]), out=out)
    return out.to(device, non_blocking=True)


def _bk_process_ppn_stage(cfg, stack, loss_fn, tile: Tile, states,
                          first_layer: int, last_layer: int, device,
                          teacher_hidden: bool, layer_types, epoch_lr: float):
    """Process the owned contiguous blocks of one PPn tile.

    ``states`` contains one execution shard per activation-memory shard.  A
    stage changes only its own blocks and its detached outgoing ``h`` values;
    the PPn executor detaches the cross-stage packet before the next stage
    admits it.  The returned scalars are aggregated at the cohort boundary,
    never inside the stage callback.
    """
    cells = int(tile.users)
    n = stack.n_layers
    first_params = stack.block_params(first_layer)
    if not first_params:
        raise RuntimeError(f"PPn block {first_layer} has no parameters")
    device = first_params[0].device
    # Tile construction happens on stage 0.  Only the detached activation and
    # small immutable indexing/RoPE inputs cross a boundary; layer targets are
    # copied one owned layer at a time below instead of replicating the full
    # depth target tensor on every stage.
    for state in states:
        for key in ("source_index", "query_positions", "query_valid",
                    "valid_index", "eval_index", "eval_target_ids",
                    "full_mask", "query_rope"):
            state[key] = _to_device_value(state[key], device)
        if not teacher_hidden or first_layer == 1:
            state["h"] = _to_device_value(state["h"], device)
        else:
            # i1 belongs only to the stage owning block 1. Independent later
            # stages obtain their first input from the preceding hidden target.
            state["h"] = None
        if teacher_hidden:
            if cfg.train.teacher_hidden_source == "gpu_cache":
                # Cache only this tile's targets on their owning card.  Include
                # h[first-1] at a stage boundary because it is the immutable
                # input of the first owned block; never place the full-depth
                # cohort cache on stage 0 (or on any GPU).
                target_base = max(0, first_layer - 2)
                state["_teacher_target_base"] = target_base
                state["_teacher_targets_gpu"] = state["window_targets"][
                    target_base:last_layer].to(device, non_blocking=True)
            if first_layer == 1:
                state["_teacher_next"] = state["h"].detach()
            elif cfg.train.teacher_hidden_source == "gpu_cache":
                state["_teacher_next"] = state[
                    "_teacher_targets_gpu"][0].detach()
            else:
                state["_teacher_next"] = state["window_targets"][
                    first_layer - 2].to(device, non_blocking=True).detach()
    loss_sums = []
    grad_sums = []
    ce_sum = None
    kl_sum = None
    eval_tokens = 0
    for layer in range(first_layer, last_layer + 1):
        params = _clear_block_grads(stack, layer)
        loss_sum = torch.zeros((), dtype=torch.float32,
                              device=next(iter(params)).device)
        eval_student_parts = []
        eval_teacher_parts = []
        eval_id_parts = []
        for state in states:
            shard = state["shard"]
            h_in = (state["_teacher_next"]
                    if teacher_hidden else state["h"].detach())
            layer_mask = (
                state["query_valid"]
                if layer_types[layer - 1] == "linear_attention"
                else state["full_mask"])
            with torch.autocast(
                    device.type, dtype=torch.bfloat16,
                    enabled=cfg.train.lora.enabled):
                h_out = stack.run_block(
                    layer, h_in, state["query_rope"],
                    position_ids=state["query_positions"],
                    flow_keep=state["query_valid"],
                    past_key_values=shard["history"], use_cache=True,
                    causal_length=shard["full_keep"].shape[1],
                    prepared_attention_mask=layer_mask)
                flat_view = stack.loss_view(layer, h_out).reshape(
                    -1, h_out.shape[-1])
                valid_view = flat_view.index_select(0, state["valid_index"])
                target_all = (
                    state["_teacher_targets_gpu"][
                        layer - 1 - state["_teacher_target_base"]]
                    if teacher_hidden
                    and cfg.train.teacher_hidden_source == "gpu_cache" else
                    state["window_targets"][layer - 1].to(
                        device, non_blocking=True))
                flat_target = target_all.reshape(-1, target_all.shape[-1])
                target = flat_target.index_select(0, state["valid_index"])
                mean_loss = loss_fn(
                    valid_view, target, normed=(layer == n), layer=layer)
                summed_loss = mean_loss * state["cells"]
            summed_loss.backward()
            if layer == n and state["eval_index"].numel():
                eval_student_parts.append(
                    flat_view.detach().index_select(0, state["eval_index"]))
                eval_teacher_parts.append(
                    flat_target.detach().index_select(0, state["eval_index"]))
                eval_id_parts.append(state["eval_target_ids"])
            loss_sum.add_(
                (mean_loss.detach().float() * state["cells"]).to(
                    loss_sum.device))
            _detach_cache_layer(shard["history"], layer - 1)
            if teacher_hidden:
                if layer < last_layer:
                    # h[L], staged for block L's loss, is exactly i[L+1].
                    # Keep this detached device tensor for the next local block
                    # instead of performing a second cache read and H2D copy.
                    state["_teacher_next"] = target_all.detach()
            else:
                state["h"] = h_out.detach()
            del h_in, h_out, flat_view, valid_view, target_all
            del flat_target, target, summed_loss, mean_loss
        if eval_id_parts:
            ce_sum, kl_sum, eval_tokens, _sm_unused, _tm_unused = teacher_output_eval_sums(
                torch.cat(eval_student_parts),
                torch.cat(eval_teacher_parts),
                torch.cat(eval_id_parts),
                stack.lm_head,
                chunk_rows=256,
            )
        grad = _immediate_sgd(params, epoch_lr)
        loss_sums.append(loss_sum)
        grad_sums.append(grad.detach())
    if teacher_hidden:
        for state in states:
            state.pop("_teacher_next", None)
            state.pop("_teacher_target_base", None)
            state.pop("_teacher_targets_gpu", None)
    return StageResult(states, {
        "loss_sums": loss_sums,
        "grad_sums": grad_sums,
        "ce_sum": ce_sum,
        "kl_sum": kl_sum,
        "eval_tokens": eval_tokens,
        "cells": cells,
        "physical_writes": last_layer - first_layer + 1,
        "boundary_bytes": int(tile.metadata.get("boundary_bytes", 0)),
    })


def _bk_run_ppn_cohort(cfg, stack, loss_fn, shards, max_answer, n, device,
                       teacher_hidden, layer_types, epoch_lr, partition,
                       *, execution: str):
    """Run every tile in one cohort through a serial or wavefront PPn."""
    stage_metrics: dict[int, dict[int, Mapping[str, object]]] = {}
    metrics_lock = threading.Lock()

    def callback(context, tile, states):
        if execution == "independent":
            # Each stage mutates only its shallow packet (device placement and
            # owner-local target view).  The immutable host targets and the
            # causal-history object are shared; cache layers inside that object
            # remain disjoint by the unchanged partition ownership.
            states = [{**state} for state in states]
        return _bk_process_ppn_stage(
            cfg, stack, loss_fn, tile, states, context.first_block,
            context.last_block, device, teacher_hidden, layer_types, epoch_lr)

    def observe(stage, tile, result):
        with metrics_lock:
            stage_metrics.setdefault(tile.index, {})[stage] = result.metrics

    def tile_source():
        tile_index = 0
        for start in range(0, max_answer, cfg.train.stale_gradient_window):
            tile_states = [
                state for shard in shards
                if (state := _bk_prepare_shard_tile(
                    shard, start, cfg.train.stale_gradient_window, n, device,
                    stack, teacher_hidden, (
                        torch.bfloat16 if cfg.train.lora.enabled else torch.float32),
                    compact_finished=False, isolated_target=True))
            ]
            cells = sum(state["cells"] for state in tile_states)
            if not cells:
                continue
            live_users = sum(
                int((shard["lengths_cpu"] > start).sum()) for shard in shards)
            yield Tile(
                tile_index, tile_states, users=cells,
                width=cfg.train.stale_gradient_window,
                metadata={
                    "start": start,
                    "live_users": live_users,
                    "boundary_bytes": (
                        0 if execution == "independent" else
                        boundary_volume_bytes(
                            live_users=live_users,
                            tile_width=min(
                                cfg.train.stale_gradient_window,
                                max(0, max_answer - start)),
                            hidden_size=int(getattr(
                                stack.text_config, "hidden_size", 0)),
                            element_size=2 if cfg.train.lora.enabled else 4)),
                },
            )
            tile_index += 1

    executor = PPnExecutor(
        partition,
        [callback] * partition.stages,
        detach_boundary=_detach_value,
        queue_depth=1,
        telemetry_callback=observe,
    )
    results = executor.run(tile_source(), execution=execution)
    if not results and any(shard["max_answer"] for shard in shards):
        raise RuntimeError("PPn admitted no nonempty BxK tiles")
    aggregate_loss = [None] * n
    aggregate_grad = [None] * n
    total_cells = 0
    total_physical_writes = 0
    total_ce = None
    total_kl = None
    total_eval_tokens = 0
    total_boundary_bytes = 0
    for result in results:
        # Stage metrics are collected separately because only the final stage
        # has output-evaluation values.  All tensors remain on their owning
        # stage until this cohort boundary.
        tile_index = next(
            index for index, values in stage_metrics.items()
            if values.get(partition.stages - 1) is result.metrics)
        values_by_stage = stage_metrics[tile_index]
        for stage, metrics in values_by_stage.items():
            first, last = partition.ranges[stage]
            for offset, value in enumerate(metrics["loss_sums"]):
                layer = first - 1 + offset
                if aggregate_loss[layer] is None:
                    aggregate_loss[layer] = value.detach()
                    aggregate_grad[layer] = metrics["grad_sums"][offset].detach()
                else:
                    aggregate_loss[layer].add_(value.to(aggregate_loss[layer].device))
                    aggregate_grad[layer].add_(
                        metrics["grad_sums"][offset].to(
                            aggregate_grad[layer].device))
            if stage == partition.stages - 1:
                if metrics["ce_sum"] is not None:
                    total_ce = (metrics["ce_sum"].detach() if total_ce is None
                                else total_ce + metrics["ce_sum"].to(total_ce.device))
                    total_kl = (metrics["kl_sum"].detach() if total_kl is None
                                else total_kl + metrics["kl_sum"].to(total_kl.device))
                    total_eval_tokens += int(metrics["eval_tokens"])
            total_physical_writes += int(metrics["physical_writes"])
        final_states = result.payload
        for state in final_states:
            del state["window_targets"], state["query_rope"]
            del state["valid_index"], state["full_mask"], state["h"]
            del state["eval_index"], state["eval_target_ids"]
            del state["source_index"], state["query_positions"]
            del state["query_valid"], state["teacher_staging"]
        total_cells += int(result.metrics["cells"])
        total_boundary_bytes += int(result.metrics.get("boundary_bytes", 0))
    if any(value is None for value in aggregate_loss):
        raise RuntimeError("PPn did not produce a loss for every owned block")
    return {
        "loss_sums": aggregate_loss,
        "grad_sums": aggregate_grad,
        "cells": total_cells,
        "physical_writes": total_physical_writes,
        "ce_sum": total_ce,
        "kl_sum": total_kl,
        "eval_tokens": total_eval_tokens,
        "boundary_bytes": total_boundary_bytes,
        "executor": executor,
    }


def train_bk_v32(cfg, stack, tok, log, cache, teacher=None) -> bool:
    """Full dataset-v5 B-user × K-token pipeline-v3.2 training.

    One cohort represents B simultaneous conversations. Its causal state is
    retained until every answer in that cohort finishes; completed rows stay
    in place with zero loss/gradient and are never replaced. Each block gets
    one unaveraged sum over the valid B×K cells and writes immediately.
    """
    if cfg.train.pipeline_revision != "3.2":
        raise RuntimeError(
            "the superseded v3.1 causal_bk runtime was removed; recover it "
            "from git commit ba0d30b for archaeology, or set up a reviewed "
            "pipeline_revision=3.2 experiment")
    if cfg.train.max_steps:
        raise NotImplementedError(
            "causal_bk production runs currently require full epochs; "
            "max_steps would split a fixed cohort ambiguously")
    layer_types = [_bk_layer_type(stack, layer)
                   for layer in range(1, stack.n_layers + 1)]
    unsupported = sorted(set(layer_types) - {
        "full_attention", "linear_attention",
    })
    if unsupported:
        raise NotImplementedError(
            f"causal_bk lacks authoritative state semantics for {unsupported}")
    teacher_hidden = cfg.train.trajectory_source == "teacher_hidden"
    cached_teacher_hidden = (
        teacher_hidden and cfg.train.teacher_hidden_source in (
            "cpu_cache", "gpu_cache"))
    if teacher_hidden and not cached_teacher_hidden and teacher is None:
        raise ValueError("online causal_bk teacher_hidden requires a teacher")
    if cached_teacher_hidden and not cache.has_full_teacher_inputs:
        raise ValueError(
            "cached causal_bk teacher_hidden requires full teacher inputs")

    n = stack.n_layers
    partition = partition_from_config(cfg, num_blocks=n)
    if partition.stages > 1 and stack.block_devices is None:
        raise ValueError(
            "a multi-stage PPn execution needs explicit model.pipeline_splits "
            "so block ownership is unambiguous")
    # The v3.2 reference names a single model device.  PPn's input staging,
    # epoch accumulators, and footprint guard instead belong to the first
    # owning stage.  This matters for sparse physical placement (for example
    # stages [1, 3]): checking cuda:0 can reject a healthy PPn run merely
    # because an unrelated job occupies that card.
    device = torch.device(
        "cuda", partition.physical_devices[0]) if (
            partition.stages > 1 and partition.physical_devices
        ) else torch.device(cfg.model.device)
    B = cfg.train.micro_batch
    K = cfg.train.stale_gradient_window
    activation_shard_users = cfg.train.activation_shard_users
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    trainable_dtypes = sorted({str(param.dtype)
                               for layer in range(1, n + 1)
                               for param in stack.block_params(layer)
                               if param.requires_grad})
    if not cfg.train.lora.enabled and trainable_dtypes != ["torch.float32"]:
        raise RuntimeError(
            "causal_bk full-weight immediate SGD requires fp32 trainable "
            f"block weights, got {trainable_dtypes}; reduced-precision direct "
            "writes can round the configured learning rate to zero")
    execution_dtype = (
        torch.bfloat16 if cfg.train.lora.enabled else torch.float32)
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=list(range(1, n + 1)),
        with_teacher_ids=teacher_hidden and not cached_teacher_hidden,
        with_teacher_inputs=cached_teacher_hidden,
        pad_random=(cfg.mask.compaction == "pad_random"),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        item_cache_items=cfg.cache.item_cache_items,
    )
    teacher_chain_tripwire = (
        _bk_certify_cached_teacher_chain(ds[0], n)
        if cached_teacher_hidden else None)
    full_training_answer_tokens = sum(
        pair.s_answer.stop - pair.s_answer.start for pair in ds.pairs)
    footprint = _bk_footprint_guard(
        cfg, stack, ds, device, layer_types, activation_shard_users, K)
    delta_tracker = ParameterDeltaTracker(stack)
    started = time.time()
    token_events = conceptual_writes = physical_writes = answers_seen = 0
    standard_baseline = _epoch_zero_telemetry(
        cfg, stack, tok, log, started_at=started)
    delta_tracker.log(log, epoch=0, phase="epoch0", started_at=started)
    log.log(
        kind="pipeline_v32_contract",
        layerwise_project_version=getattr(
            cfg, "layerwise_project_version", "3.4"),
        pp_execution=cfg.train.pp_execution,
        physical_gpu_mapping=list(
            partition.physical_devices or range(partition.stages)),
        ordered_block_ranges=[list(item) for item in partition.ranges],
        partition_profile_id=partition.profile_id or None,
        pp_dependency_graph=(
            "O[s,t]->O[s,t+1] only; teacher inputs are immutable stage-local"
            if cfg.train.pp_execution == "independent" else
            "O[s,t]->O[s,t+1] and O[s,t]->O[s+1,t]"),
        pp_boundary_activation=(
            "none_cached_teacher_input" if cfg.train.pp_execution == "independent"
            else "detached_exact_copy"),
        frozen_output_head_replica=bool(getattr(
            stack, "pp_frozen_output_head_replica", False)),
        pp_queue_depth=1,
        pp_write_semantics="immediate_state_free_sgd_before_next_tile",
        B_simultaneous_users=B,
        activation_shard_users=activation_shard_users,
        prefill_parallel_shards=cfg.train.prefill_parallel_shards,
        prefill_query_chunk=cfg.train.prefill_query_chunk,
        K_context_tokens=K,
        lane_refill=False,
        finished_rows="compacted_without_replacement_at_tile_boundaries",
        gradient_aggregation="unaveraged_sum_over_valid_BxK_cells",
        physical_write="one_per_block_per_nonempty_tile",
        trajectory_source=cfg.train.trajectory_source,
        teacher_answer_input_reuse=(
            "h[L]_local_loss_target_is_detached_i[L+1]"
            if teacher_hidden else "student_block_output"),
        teacher_prefill_residency=(
            "one_full_teacher_layer_transient_plus_persistent_i1"
            if teacher_hidden else None),
        teacher_gpu_residency=(
            "owner_local_active_BxK_targets_only"
            if teacher_hidden
            and cfg.train.teacher_hidden_source == "gpu_cache" else
            "owner_local_current_target_transfer"
            if teacher_hidden else None),
        lr_rule=cfg.train.lr_rule,
        base_learning_rate=cfg.train.lr,
        lr_epoch_multipliers=cfg.train.lr_epoch_multipliers,
        vocab_cosine_samples=cfg.train.vocab_cosine_samples,
        vocab_cosine_seed=cfg.train.vocab_cosine_seed,
        parameterization=("lora" if cfg.train.lora.enabled else "full_weight"),
        trainable_parameter_dtypes=trainable_dtypes,
        execution_dtype=str(execution_dtype),
        output_evaluation=(
            "CE-eval-loss and KL-eval-loss over every answer token in the "
            "whole training-set traversal; evaluation_only=true; "
            "used_for_backward=false; optimizer_weight=0"),
        output_evaluation_answer_tokens_per_epoch=full_training_answer_tokens,
        lookahead_contract=(
            "next_token_online" if K == 1 else
            "teacher_prefetched_or_speculative_confirmed_tokens"),
        layer_types=layer_types,
        cache_hash=cache._index["config_hash"],
        teacher_cache_chain_tripwire=teacher_chain_tripwire,
        footprint_guard=footprint,
    )

    for epoch in range(cfg.train.epochs):
        lr_multiplier = (
            cfg.train.lr_epoch_multipliers[epoch]
            if cfg.train.lr_rule == "epoch_piecewise" else 1.0)
        epoch_lr = cfg.train.lr * lr_multiplier
        epoch_started = time.time()
        epoch_events_start = token_events
        epoch_conceptual_start = conceptual_writes
        epoch_physical_start = physical_writes
        epoch_loss_sums = None
        epoch_grad_sums = None
        epoch_cells = 0
        epoch_ce_eval_sum = torch.zeros(
            (), dtype=torch.float32, device=device)
        epoch_kl_eval_sum = torch.zeros(
            (), dtype=torch.float32, device=device)
        epoch_output_eval_tokens = 0
        cohorts = _bk_bucketed_cohorts(ds, B, cfg.train.seed + epoch)
        for cohort_index, indices in enumerate(cohorts):
            items = [ds[index] for index in indices]
            shards = _bk_prepare_cohort_shards(
                cfg, stack, items, device, teacher, teacher_hidden,
                layer_types, activation_shard_users, K, execution_dtype)
            b_now = len(items)
            max_answer = max(shard["max_answer"] for shard in shards)
            # PPn owns the stage coordinate while the code below remains the
            # one-device v3.2 reference.  The cohort is the synchronization
            # boundary: all wavefront tiles drain before completed users are
            # released and before stop/checkpoint handling runs.
            if partition.stages > 1:
                pp_result = _bk_run_ppn_cohort(
                    cfg, stack, loss_fn, shards, max_answer, n, device,
                    teacher_hidden, layer_types, epoch_lr, partition,
                    execution=cfg.train.pp_execution)
                tile_loss_sums = pp_result["loss_sums"]
                tile_grad_sums = pp_result["grad_sums"]
                if epoch_loss_sums is None:
                    epoch_loss_sums = tile_loss_sums
                    epoch_grad_sums = tile_grad_sums
                else:
                    _foreach_accumulate(epoch_loss_sums, tile_loss_sums)
                    _foreach_accumulate(epoch_grad_sums, tile_grad_sums)
                epoch_cells += pp_result["cells"]
                token_events += pp_result["cells"]
                conceptual_writes += pp_result["cells"] * n
                physical_writes += pp_result["physical_writes"]
                if pp_result["eval_tokens"]:
                    epoch_ce_eval_sum.add_(
                        pp_result["ce_sum"].to(epoch_ce_eval_sum.device))
                    epoch_kl_eval_sum.add_(
                        pp_result["kl_sum"].to(epoch_kl_eval_sum.device))
                    epoch_output_eval_tokens += pp_result["eval_tokens"]
                log.log(
                    kind="ppn_cohort",
                    epoch=epoch + 1,
                    cohort=cohort_index,
                    pp_execution=cfg.train.pp_execution,
                    stage_telemetry=pp_result["executor"].telemetry(),
                    tile_count=(pp_result["physical_writes"] // n),
                    boundary_bytes_per_tile=pp_result["boundary_bytes"],
                )
            # The reference path is deliberately left byte-for-byte in its
            # tile-major order.  PPn has already consumed this cohort when it
            # has more than one stage.
            for start in range(0, 0 if partition.stages > 1 else max_answer, K):
                tile_states = [
                    state for shard in shards
                    if (state := _bk_prepare_shard_tile(
                        shard, start, K, n, device, stack, teacher_hidden,
                        execution_dtype))
                ]
                cells = sum(state["cells"] for state in tile_states)
                if not cells:
                    continue
                tile_loss_sums = []
                tile_grad_sums = []
                for layer in range(1, n + 1):
                    params = _clear_block_grads(stack, layer)
                    loss_sum = torch.zeros(
                        (), dtype=torch.float32, device=device)
                    eval_student_parts = []
                    eval_teacher_parts = []
                    eval_id_parts = []
                    # Each shard sees the same pre-write parameters. Backward
                    # releases its local graph before the next shard, then
                    # the accumulated B×K gradient writes once below.
                    for state in tile_states:
                        shard = state["shard"]
                        if teacher_hidden and layer == 1:
                            state["_teacher_next"] = state["h"].detach()
                        h_in = (state["_teacher_next"] if teacher_hidden
                                else state["h"].detach())
                        layer_mask = (
                            state["query_valid"]
                            if layer_types[layer - 1] == "linear_attention"
                            else state["full_mask"])
                        with torch.autocast(
                                device.type, dtype=torch.bfloat16,
                                enabled=cfg.train.lora.enabled):
                            h_out = stack.run_block(
                                layer, h_in, state["query_rope"],
                                position_ids=state["query_positions"],
                                # The additive mask addresses the full static
                                # cache; flow_keep here is only the current
                                # query-row zeroing mask.
                                flow_keep=state["query_valid"],
                                past_key_values=shard["history"], use_cache=True,
                                causal_length=shard["full_keep"].shape[1],
                                prepared_attention_mask=layer_mask)
                            flat_view = stack.loss_view(layer, h_out).reshape(
                                -1, h_out.shape[-1])
                            view = flat_view.index_select(
                                0, state["valid_index"])
                            target_all = state["window_targets"][
                                layer - 1]
                            flat_target = target_all.reshape(
                                -1, target_all.shape[-1])
                            target = flat_target.index_select(
                                0, state["valid_index"])
                            mean_loss = loss_fn(
                                view, target, normed=(layer == n), layer=layer)
                            summed_loss = mean_loss * state["cells"]
                        summed_loss.backward()
                        if layer == n and state["eval_index"].numel():
                            eval_student_parts.append(
                                flat_view.detach().index_select(
                                    0, state["eval_index"]))
                            eval_teacher_parts.append(
                                flat_target.detach().index_select(
                                    0, state["eval_index"]))
                            eval_id_parts.append(state["eval_target_ids"])
                        loss_sum.add_(mean_loss.detach().float()
                                      * state["cells"])
                        _detach_cache_layer(shard["history"], layer - 1)
                        if teacher_hidden:
                            if layer < n:
                                state["_teacher_next"] = target_all.detach()
                        else:
                            state["h"] = h_out.detach()
                        del h_in, h_out, flat_view, view, target_all
                        del flat_target, target, summed_loss, mean_loss
                    if eval_id_parts:
                        ce_sum, kl_sum, eval_tokens, _sm_unused, _tm_unused = teacher_output_eval_sums(
                            torch.cat(eval_student_parts),
                            torch.cat(eval_teacher_parts),
                            torch.cat(eval_id_parts),
                            stack.lm_head,
                            chunk_rows=256,
                        )
                        epoch_ce_eval_sum.add_(ce_sum)
                        epoch_kl_eval_sum.add_(kl_sum)
                        epoch_output_eval_tokens += eval_tokens
                    grad = _immediate_sgd(params, epoch_lr)
                    tile_loss_sums.append(loss_sum)
                    tile_grad_sums.append(grad.detach())
                if epoch_loss_sums is None:
                    epoch_loss_sums = tile_loss_sums
                    epoch_grad_sums = tile_grad_sums
                else:
                    _foreach_accumulate(epoch_loss_sums, tile_loss_sums)
                    _foreach_accumulate(epoch_grad_sums, tile_grad_sums)
                epoch_cells += cells
                token_events += cells
                conceptual_writes += cells * n
                physical_writes += n
                for state in tile_states:
                    state.pop("_teacher_next", None)
                    del state["window_targets"], state["query_rope"]
                    del state["valid_index"], state["full_mask"], state["h"]
                    del state["eval_index"], state["eval_target_ids"]
                    del state["source_index"], state["query_positions"]
                    del state["query_valid"]

            answers_seen += b_now
            log.log(
                kind="v31_cohort",
                epoch=epoch + 1,
                cohort=cohort_index,
                users=b_now,
                activation_shard_users=activation_shard_users,
                activation_shards=len(shards),
                prompt_length_padded=max(
                    shard["prompt_length"] for shard in shards),
                max_answer_tokens=max_answer,
                token_events_seen=token_events,
                answers_seen=answers_seen,
                physical_optimizer_updates_seen=physical_writes,
            )
            for shard in shards:
                del shard["teacher_input"], shard["history"]
                del shard["student_ids"], shard["batch_positions"]
                del shard["lengths"], shard["answer_keep"]
                del shard["full_keep"], shard["source_s0"]
                del shard["target_items"], shard["target_staging"]
                del shard["teacher_staging"], shard["lengths_cpu"]
            del shards, items

            # A cohort boundary is a coherent model state: every BxK tile in
            # the cohort has completed every block-local write, and all CUDA
            # graphs/caches owned by the cohort have been released. Poll only
            # here so SIGTERM never publishes a half-written layer tile.
            if stop_requested():
                seconds = time.time() - epoch_started
                events = token_events - epoch_events_start
                pending_losses = (
                    [[value / epoch_cells for value in epoch_loss_sums]]
                    if epoch_cells else [])
                _flush_train_log(
                    log, epoch=epoch, step=token_events, accum=answers_seen,
                    pending=pending_losses, pending_items=epoch_cells,
                    n_layers=n, partial=True,
                    pipeline_version=3, update_granularity="online",
                    update_reduction="unaveraged_sum_BxK",
                    trajectory_source=cfg.train.trajectory_source,
                    history_policy=cfg.train.history_policy,
                    token_events_seen=token_events,
                    optimizer_updates_seen=conceptual_writes,
                    physical_optimizer_updates_seen=physical_writes,
                    optimizer_updates_per_token=n,
                    stale_gradient_window=K,
                    lr_rule=cfg.train.lr_rule,
                    learning_rate=epoch_lr,
                    lr_multiplier=lr_multiplier,
                    gradient_aggregation=(
                        "unaveraged_sum_at_shared_weight_snapshot"),
                    completed_epochs=epoch, partial_epoch=True,
                )
                log.log(
                    kind="teacher_output_eval_partial",
                    epoch=epoch + 1,
                    answer_token_count=epoch_output_eval_tokens,
                    expected_answer_token_count=full_training_answer_tokens,
                    dataset_item_count=len(ds),
                    dataset_coverage="partial_epoch_not_a_validation_result",
                    evaluation_only=True,
                    validation_subset=False,
                    used_for_backward=False,
                    optimizer_weight=0.0,
                )
                if epoch_grad_sums is not None and epoch_cells:
                    log.log(
                        kind="v3_gradient_norm",
                        epoch=epoch + 1,
                        per_layer_mean=[
                            float((value / epoch_cells).cpu())
                            for value in epoch_grad_sums],
                        measure=(
                            "BxK_window_sum_gradient_norm_normalized_per_cell"),
                        token_events_seen=token_events,
                        optimizer_updates_seen=conceptual_writes,
                        physical_optimizer_updates_seen=physical_writes,
                        partial_epoch=True,
                    )
                log.log(
                    kind="v3_throughput", epoch=epoch + 1, seconds=seconds,
                    token_events=events,
                    local_writes=(
                        conceptual_writes - epoch_conceptual_start),
                    physical_local_writes=(
                        physical_writes - epoch_physical_start),
                    token_events_per_s=events / max(seconds, 1e-9),
                    local_writes_per_s=(
                        conceptual_writes - epoch_conceptual_start)
                        / max(seconds, 1e-9),
                    physical_local_writes_per_s=(
                        physical_writes - epoch_physical_start)
                        / max(seconds, 1e-9),
                    includes=(
                        "partial_epoch_bucketed_cache_io_plus_prompt_prefill_"
                        "plus_BxK_writes"),
                    history_policy=cfg.train.history_policy,
                    trajectory_source=cfg.train.trajectory_source,
                    stale_gradient_window=K,
                    lr_rule=cfg.train.lr_rule,
                    learning_rate=epoch_lr,
                    lr_multiplier=lr_multiplier,
                    completed_epochs=epoch,
                    partial_epoch=True,
                )
                log.log(
                    kind="graceful_stop",
                    signal=requested_signal(),
                    boundary="completed_cohort",
                    completed_epochs=epoch,
                    partial_epoch=epoch + 1,
                    token_events_seen=token_events,
                    answers_seen=answers_seen,
                    checkpoint_pending=True,
                )
                return True

        pending_losses = (
            [[value / epoch_cells for value in epoch_loss_sums]]
            if epoch_cells else [])
        _flush_train_log(
            log, epoch=epoch, step=token_events, accum=answers_seen,
            pending=pending_losses, pending_items=epoch_cells,
            n_layers=n, partial=False,
            pipeline_version=3, update_granularity="online",
            update_reduction="unaveraged_sum_BxK",
            trajectory_source=cfg.train.trajectory_source,
            history_policy=cfg.train.history_policy,
            token_events_seen=token_events,
            optimizer_updates_seen=conceptual_writes,
            physical_optimizer_updates_seen=physical_writes,
            optimizer_updates_per_token=n,
            stale_gradient_window=K,
            lr_rule=cfg.train.lr_rule,
            learning_rate=epoch_lr,
            lr_multiplier=lr_multiplier,
            gradient_aggregation="unaveraged_sum_at_shared_weight_snapshot",
            completed_epochs=epoch + 1, partial_epoch=False,
        )
        if epoch_output_eval_tokens != full_training_answer_tokens:
            raise RuntimeError(
                "output evaluation did not cover the whole training set: "
                f"measured {epoch_output_eval_tokens:,} answer tokens, "
                f"expected {full_training_answer_tokens:,}")
        log.log(
            kind="teacher_output_eval",
            epoch=epoch + 1,
            CE_eval_loss=float(
                (epoch_ce_eval_sum / epoch_output_eval_tokens).cpu()),
            KL_eval_loss=float(
                (epoch_kl_eval_sum / epoch_output_eval_tokens).cpu()),
            answer_token_count=epoch_output_eval_tokens,
            expected_answer_token_count=full_training_answer_tokens,
            dataset_item_count=len(ds),
            dataset_coverage="whole_training_set_once_per_completed_epoch",
            token_coverage="every_teacher_realized_answer_token",
            answer_only=True,
            evaluation_only=True,
            validation_subset=False,
            used_for_backward=False,
            optimizer_weight=0.0,
            aggregation="token_weighted_mean",
            temporal_semantics=(
                "streaming_pre_final_block_write_at_each_sample_visit"),
            CE_target="teacher_realized_answer_token_ids",
            KL_direction="teacher_to_student",
            vocabulary_head="frozen",
        )
        if epoch_grad_sums is not None:
            log.log(
                kind="v3_gradient_norm",
                epoch=epoch + 1,
                per_layer_mean=[float((value / epoch_cells).cpu())
                                for value in epoch_grad_sums],
                measure="BxK_window_sum_gradient_norm_normalized_per_cell",
                token_events_seen=token_events,
                optimizer_updates_seen=conceptual_writes,
                physical_optimizer_updates_seen=physical_writes,
            )
        seconds = time.time() - epoch_started
        events = token_events - epoch_events_start
        log.log(
            kind="v3_throughput",
            epoch=epoch + 1,
            seconds=seconds,
            token_events=events,
            local_writes=conceptual_writes - epoch_conceptual_start,
            physical_local_writes=physical_writes - epoch_physical_start,
            token_events_per_s=events / seconds,
            local_writes_per_s=(conceptual_writes - epoch_conceptual_start) / seconds,
            physical_local_writes_per_s=(physical_writes - epoch_physical_start) / seconds,
            includes="bucketed_cache_io_plus_prompt_prefill_plus_BxK_writes",
            history_policy=cfg.train.history_policy,
            trajectory_source=cfg.train.trajectory_source,
            stale_gradient_window=K,
            lr_rule=cfg.train.lr_rule,
            learning_rate=epoch_lr,
            lr_multiplier=lr_multiplier,
            completed_epochs=epoch + 1,
            partial_epoch=False,
        )
        standard_baseline = _epoch_end_telemetry(
            cfg, stack, tok, log, epoch=epoch,
            baseline=standard_baseline, started_at=started)
        delta_tracker.log(
            log, epoch=epoch + 1, phase=f"after_epoch_{epoch + 1}",
            started_at=started)
    return False


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
    first = teacher_states[0][:, :stop].detach()
    first_pos = it.position_ids[:stop].to(first.device)[None]
    pos_emb = stack.rope(first, first_pos)
    for layer in range(1, stack.n_layers + 1):
        h_in = teacher_states[layer - 1][:, :stop].detach()
        pos = it.position_ids[:stop].to(h_in.device)[None]
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
        first = teacher_states[0][:, :stop].detach()
        pos_emb = stack.rope(first, pos.to(first.device))
        for layer in range(1, stack.n_layers + 1):
            h_in = teacher_states[layer - 1][:, :stop].detach()
            local_pos = pos.to(h_in.device)
            keep = _flow_keep(cfg, it, stop, h_in.device)
            if deferred:
                loss, _, params = _local_forward(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, local_pos,
                    (targets or it.hidden)[layer][offset], pos_index,
                    flow_keep=keep)
                losses.append(loss)
                deferred_params.append(params)
            else:
                loss, grad, _ = _local_update(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, local_pos,
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
                  student_ids=None, position_ids=None, targets=None,
                  prepared_masks=None):
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
        pos_emb = stack.rope(h, pos)
        for layer in range(1, stack.n_layers + 1):
            if deferred:
                loss, h_out, params = _local_forward(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1,
                    prepared_attention_mask=(
                        _prepared_mask_row(prepared_masks[layer], pos_index)
                        if prepared_masks is not None else None))
                losses.append(loss)
                deferred_params.append(params)
                h = h_out.detach()
            else:
                loss, grad, h = _local_update(
                    cfg, stack, loss_fn, layer, h, pos_emb, pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1,
                    prepared_attention_mask=(
                        _prepared_mask_row(prepared_masks[layer], pos_index)
                        if prepared_masks is not None else None))
                losses.append(loss)
                grad_norms.append(grad)
    else:
        first = teacher_states[0][
            :, pos_index:pos_index + 1].detach()
        pos_emb = stack.rope(first, pos.to(first.device))
        for layer in range(1, stack.n_layers + 1):
            h_in = teacher_states[layer - 1][
                :, pos_index:pos_index + 1].detach()
            local_pos = pos.to(h_in.device)
            if deferred:
                loss, _, params = _local_forward(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, local_pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1,
                    prepared_attention_mask=(
                        _prepared_mask_row(prepared_masks[layer], pos_index)
                        if prepared_masks is not None else None))
                losses.append(loss)
                deferred_params.append(params)
            else:
                loss, grad, _ = _local_update(
                    cfg, stack, loss_fn, layer, h_in, pos_emb, local_pos,
                    (targets or it.hidden)[layer][offset], 0,
                    flow_keep=full_keep, cache=cache,
                    causal_length=pos_index + 1,
                    prepared_attention_mask=(
                        _prepared_mask_row(prepared_masks[layer], pos_index)
                        if prepared_masks is not None else None))
                losses.append(loss)
                grad_norms.append(grad)
    if deferred:
        losses, grad_norms = _finish_disconnected_token(
            cfg, losses, deferred_params)
        for layer in range(stack.n_layers):
            _detach_cache_layer(cache, layer)
    return losses, grad_norms


def train_online_v3(cfg, stack, tok, log, cache, teacher=None) -> bool:
    """Run the pipeline-v3 online walk; checkpoint publication stays in runtime."""
    if cfg.train.history_policy == "causal_bk":
        return train_bk_v32(
            cfg, stack, tok, log, cache, teacher=teacher)
    if cfg.train.history_policy in (
            "causal_static_eager_probe", "causal_static_graph_probe",
            "causal_bk_probe"):
        raise NotImplementedError(
            "v3 probe history policies are certification-only; use the "
            "matching smoke instrument before adding campaign dispatch")
    device = cfg.model.device
    n = stack.n_layers
    if (not cfg.train.lora.enabled
            and cfg.model.dtype in ("bfloat16", "float16")):
        raise NotImplementedError(
            "pipeline-v3 full-weight immediate SGD in reduced precision can "
            "round token writes to zero; use LoRA until an fp32 master or "
            "stochastic-rounding path is implemented")
    unsupported_attention = sorted(
        set(stack.layer_types) & {"sliding_attention", "chunked_attention"})
    if unsupported_attention:
        raise NotImplementedError(
            "pipeline-v3 cached execution does not yet implement the model-"
            "authoritative semantics for attention types "
            f"{unsupported_attention}; do not approximate them as rolling "
            "windows")
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
    token_events = optimizer_updates = physical_optimizer_updates = 0
    answers_seen = 0
    max_answer_stage_bytes = 0
    done = False
    standard_baseline = _epoch_zero_telemetry(
        cfg, stack, tok, log, started)
    delta_tracker.log(log, epoch=0, phase="epoch0", started_at=started)
    log.log(
        kind="pipeline_v3_contract",
        atomic_event=(
            "one_aligned_token"
            if cfg.train.stale_gradient_window == 1 else
            "logical_tokens_grouped_by_stale_weight_snapshot"),
        block_order=(
            "teacher_window_major_forward_layers"
            if cfg.train.stale_gradient_window != 1 else
            "independent_teacher_layer_lanes"
            if cfg.train.backward_dispatch == "teacher_layer_lanes" else
            "student_answer_antidiagonal"
            if cfg.train.backward_dispatch in (
                "answer_wavefront_disconnected", "answer_pipeline_lanes") else
            "forward_within_token"),
        updates_per_token=n,
        optimizer="state_free_immediate_sgd",
        backward_dispatch=cfg.train.backward_dispatch,
        online_write_dispatch=cfg.train.online_write_dispatch,
        stale_gradient_window=cfg.train.stale_gradient_window,
        gradient_aggregation=(
            "none"
            if cfg.train.stale_gradient_window == 1 else
            "unaveraged_sum_at_shared_weight_snapshot"),
        staleness=(
            "none"
            if cfg.train.stale_gradient_window == 1 else
            "later_window_tokens_do_not_recompute_after_earlier_logical_writes"),
        history_policy=cfg.train.history_policy,
        history_lifetime="current_answer_only",
        trajectory_source=cfg.train.trajectory_source,
        teacher_hidden_identity="uncensored_teacher_h[L-1]",
    )

    for epoch in range(cfg.train.epochs):
        epoch_started = time.time()
        epoch_token_start = token_events
        epoch_write_start = optimizer_updates
        epoch_physical_write_start = physical_optimizer_updates
        progress_token_start = token_events
        progress_started = epoch_started
        epoch_loss_sums = None
        epoch_grad_sums = None
        epoch_answer_count = 0
        order = list(range(len(ds)))
        random.Random(cfg.train.seed + epoch).shuffle(order)
        for index in order:
            it = ds[index]
            student_ids, position_ids, targets, staged_bytes = (
                stage_answer_tensors(stack, it, device))
            max_answer_stage_bytes = max(max_answer_stage_bytes, staged_bytes)
            teacher_states = (
                teacher.full_inputs_resident(it, device)
                if teacher_hidden else None)
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
            prepared_masks = None
            if (history is not None
                    and cfg.train.stale_gradient_window == 1
                    and cfg.train.backward_dispatch not in (
                    "teacher_layer_lanes", "answer_pipeline_lanes",
                    "answer_wavefront_disconnected")):
                prepared_masks = _prepared_cached_masks(
                    cfg, stack, it, position_ids, targets)
            use_teacher_window_path = (
                teacher_hidden and history is not None
                and cfg.train.backward_dispatch == "per_block"
                and cfg.train.online_write_dispatch == "after_backward")
            if use_teacher_window_path:
                remaining = (cfg.train.max_steps - token_events
                             if cfg.train.max_steps else it.A)
                answer_tokens = min(it.A, remaining)
                losses, grads, writes, physical_writes = (
                    answer_teacher_stale_windows_cached(
                        cfg, stack, loss_fn, it, answer_tokens, device,
                        history, teacher_states=teacher_states,
                        position_ids=position_ids, targets=targets))
                answer_loss_sums = [x * answer_tokens for x in losses]
                answer_grad_sums = [x * answer_tokens for x in grads]
                token_events += answer_tokens
                optimizer_updates += writes
                physical_optimizer_updates += physical_writes
                done = bool(cfg.train.max_steps
                            and token_events >= cfg.train.max_steps)
            elif cfg.train.backward_dispatch == "teacher_layer_lanes":
                remaining = (cfg.train.max_steps - token_events
                             if cfg.train.max_steps else it.A)
                answer_tokens = min(it.A, remaining)
                losses, grads, writes = answer_teacher_layer_lanes_cached(
                    cfg, stack, loss_fn, it, answer_tokens, device, history,
                    teacher_states=teacher_states, position_ids=position_ids,
                    targets=targets)
                answer_loss_sums = [x * answer_tokens for x in losses]
                answer_grad_sums = [x * answer_tokens for x in grads]
                token_events += answer_tokens
                optimizer_updates += writes
                physical_optimizer_updates += writes
                done = bool(cfg.train.max_steps
                            and token_events >= cfg.train.max_steps)
            elif cfg.train.backward_dispatch == "answer_pipeline_lanes":
                remaining = (cfg.train.max_steps - token_events
                             if cfg.train.max_steps else it.A)
                answer_tokens = min(it.A, remaining)
                losses, grads, writes = answer_student_pipeline_lanes_cached(
                    cfg, stack, loss_fn, it, answer_tokens, device, history,
                    student_ids=student_ids, position_ids=position_ids,
                    targets=targets)
                answer_loss_sums = [x * answer_tokens for x in losses]
                answer_grad_sums = [x * answer_tokens for x in grads]
                token_events += answer_tokens
                optimizer_updates += writes
                physical_optimizer_updates += writes
                done = bool(cfg.train.max_steps
                            and token_events >= cfg.train.max_steps)
            elif cfg.train.backward_dispatch == "answer_wavefront_disconnected":
                remaining = (cfg.train.max_steps - token_events
                             if cfg.train.max_steps else it.A)
                answer_tokens = min(it.A, remaining)
                losses, grads, writes = answer_wavefront_cached(
                    cfg, stack, loss_fn, it, answer_tokens, device, history,
                    student_ids=student_ids, position_ids=position_ids,
                    targets=targets)
                answer_loss_sums = [x * answer_tokens for x in losses]
                answer_grad_sums = [x * answer_tokens for x in grads]
                token_events += answer_tokens
                optimizer_updates += writes
                physical_optimizer_updates += writes
                done = bool(cfg.train.max_steps
                            and token_events >= cfg.train.max_steps)
            else:
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
                            targets=targets, prepared_masks=prepared_masks)
                    if answer_loss_sums is None:
                        answer_loss_sums = losses
                        answer_grad_sums = grads
                    else:
                        _foreach_accumulate(answer_loss_sums, losses)
                        _foreach_accumulate(answer_grad_sums, grads)
                    answer_tokens += 1
                    token_events += 1
                    optimizer_updates += n
                    physical_optimizer_updates += n
                    if (cfg.train.max_steps
                            and token_events >= cfg.train.max_steps):
                        done = True
                        break
            if answer_tokens:
                answer_losses = [x / answer_tokens for x in answer_loss_sums]
                answer_grads = [x / answer_tokens for x in answer_grad_sums]
                if epoch_loss_sums is None:
                    epoch_loss_sums = answer_losses
                    epoch_grad_sums = answer_grads
                else:
                    _foreach_accumulate(epoch_loss_sums, answer_losses)
                    _foreach_accumulate(epoch_grad_sums, answer_grads)
                epoch_answer_count += 1
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
                    physical_optimizer_updates_seen=physical_optimizer_updates,
                    answers_seen=answers_seen,
                    interval_token_events=progress_tokens,
                    interval_seconds=progress_seconds,
                    interval_token_events_per_s=(
                        progress_tokens / progress_seconds),
                    backward_dispatch=cfg.train.backward_dispatch,
                    online_write_dispatch=cfg.train.online_write_dispatch,
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

        pending_losses = (
            [[value / epoch_answer_count for value in epoch_loss_sums]]
            if epoch_answer_count else [])
        _flush_train_log(
            log, epoch=epoch, step=token_events, accum=answers_seen,
            pending=pending_losses, pending_items=epoch_answer_count,
            n_layers=n, partial=done,
            pipeline_version=3, update_granularity="online",
            update_reduction="none", trajectory_source=cfg.train.trajectory_source,
            history_policy=cfg.train.history_policy,
            token_events_seen=token_events,
            optimizer_updates_seen=optimizer_updates,
            physical_optimizer_updates_seen=physical_optimizer_updates,
            optimizer_updates_per_token=n,
            stale_gradient_window=cfg.train.stale_gradient_window,
            gradient_aggregation=(
                "none" if cfg.train.stale_gradient_window == 1 else
                "unaveraged_sum_at_shared_weight_snapshot"),
            completed_epochs=(epoch if done else epoch + 1),
            partial_epoch=done,
            max_answer_stage_mib=max_answer_stage_bytes / 2**20,
        )
        if done:
            log.log(
                kind="v3_partial_boundary",
                token_events_seen=token_events,
                optimizer_updates_seen=optimizer_updates,
                physical_optimizer_updates_seen=physical_optimizer_updates,
                completed_epochs=epoch,
                partial_epoch_index=epoch + 1,
                meaning=(
                    "budget checkpoint inside a dataset traversal; not a "
                    "completed epoch"),
            )
        if epoch_grad_sums is not None:
            grad_device = epoch_grad_sums[0].device
            by_layer = torch.stack([
                (value / epoch_answer_count).to(
                    grad_device, non_blocking=True)
                for value in epoch_grad_sums
            ]).detach().cpu()
            log.log(
                kind="v3_gradient_norm",
                epoch=epoch + 1,
                per_layer_mean=[float(x) for x in by_layer],
                measure=(
                    "mean_immediate_gradient_norm"
                    if cfg.train.stale_gradient_window == 1 else
                    "window_sum_gradient_norm_normalized_per_token"),
                token_events_seen=token_events,
                optimizer_updates_seen=optimizer_updates,
                physical_optimizer_updates_seen=physical_optimizer_updates,
            )
        epoch_seconds = time.time() - epoch_started
        epoch_tokens = token_events - epoch_token_start
        epoch_writes = optimizer_updates - epoch_write_start
        epoch_physical_writes = (
            physical_optimizer_updates - epoch_physical_write_start)
        log.log(
            kind="v3_throughput",
            epoch=epoch + 1,
            seconds=epoch_seconds,
            token_events=epoch_tokens,
            local_writes=epoch_writes,
            physical_local_writes=epoch_physical_writes,
            token_events_per_s=(epoch_tokens / epoch_seconds),
            local_writes_per_s=(epoch_writes / epoch_seconds),
            physical_local_writes_per_s=(
                epoch_physical_writes / epoch_seconds),
            includes="dataset_cache_io_plus_prompt_prefill_plus_online_writes",
            history_policy=cfg.train.history_policy,
            trajectory_source=cfg.train.trajectory_source,
            stale_gradient_window=cfg.train.stale_gradient_window,
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
