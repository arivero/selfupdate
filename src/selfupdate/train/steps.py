"""Block-local and windowed step primitives.

Schedule-agnostic layer: every function here performs one forward+backward
for one block or one connected window, on tensors the caller prepared.
Nothing in this module knows about configs, schedules, epochs, optimizers,
teachers, or logging — that separation is what keeps the locality contract
auditable: detach discipline (inputs detached entering and leaving every
step) lives HERE and only here.

Window semantics (gradient-isolation, NOT memory management; endpoint vs
in-window loss; teacher- vs student-stream input): docs/windows.md — read it
before touching ``window_step`` or the sliding dedup.
"""

from __future__ import annotations

import contextlib

import torch

from ..data.dataset import Batch
from .losses import HiddenLoss
from .moe import pending_router_loss


def local_block_step(stack, L, h_in, pos_emb, target, s0, A, kind, autocast=True,
                     previous_target=None):
    """One local forward+backward for block L. ``h_in`` must be detached, so
    the recorded graph — and therefore the backward — is confined to block L:
    no gradient from this loss can reach any other block, the lm_head, or the
    logits. Returns (loss value, detached block output). Autocast wraps only
    the forward+loss; backward runs outside it.

    ``kind`` is a HiddenLoss or a kind string (coerced; vocab-metric kinds
    need the constructed HiddenLoss carrying the frozen norm/head)."""
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(L, h_in, pos_emb)
        if loss_fn.is_delta and 1 < L < stack.n_layers:
            if previous_target is None:
                raise ValueError(
                    f"{loss_fn.kind} at interior layer {L} needs h{L - 1} teacher target"
                )
            loss = loss_fn.delta(h_out[0, s0: s0 + A], h_in[0, s0: s0 + A],
                                 target, previous_target)
        else:
            loss = loss_fn(stack.loss_view(L, h_out)[0, s0: s0 + A], target,
                           normed=(L == stack.n_layers), layer=L)
    extra = pending_router_loss()
    (loss if extra is None else loss + extra).backward()
    return loss.detach(), h_out.detach()


def last_block_step(stack, h_in, pos_emb, target, s0, A, kind, autocast=True):
    """Block n's local teacher-state step."""
    n = stack.n_layers
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(n, h_in, pos_emb)
        normed = stack.final_norm(h_out)
        loss = loss_fn(normed[0, s0: s0 + A], target, normed=True, layer=n)
    extra = pending_router_loss()
    (loss if extra is None else loss + extra).backward()
    return loss.detach(), h_out.detach()


def _gather_batch_rows(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    idx = index.to(h.device).unsqueeze(-1).expand(-1, -1, h.shape[-1])
    return h.gather(1, idx)


def _hidden_loss_per_example(loss_fn, student_h: torch.Tensor, teacher_h: torch.Tensor,
                             lens: list[int], *, normed: bool,
                             layer: int | None = None) -> torch.Tensor:
    """Right-padded batches keep every valid row in a PREFIX (collate
    invariant), so slicing by the CPU-side length replaces bool-mask
    indexing — whose implicit nonzero() is a host-device sync per example
    per layer, the same stall class as .item() in the block walk."""
    teacher_h = teacher_h.to(student_h.device)
    losses = []
    for i, k in enumerate(lens):
        losses.append(loss_fn(student_h[i, :k], teacher_h[i, :k],
                              normed=normed, layer=layer))
    return torch.stack(losses)


def _delta_loss_per_example(loss_fn, student_h: torch.Tensor,
                            student_prev: torch.Tensor,
                            teacher_h: torch.Tensor,
                            teacher_prev: torch.Tensor,
                            lens: list[int]) -> torch.Tensor:
    """Per-example counterpart of :meth:`HiddenLoss.delta`.

    The prefix-slicing invariant is the same as the ordinary hidden loss;
    keeping it here avoids per-layer boolean-mask synchronizations in padded
    batches.
    """
    teacher_h = teacher_h.to(student_h.device)
    teacher_prev = teacher_prev.to(student_h.device)
    losses = []
    for i, k in enumerate(lens):
        losses.append(loss_fn.delta(student_h[i, :k], student_prev[i, :k],
                                    teacher_h[i, :k], teacher_prev[i, :k]))
    return torch.stack(losses)


def _layer_loss_per_example(loss_fn, stack, L: int, h_out: torch.Tensor,
                            h_prev: torch.Tensor, teacher_h: torch.Tensor,
                            teacher_prev: torch.Tensor | None,
                            batch: Batch) -> torch.Tensor:
    """Select raw-increment or absolute-state measurement for one layer.

    Cache convention makes the first and final boundaries special: no cached
    h0, and h_n is post-final-norm.  Delta kinds therefore use their paired
    state fallback at those two boundaries, while every interior transformer
    block is measured by its raw update.
    """
    if loss_fn.is_delta and 1 < L < stack.n_layers:
        if teacher_prev is None:
            raise ValueError(
                f"{loss_fn.kind} at interior layer {L} needs h{L - 1} teacher target"
            )
        return _delta_loss_per_example(
            loss_fn,
            _gather_batch_rows(h_out, batch.aligned_index),
            _gather_batch_rows(h_prev, batch.aligned_index),
            teacher_h, teacher_prev, batch.A.tolist(),
        )
    aligned = _gather_batch_rows(stack.loss_view(L, h_out), batch.aligned_index)
    return _hidden_loss_per_example(
        loss_fn, aligned, teacher_h, batch.A.tolist(),
        normed=(L == stack.n_layers), layer=L,
    )


def _reduce_example_losses(losses: torch.Tensor, batch: Batch,
                           update_reduction: str) -> torch.Tensor:
    """Reduce per-answer means to the configured optimizer-update scalar.

    Historical ``answer`` remains intentionally B=1. Grid ``answer_mean``
    gives every selected answer row equal weight; ``token_mean`` (and legacy
    ``token``) weights each per-answer mean by its selected valid-token count,
    producing the mean over answer x token cells. The legacy sum is retained
    only for exact historical reproducibility.
    """
    if update_reduction == "legacy_answer_sum":
        return losses.sum()
    if update_reduction == "answer":
        if losses.numel() != 1:
            raise ValueError("answer aggregation received more than one answer")
        return losses[0]
    if update_reduction == "answer_mean":
        return losses.mean()
    if update_reduction in ("token", "token_mean"):
        weights = batch.A.to(device=losses.device, dtype=losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1)
    raise ValueError(f"unknown update_reduction {update_reduction!r}")


def local_block_step_batch(stack, L, h_in, pos_emb, target, batch: Batch, kind,
                           autocast=True, previous_target=None,
                           update_reduction="legacy_answer_sum"):
    """Batched counterpart of :func:`local_block_step`.

    The total backward scalar is the sum of per-example losses, matching the
    historical item loop's gradient scale while sharing the block forward.
    """
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        if loss_fn.kind == "component_nmse":
            state_target, attn_target, mlp_target = target
            with _capture_block_components(stack, L) as components:
                h_out = stack.run_block(L, h_in, pos_emb)
            losses = _component_loss_per_example(
                components, attn_target, mlp_target, batch)
        else:
            h_out = stack.run_block(L, h_in, pos_emb)
            losses = _layer_loss_per_example(
                loss_fn, stack, L, h_out, h_in, target, previous_target, batch,
            )
        total = _reduce_example_losses(losses, batch, update_reduction)
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return losses.detach(), h_out.detach()


def last_block_step_batch(stack, h_in, pos_emb, target, batch: Batch, kind,
                          autocast=True, update_reduction="legacy_answer_sum"):
    n = stack.n_layers
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        if loss_fn.kind == "component_nmse":
            _, attn_target, mlp_target = target
            with _capture_block_components(stack, n) as components:
                h_out = stack.run_block(n, h_in, pos_emb)
            losses = _component_loss_per_example(
                components, attn_target, mlp_target, batch)
            total = _reduce_example_losses(losses, batch, update_reduction)
            extra = pending_router_loss()
            (total if extra is None else total + extra).backward()
            return losses.detach(), h_out.detach()
        h_out = stack.run_block(n, h_in, pos_emb)
        normed = stack.final_norm(h_out)
        aligned = _gather_batch_rows(normed, batch.aligned_index)
        losses = _hidden_loss_per_example(
            loss_fn, aligned, target, batch.A.tolist(), normed=True,
            layer=n
        )
        total = _reduce_example_losses(losses, batch, update_reduction)
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return losses.detach(), h_out.detach()


@contextlib.contextmanager
def _capture_block_components(stack, L):
    """Record recombined attention and MLP writes, never probabilities."""
    block = stack.blocks[L - 1]
    modules = (getattr(block, "self_attn", None), getattr(block, "mlp", None))
    if any(module is None for module in modules):
        raise ValueError(f"component_nmse unsupported by block {L}: needs self_attn and mlp")
    got = {}
    def save(name):
        def hook(_module, _args, output):
            got[name] = output[0] if isinstance(output, tuple) else output
        return hook
    handles = [modules[0].register_forward_hook(save("attn")),
               modules[1].register_forward_hook(save("mlp"))]
    try:
        yield got
    finally:
        for handle in handles:
            handle.remove()
    if set(got) != {"attn", "mlp"}:
        raise RuntimeError(f"component hooks did not fire at layer {L}: {sorted(got)}")


def _component_loss_per_example(components, attn_target, mlp_target, batch):
    attn = _gather_batch_rows(components["attn"], batch.aligned_index)
    mlp = _gather_batch_rows(components["mlp"], batch.aligned_index)
    metric = HiddenLoss("nmse")
    return 0.5 * (
        _hidden_loss_per_example(metric, attn, attn_target, batch.A.tolist(), normed=False)
        + _hidden_loss_per_example(metric, mlp, mlp_target, batch.A.tolist(), normed=False)
    )


def _span_batch(s0: int, A: int) -> Batch:
    """Index-only B=1 Batch for the hidden-state step functions."""
    return Batch(
        example_ids=["_span"],
        student_ids=None,
        position_ids=None,
        lengths=torch.tensor([s0 + A]),
        s0=torch.tensor([s0]),
        A=torch.tensor([A]),
        ans0=torch.tensor([s0 + A]),
        aligned_index=torch.arange(s0, s0 + A)[None],
        hidden_mask=torch.ones(1, A, dtype=torch.bool),
        hidden={},
        aligned_offset=torch.zeros(1, dtype=torch.long),
        source_A=torch.tensor([A], dtype=torch.long),
    )


def window_step(stack, L0, h_in, pos_emb, targets, s0, A, kind,
                hidden_w=1.0, L1=None, autocast=True, all_targets=None):
    """Joint hidden-state step for a CONNECTED window [L0..L1].

    Gradient flows only within the window and stops at its detached input.
    There is no behavioral readout or final-logit objective.

    Single-item adapter over :func:`window_step_batch` — a B=1 batch is
    bit-exact against the historical item code (same kernel shapes,
    gather == slice)."""
    batch = _span_batch(s0, A)
    all_targets = targets if all_targets is None else all_targets
    losses, h = window_step_batch(
        stack, L0, h_in, pos_emb, {L: t[None] for L, t in targets.items()},
        batch, kind, hidden_w=hidden_w, L1=L1, autocast=autocast,
        all_targets={L: t[None] for L, t in all_targets.items()})
    return [l[0] for l in losses], h


def window_step_batch(stack, L0, h_in, pos_emb, targets, batch: Batch, kind,
                      hidden_w=1.0, L1=None, autocast=True,
                      all_targets=None,
                      update_reduction="legacy_answer_sum"):
    """Batched connected-window step.

    Returns a list of per-example loss vectors in the same layer order as
    :func:`window_step`.
    """
    n = stack.n_layers
    L1 = n if L1 is None else L1
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    all_targets = targets if all_targets is None else all_targets
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h = h_in
        raw_states = {L0 - 1: h_in}
        losses = []
        for L in range(L0, L1 + 1):
            h_prev = h
            if loss_fn.kind == "component_nmse" and L in targets:
                with _capture_block_components(stack, L) as components:
                    h = stack.run_block(L, h, pos_emb)
            else:
                h = stack.run_block(L, h, pos_emb)
            raw_states[L] = h
            if L in targets:
                if loss_fn.kind == "component_nmse":
                    losses.append(_component_loss_per_example(
                        components, all_targets[("attn", L)],
                        all_targets[("mlp", L)], batch))
                elif loss_fn.is_multiscale:
                    vals = []
                    aligned = _gather_batch_rows(h, batch.aligned_index)
                    for i, k in enumerate(batch.A.tolist()):
                        student_history = {
                            depth: _gather_batch_rows(state, batch.aligned_index)[i, :k]
                            for depth, state in raw_states.items()
                        }
                        teacher_history = {
                            depth: value[i, :k]
                            for depth, value in all_targets.items()
                        }
                        vals.append(loss_fn.multiscale_delta(
                            aligned[i, :k], student_history, targets[L][i, :k],
                            teacher_history, L))
                    losses.append(torch.stack(vals))
                else:
                    losses.append(_layer_loss_per_example(
                        loss_fn, stack, L, h, h_prev, targets[L],
                        all_targets.get(L - 1), batch,
                    ))
        if losses:
            # under pipeline parallel a tied-vocab model computes the L == n
            # loss on the vocab card while in-window losses live on a block
            # card — accumulate the backward scalar on ONE device (scalar
            # moves, autograd-recorded)
            total = hidden_w * sum(
                _reduce_example_losses(loss, batch, update_reduction).to(h.device)
                for loss in losses)
        else:
            total = h.sum() * 0.0
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return [l.detach() for l in losses], h.detach()


def _sliding_windows_dedup(stack, L_start, last_body, W, h_traj, pos_emb,
                           compute_loss, autocast=True):
    """Forward-deduplicated FAITHFUL sliding windows (train.window_dedup).

    Same semantics as the per-endpoint ``window_step`` replay: every body
    layer L1 is the ENDPOINT of a window [L1-W+1 .. L1] rooted at the
    detached trajectory state h_traj[L0-1], and its backward updates ALL
    covered blocks — uniform k-deep credit (docs/windows.md). The backward
    count per block is untouched (that IS the credit assignment); what goes
    away is the forward duplication: instead of re-forwarding W blocks per
    endpoint, each block is grad-forwarded ONCE from its detached trajectory
    root, and every window chains its backward through the stored per-block
    graphs via ``torch.autograd.grad`` grad_outputs injection. This is valid
    because all windows follow the same trajectory: the value of window
    [L0..L1]'s intermediate state at depth b equals h_traj[b], the root of
    block b+1's stored graph (exactly in fp32; up to autocast replay rounding
    in bf16). Gradient isolation is preserved — each chain stops at the
    detached root x_{L0-1}, and no graph connects two blocks.

    Peak graph memory stays at W blocks: a block's graph is freed right
    after its last covering window (endpoint min(last_body, b+W-1)).

    ``compute_loss(L1, x, y)`` returns (backward scalar, detached report
    value) for the endpoint loss on block input/output ``x, y``; it runs
    under the same autocast as the forward.  Passing x is necessary for a
    delta lens to stop-gradient its preceding state.  Returns reports in
    endpoint order.
    """
    dev_type = h_traj[L_start - 1].device.type
    xs, ys, params = {}, {}, {}
    reports = []
    for L1 in range(L_start, last_body + 1):
        # .detach() gives x its own autograd identity (sharing storage);
        # POPPING the trajectory ref makes the xs/ys last-use deletions
        # below actually release that storage — otherwise the caller's dict
        # keeps every state alive for the whole item and the documented
        # W-block residency only holds for graphs, not activations
        # (h_traj[last_body] stays: it is the caller's walk output)
        root = (h_traj.pop(L1 - 1) if L1 - 1 != last_body
                else h_traj[L1 - 1])
        x = root.detach().requires_grad_(True)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=autocast):
            y = stack.run_block(L1, x, pos_emb)
            loss, report = compute_loss(L1, x, y)
        xs[L1], ys[L1] = x, y
        params[L1] = [p for p in stack.block_params(L1) if p.requires_grad]
        L0 = max(L_start, L1 - W + 1)
        g = None
        for b in range(L1, L0 - 1, -1):
            last_use = L1 == min(last_body, b + W - 1)
            inputs = [xs[b]] + params[b]
            if b == L1:
                grads = torch.autograd.grad(
                    loss, inputs, retain_graph=not last_use, allow_unused=True)
            else:
                g = g.to(device=ys[b].device, dtype=ys[b].dtype)
                grads = torch.autograd.grad(
                    ys[b], inputs, grad_outputs=g,
                    retain_graph=not last_use, allow_unused=True)
            g = grads[0]
            for p, gp in zip(params[b], grads[1:]):
                if gp is None:
                    continue
                p.grad = gp.detach() if p.grad is None else p.grad.add_(gp)
            if last_use:
                del xs[b], ys[b], params[b]
        reports.append(report)
    return reports
