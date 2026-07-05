"""Sliding connected windows + tail_hidden_weight: gradient extents."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from selfupdate.train.blocks import BlockStack
from selfupdate.train.losses import HiddenLoss
from selfupdate.train.layerwise import tail_step

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.fixture(scope="module")
def stack():
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             dtype=torch.float32).to("cuda")
    s = BlockStack(m)
    s.freeze_non_blocks()
    return s


def _setup(stack, T=12, A=5):
    torch.manual_seed(3)
    ids = torch.randint(10, 1000, (1, T), device="cuda")
    pos = torch.arange(T, device="cuda")[None]
    h = stack.embed(ids)
    pe = stack.rope(h, pos)
    with torch.no_grad():
        t = h
        targets = {}
        for L in range(1, stack.n_layers + 1):
            t = stack.run_block(L, t, pe)
            targets[L] = stack.loss_view(L, t)[0, T - A: T].detach()
    label_ids = ids[0, T - A + 1: T]
    return h, pe, targets, label_ids, T - A, A


def _grad_norm(stack, L):
    return sum(float(p.grad.abs().sum()) for p in stack.block_params(L)
               if p.grad is not None)


@cuda
def test_body_window_grads_confined(stack):
    h, pe, targets, label_ids, s0, A = _setup(stack)
    stack.model.zero_grad(set_to_none=True)
    loss_fn = HiddenLoss("nmse")
    # connected body window [3..5], no CE
    win_targets = {L: targets[L] for L in (3, 4, 5)}
    losses, h_out = tail_step(stack, 3, h.detach(), pe, win_targets,
                              s0, A, 1, label_ids, loss_fn, ce_w=0.0, L1=5)
    assert len(losses) == 3
    assert _grad_norm(stack, 3) > 0 and _grad_norm(stack, 5) > 0
    assert _grad_norm(stack, 2) == 0 and _grad_norm(stack, 6) == 0
    assert h_out.requires_grad is False


@cuda
def test_tail_hidden_weight_zero_is_pure_ce(stack):
    h, pe, targets, label_ids, s0, A = _setup(stack)
    n = stack.n_layers
    loss_fn = HiddenLoss("nmse")
    L0 = n - 2
    stack.model.zero_grad(set_to_none=True)
    tail_targets = {L: targets[L] for L in range(L0, n + 1)}
    losses, _ = tail_step(stack, L0, h.detach(), pe, tail_targets,
                          s0, A, 1, label_ids, loss_fn, ce_w=0.5, hidden_w=0.0)
    # hidden losses still REPORTED (storage telemetry), but the only
    # gradient source is the CE — window blocks get grads, frozen head none
    assert all(v >= 0 for v in losses)
    assert _grad_norm(stack, L0) > 0 and _grad_norm(stack, n) > 0
    assert _grad_norm(stack, L0 - 1) == 0
    assert all(p.grad is None for p in stack.lm_head.parameters())


@cuda
def test_endpoint_sliding_window_semantics(stack):
    """Faithful mode: loss ONLY at the window endpoint; ALL covered blocks
    updated; vocabulary untouched."""
    h, pe, targets, label_ids, s0, A = _setup(stack)
    stack.model.zero_grad(set_to_none=True)
    loss_fn = HiddenLoss("nmse")
    # endpoint L1=6, window [3..6]: sparse targets dict
    losses, _ = tail_step(stack, 3, h.detach(), pe, {6: targets[6]},
                          s0, A, 1, label_ids, loss_fn, ce_w=0.0, L1=6)
    assert len(losses) == 1  # only the endpoint is matched
    for L in (3, 4, 5, 6):
        assert _grad_norm(stack, L) > 0, L  # all covered blocks updated
    assert _grad_norm(stack, 2) == 0 and _grad_norm(stack, 7) == 0
    assert all(p.grad is None for p in stack.embed_tokens.parameters())
    assert all(p.grad is None for p in stack.lm_head.parameters())


@cuda
def test_teacher_kl_readout(stack):
    """teacher_kl: readout driven by the teacher's own context-conditioned
    logits (derived from targets[n]) — no label_ids labels touch the gradient."""
    h, pe, targets, label_ids, s0, A = _setup(stack)
    n = stack.n_layers
    stack.model.zero_grad(set_to_none=True)
    loss_fn = HiddenLoss("nmse")
    tail_targets = {L: targets[L] for L in range(n - 2, n + 1)}
    losses, _ = tail_step(stack, n - 2, h.detach(), pe, tail_targets,
                          s0, A, 1, None, loss_fn, ce_w=0.5, hidden_w=0.0,
                          ce_kind="teacher_kl")
    assert _grad_norm(stack, n) > 0 and _grad_norm(stack, n - 2) > 0
    assert _grad_norm(stack, n - 3) == 0
    assert all(p.grad is None for p in stack.lm_head.parameters())
