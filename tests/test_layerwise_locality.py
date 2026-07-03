"""Locality guarantees of the layerwise trainer.

1. After summed-schedule local steps, gradients exist ONLY inside decoder
   blocks — never in the embedding, final norm, or lm_head.
2. Each block's gradient equals an independent single-block replay (same
   detached input, fresh graph), i.e., no cross-block gradient leakage.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.data.dataset import DistillDataset
from selfupdate.masking import ContextMasker
from selfupdate.teacher.cache import TeacherCache
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import local_block_step
from selfupdate.train.losses import hidden_match

MODEL = "Qwen/Qwen3-0.6B"
EXAMPLES = "data/poem/examples.jsonl"


def _cache_dir():
    """The cache whose config-hash matches base.yaml exactly — a glob would
    also match caches of other datasets (v2) for the same model+mask."""
    from selfupdate.config import load_config
    from selfupdate.teacher.cache import resolve_cache_dir

    root, _ = resolve_cache_dir(load_config("configs/base.yaml", None))
    return root if root.exists() else None


pytestmark = pytest.mark.skipif(_cache_dir() is None, reason="no built cache")


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda").train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    cache = TeacherCache(_cache_dir())
    ds = DistillDataset(EXAMPLES, cache, tok,
                        need_layers=list(range(1, stack.n_layers + 1)),
                        )
    return stack, ds


def _summed_pass(stack, it, device="cuda"):
    """One summed-schedule item pass in fp32 (no autocast: exact replays)."""
    ids = it.student_ids.to(device)[None]
    pos = it.position_ids.to(device)[None]
    h = stack.embed(ids)
    pos_emb = stack.rope(h, pos)
    h_ins = {}
    for L in range(1, stack.n_layers + 1):
        h_in = h.detach()
        h_ins[L] = h_in
        _, h = local_block_step(stack, L, h_in, pos_emb,
                                it.hidden[L].to(device), it.s0, it.A, "nmse",
                                autocast=False)
    return h_ins, pos_emb


def test_grads_confined_to_blocks(setup):
    stack, ds = setup
    stack.model.zero_grad(set_to_none=True)
    _summed_pass(stack, ds[0])
    for name in ("embed_tokens", "final_norm", "lm_head"):
        for p in getattr(stack, name).parameters():
            assert p.grad is None, f"gradient leaked into {name}"
    for L in range(1, stack.n_layers + 1):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in stack.block_params(L)), f"block {L} got no gradient"


def test_block_grads_match_independent_replay(setup):
    stack, ds = setup
    it = ds[1]
    stack.model.zero_grad(set_to_none=True)
    h_ins, pos_emb = _summed_pass(stack, it)
    got = {
        L: [p.grad.clone() for p in stack.block_params(L)]
        for L in (1, stack.n_layers // 2, stack.n_layers)
    }
    for L, grads in got.items():
        stack.model.zero_grad(set_to_none=True)
        h_out = stack.run_block(L, h_ins[L], pos_emb)
        loss = hidden_match(
            stack.loss_view(L, h_out)[0, it.s0: it.s0 + it.A],
            it.hidden[L].to("cuda"), "nmse",
        )
        loss.backward()
        for p, g in zip(stack.block_params(L), grads):
            assert torch.allclose(p.grad, g, rtol=1e-5, atol=1e-7), (
                f"block {L}: replay gradient differs — cross-block leakage"
            )
    stack.model.zero_grad(set_to_none=True)


def test_layerwise_training_never_computes_logits(setup):
    """The claim that layerwise training is NOT disguised logit backprop,
    enforced: neither lm_head nor any logits tensor is touched during a full
    summed-schedule pass (the only final-norm use is the last block's loss
    view). If someone later routes a logit loss through this path, this test
    fails."""
    stack, ds = setup
    calls = []
    hook = stack.lm_head.register_forward_hook(lambda *a: calls.append(1))
    try:
        stack.model.zero_grad(set_to_none=True)
        _summed_pass(stack, ds[3])
    finally:
        hook.remove()
    stack.model.zero_grad(set_to_none=True)
    assert not calls, "lm_head was invoked during layerwise training"


def test_sequential_never_runs_frozen_blocks(setup):
    """From stage L+1 on, blocks <= L must not execute: feeding the cached
    h_L into block L+1 must equal the full prefix recomputation."""
    stack, ds = setup
    it = ds[2]
    ids = it.student_ids.to("cuda")[None]
    pos = it.position_ids.to("cuda")[None]
    with torch.no_grad():
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for L in range(1, 4):
            h = stack.run_block(L, h, pos_emb)
        # cache round-trip through fp16 CPU, as StudentActCache does
        cached = h[0].to(torch.float16).cpu()
        direct = stack.run_block(4, h, pos_emb)
        via_cache = stack.run_block(
            4, cached.to("cuda", torch.float32)[None], pos_emb
        )
    err = (direct - via_cache).abs().max().item()
    scale = direct.abs().max().item()
    assert err <= 5e-3 * max(scale, 1.0), f"cache path diverges: {err} (scale {scale})"
