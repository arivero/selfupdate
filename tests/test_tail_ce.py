"""Tail-CE hybrid: gradient confinement of the joint tail window.

The contract: blocks inside [L0..n] train jointly (CE credit crosses block
boundaries within the window — the whole point, and what single-block CE
lacked), while nothing below L0 and none of the frozen non-block params ever
see a gradient.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.data.dataset import DistillDataset
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import tail_step

MODEL = "Qwen/Qwen3-0.6B"
EXAMPLES = "data/poem/examples.jsonl"
K = 4  # tail window size under test


def _cache_dir():
    from selfupdate.config import load_config
    from selfupdate.teacher.cache import resolve_cache_dir

    root, _ = resolve_cache_dir(load_config("configs/base.yaml", None))
    return root if root.exists() else None


pytestmark = pytest.mark.skipif(_cache_dir() is None, reason="no built cache")


@pytest.fixture(scope="module")
def setup():
    from selfupdate.teacher.cache import TeacherCache

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda").train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    ds = DistillDataset(EXAMPLES, TeacherCache(_cache_dir()), tok,
                        need_layers=list(range(1, stack.n_layers + 1)),
                        need_logits=False)
    return stack, ds


def _run_tail(stack, it, ce_w, device="cuda"):
    n = stack.n_layers
    L0 = n - K + 1
    ids = it.student_ids.to(device)[None]
    pos = it.position_ids.to(device)[None]
    with torch.no_grad():
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for L in range(1, L0):
            h = stack.run_block(L, h, pos_emb)
    targets = {L: it.hidden[L].to(device) for L in range(L0, n + 1)}
    gold = ids[0, it.ans0: it.s0 + it.A]
    tail_step(stack, L0, h.detach(), pos_emb, targets, it.s0, it.A,
              it.ans0 - it.s0, gold, "nmse", ce_w, autocast=False)
    return L0


def _grad(p_list):
    return [p.grad.clone() for p in p_list if p.grad is not None]


def test_tail_grads_confined_to_window(setup):
    stack, ds = setup
    stack.model.zero_grad(set_to_none=True)
    L0 = _run_tail(stack, ds[0], ce_w=1.0)
    for name in ("embed_tokens", "final_norm", "lm_head"):
        for pname, p in stack.model.named_parameters():
            if name in pname:
                assert p.grad is None, f"{pname} got a gradient"
    for L in range(1, L0):
        assert not _grad(stack.block_params(L)), f"block {L} below window got grads"
    for L in range(L0, stack.n_layers + 1):
        assert _grad(stack.block_params(L)), f"tail block {L} got no grads"


def test_ce_credit_crosses_block_boundaries(setup):
    """The CE at the top must change gradients of DEEPER-than-last tail
    blocks — the multi-block credit that last_block_ce could not provide."""
    stack, ds = setup
    L0_probe = stack.n_layers - K + 1  # deepest tail block below the last

    stack.model.zero_grad(set_to_none=True)
    _run_tail(stack, ds[0], ce_w=0.0)
    g0 = _grad(stack.block_params(L0_probe))

    stack.model.zero_grad(set_to_none=True)
    _run_tail(stack, ds[0], ce_w=10.0)
    g1 = _grad(stack.block_params(L0_probe))

    assert any(not torch.allclose(a, b, atol=1e-8) for a, b in zip(g0, g1)), (
        "CE weight had no effect on a non-last tail block: credit is not "
        "crossing block boundaries"
    )


def test_lens_ce_stays_block_local(setup):
    """Per-block lens-CE must not leak: only block L gets gradients, the
    frozen norm/head none, despite the loss passing through them."""
    from selfupdate.train.layerwise import local_block_step

    stack, ds = setup
    it = ds[0]
    device = "cuda"
    L = stack.n_layers // 2
    ids = it.student_ids.to(device)[None]
    pos = it.position_ids.to(device)[None]
    with torch.no_grad():
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for LL in range(1, L):
            h = stack.run_block(LL, h, pos_emb)
    stack.model.zero_grad(set_to_none=True)
    gold = ids[0, it.ans0: it.s0 + it.A]
    local_block_step(stack, L, h.detach(), pos_emb, it.hidden[L].to(device),
                     it.s0, it.A, "nmse", autocast=False,
                     lens_ce_w=1.0, gold=gold, ans_off=it.ans0 - it.s0)
    assert _grad(stack.block_params(L)), "block L got no grads"
    for LL in list(range(1, L)) + list(range(L + 1, stack.n_layers + 1)):
        assert not _grad(stack.block_params(LL)), f"block {LL} leaked"
    for name in ("embed_tokens", "final_norm", "lm_head"):
        for pname, p in stack.model.named_parameters():
            if name in pname:
                assert p.grad is None, f"{pname} got a gradient"
