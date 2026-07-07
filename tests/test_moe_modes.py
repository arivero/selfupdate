"""MoE routing modes (train.moe_mode): capture, forcing, alignment, tripwires.

CPU-only tiny random models — no GPU, no cache, no downloads. Covers both
wrapped families: gpt-oss (router inside mlp, instance-level Python forward)
and gemma4 MoE (router on the decoder layer).
"""

import types

import pytest
import torch

from selfupdate.config import load_config
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import (
    _moe_row_maps,
    _validate_knob_schedule,
    local_block_step,
)
from selfupdate.train.moe import MoEController, pending_router_loss


def make_model(family: str):
    torch.manual_seed(0)
    if family == "gptoss":
        from transformers import GptOssConfig, GptOssForCausalLM

        cfg = GptOssConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, num_local_experts=4, num_experts_per_tok=2,
            max_position_embeddings=64, sliding_window=16)
        return GptOssForCausalLM(cfg)
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, enable_moe_block=True, num_experts=4, top_k_experts=2,
        moe_intermediate_size=32, max_position_embeddings=64,
        hidden_size_per_layer_input=0)
    return Gemma4ForCausalLM(cfg)


def walk(stack, ids, grad=False):
    """Plain block walk, returning the final hidden state."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        h = stack.embed(ids)
        pos = torch.arange(ids.shape[1])[None]
        pe = stack.rope(h, pos)
        for L in range(1, stack.n_layers + 1):
            h = stack.run_block(L, h, pe)
            if not grad:
                h = h.detach()
    return h


FAMILIES = ["gptoss", "gemma4"]


@pytest.mark.parametrize("family", FAMILIES)
def test_passthrough_identical(family):
    """With no phase active the wrapped modules are exact passthroughs."""
    model = make_model(family)
    ids = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        before = model(ids).logits
    MoEController(BlockStack(model), "teacher_forced")
    with torch.no_grad():
        after = model(ids).logits
    assert torch.equal(before, after)


def test_dense_model_raises():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    with pytest.raises(ValueError, match="no MoE layers"):
        MoEController(BlockStack(model), "teacher_forced")


@pytest.mark.parametrize("family", FAMILIES)
def test_teacher_capture_and_forced_identity(family):
    """Teacher pass records top-k per MoE layer; forcing the student with an
    identity row map reproduces the natural forward exactly (overlap 1)."""
    model = make_model(family)
    stack = BlockStack(model)
    T = 12
    ids = torch.randint(0, 128, (1, T))
    natural = walk(stack, ids)

    ctrl = MoEController(stack, "teacher_forced")
    with ctrl.teacher_phase():
        walk(stack, ids)
    assert set(ctrl.t_idx) == {1, 2}
    assert ctrl.t_idx[1].shape == (T, 2)

    ctrl.set_maps(torch.arange(T), torch.ones(T, dtype=torch.bool))
    with ctrl.student_phase():
        forced = walk(stack, ids)
    assert torch.allclose(natural, forced, atol=1e-5)
    overlap = ctrl.overlap_flush()
    assert set(overlap) == {1, 2}
    assert all(v == 1.0 for v in overlap.values())


@pytest.mark.parametrize("family", FAMILIES)
def test_forced_scrambled_map_changes_output(family):
    """Routing through the teacher's choices for the WRONG rows must change
    the forward — proof the forcing path actually steers the experts."""
    model = make_model(family)
    stack = BlockStack(model)
    T = 12
    ids = torch.randint(0, 128, (1, T))
    natural = walk(stack, ids)

    ctrl = MoEController(stack, "teacher_forced")
    with ctrl.teacher_phase():
        walk(stack, ids)
    ctrl.set_maps(torch.arange(T).roll(5), torch.ones(T, dtype=torch.bool))
    with ctrl.student_phase():
        forced = walk(stack, ids)
    assert not torch.allclose(natural, forced, atol=1e-5)


@pytest.mark.parametrize("family", FAMILIES)
def test_router_aligned_kl_and_grad(family):
    """KL is ~0 against an identical teacher, grows after perturbing the
    router, and backpropagates into the router parameters."""
    model = make_model(family)
    stack = BlockStack(model)
    T = 12
    ids = torch.randint(0, 128, (1, T))

    ctrl = MoEController(stack, "router_aligned", router_weight=1.0)
    with ctrl.teacher_phase():
        walk(stack, ids)
    ctrl.set_maps(torch.arange(T), torch.ones(T, dtype=torch.bool))
    with ctrl.student_phase():
        walk(stack, ids, grad=True)
        kl0 = pending_router_loss()
        # not exactly 0: teacher log-probs are stored bf16
        assert kl0 is not None and kl0.item() < 5e-5

        block = stack.blocks[0]
        router_param = (block.mlp.router.weight if family == "gptoss"
                        else block.router.proj.weight)
        with torch.no_grad():
            router_param.add_(torch.randn_like(router_param))
        walk(stack, ids, grad=True)
        kl1 = pending_router_loss()
        assert kl1 is not None and kl1.item() > 1e-3
        kl1.backward()
        assert router_param.grad is not None


@pytest.mark.parametrize("family", FAMILIES)
def test_step_function_drains_pending(family):
    """local_block_step must consume pending router losses (graph-leak
    tripwire fires otherwise on phase exit)."""
    model = make_model(family)
    stack = BlockStack(model)
    T = 8
    ids = torch.randint(0, 128, (1, T))

    ctrl = MoEController(stack, "router_aligned", router_weight=1.0)
    with ctrl.teacher_phase():
        walk(stack, ids)
    with torch.no_grad():
        h0 = stack.embed(ids)
        pos = torch.arange(T)[None]
        pe = stack.rope(h0, pos)
        target = stack.run_block(1, h0, pe).squeeze(0)

    ctrl.set_maps(torch.arange(T), torch.ones(T, dtype=torch.bool))
    with ctrl.student_phase():
        local_block_step(stack, 1, h0.detach(), pe, target, 0, T, "l2mse",
                         autocast=False)
        assert pending_router_loss() is None  # drained by the step


def test_student_phase_tripwires():
    model = make_model("gptoss")
    stack = BlockStack(model)
    ctrl = MoEController(stack, "teacher_forced")
    with pytest.raises(RuntimeError, match="set_maps"):
        with ctrl.student_phase():
            pass
    ids = torch.randint(0, 128, (1, 8))
    with ctrl.teacher_phase():
        walk(stack, ids)
    # row-count mismatch: an anchor/eval-shaped forward inside student_phase
    ctrl.set_maps(torch.arange(8), torch.ones(8, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="rows"):
        with ctrl.student_phase():
            walk(stack, torch.randint(0, 128, (1, 5)))


def test_qwen35_adapter_isolated_block():
    """qwen3_5_moe's SparseMoeBlock: the full composite (vision+text) causal
    LM is fiddly to build tiny, so exercise the adapter against an isolated
    SparseMoeBlock — passthrough when idle, on_router called with the right
    shapes and a scrambled forced map that changes the output."""
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )
    from selfupdate.train.moe import _Qwen35MoE

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=32, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        intermediate_size=64)
    torch.manual_seed(0)
    mlp = Qwen3_5MoeSparseMoeBlock(cfg)
    block = types.SimpleNamespace(mlp=mlp)
    assert _Qwen35MoE.match(block)

    class FakeCtrl:
        phase = None
        force = None

        def on_router(self, L, logits, nat):
            self.seen = (L, tuple(logits.shape), tuple(nat.shape))
            return self.force if self.force is not None else nat

    fc = FakeCtrl()
    _Qwen35MoE(block).install(fc, 7)
    x = torch.randn(6, 32)
    # idle: gate returns natural routing untouched
    _, nat_w, nat_idx = mlp.gate(x)
    fc.phase = "student"
    _, w_pt, idx_pt = mlp.gate(x)
    assert torch.equal(idx_pt, nat_idx) and torch.allclose(w_pt, nat_w)
    assert fc.seen == (7, (6, 8), (6, 2))
    # force a disjoint expert set -> the gate must return exactly those indices
    # with renormalized weights (independent of the experts-module dispatch)
    fc.force = torch.tensor([[7, 6]] * 6)
    _, w_f, idx_f = mlp.gate(x)
    assert torch.equal(idx_f, fc.force)
    assert torch.allclose(w_f.sum(-1), torch.ones(6), atol=1e-4)


def _tiny_qwen35_hybrid():
    """A 2-layer Qwen3.5-MoE (1 GatedDeltaNet linear-attention + 1 full-
    attention) built from the real config with shrunk dims. Verifies the
    layerwise trainer drives HYBRID attention stacks — the linear-attention
    block runs and backprops through BlockStack unchanged."""
    import json

    from huggingface_hub import hf_hub_download
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    raw = json.load(open(hf_hub_download("Qwen/Qwen3.5-122B-A10B", "config.json")))
    tc = dict(raw.get("text_config", raw))
    tc.pop("quantization_config", None)
    tc.update(dict(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4))
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**tc))


@pytest.mark.skipif(
    __import__("shutil").which("hf") is None and not __import__("os").environ.get("HF_HOME"),
    reason="needs network/hub cache for the real Qwen3.5 config")
def test_hybrid_linear_attention_trains():
    from selfupdate.train.blocks import BlockStack
    from selfupdate.train.layerwise import local_block_step

    try:
        m = _tiny_qwen35_hybrid()
    except Exception as e:  # offline / hub unavailable
        pytest.skip(f"cannot build tiny Qwen3.5: {e}")
    st = BlockStack(m)
    assert [b.layer_type for b in st.blocks] == ["linear_attention", "full_attention"]
    ids = torch.randint(0, 128, (1, 10))
    h = st.embed(ids).detach()
    pe = st.rope(h, torch.arange(10)[None])
    # forward across the hybrid stack is finite
    t = h
    for L in range(1, st.n_layers + 1):
        t = st.run_block(L, t, pe)
        assert torch.isfinite(t).all()
    # backward is local to the LINEAR-attention block and reaches its params
    for p in st.block_params(1):
        p.requires_grad_(True)
    target = torch.randn(10, 32)
    loss, _ = local_block_step(st, 1, h, pe, target, 0, 10, "l2mse", autocast=False)
    grads = [p.grad for p in st.block_params(1) if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_row_maps_item_and_batch():
    it = types.SimpleNamespace(s0=2, t0=5, A=4, t_priv=None)
    rows, mask = _moe_row_maps(it, "cpu")
    assert rows.tolist() == [0, 1, 5, 6, 7, 8]
    assert mask.all() and len(mask) == 6


def test_moe_knob_validation():
    cfg = load_config("configs/base.yaml", None)
    cfg.train.schedule = "summed"
    cfg.train.lora.enabled = True
    cfg.train.online_teacher = True
    cfg.train.moe_mode = "teacher_forced"
    _validate_knob_schedule(cfg)  # legal

    cfg.train.moe_mode = "router_aligned"
    with pytest.raises(ValueError, match="moe_router_weight"):
        _validate_knob_schedule(cfg)  # weight not pinned
    cfg.train.moe_router_weight = 0.1
    _validate_knob_schedule(cfg)  # legal

    cfg.train.online_teacher = False
    with pytest.raises(ValueError, match="online_teacher"):
        _validate_knob_schedule(cfg)
    cfg.train.online_teacher = True

    cfg.train.schedule = "teacher_censored"
    with pytest.raises(ValueError, match="summed"):
        _validate_knob_schedule(cfg)
    cfg.train.schedule = "summed"

    cfg.train.moe_mode = "dense_or_black_box"
    with pytest.raises(ValueError, match="moe_router_weight"):
        _validate_knob_schedule(cfg)  # stray weight without the mode
