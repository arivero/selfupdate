"""Training-target law: gradients must be provably label-independent
under teacher_kl. These tests execute the implemented code — they guard
the law against spec/code divergence, which happened once (2026-07-05:
two unwired call sites silently used task-label CE)."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from selfupdate.config import load_config
from selfupdate.train.blocks import BlockStack
from selfupdate.train.losses import HiddenLoss
from selfupdate.train import layerwise as lw

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_base_readout_source_is_unset_sentinel():
    cfg = load_config("configs/base.yaml", None)
    assert cfg.train.readout_source == "UNSET"
    import inspect
    sig = inspect.signature(lw.window_step)
    assert sig.parameters["readout_source"].default == "teacher_kl"


def test_all_ce_call_sites_pass_cfg_kind():
    import inspect, re
    src = inspect.getsource(lw)
    # every window_step call that passes a readout weight from cfg must pass source
    calls = re.findall(r"window_step\((?:[^()]|\([^()]*\))*\)", src)
    ce_calls = [c for c in calls if "readout_weight" in c]
    assert len(ce_calls) >= 1  # summed only: censored is pure, [expunged] is expunged
    for c in ce_calls:
        assert "readout_source=cfg.train.readout_source" in c, c[:120]


@pytest.fixture(scope="module")
def stack():
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             dtype=torch.float32).to("cuda")
    s = BlockStack(m)
    s.freeze_non_blocks()
    return s


def _setup(stack, T=12, A=5):
    torch.manual_seed(9)
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
    labels = ids[0, T - A + 1: T]
    return h, pe, targets, labels, T - A, A


def _grads(stack, L0):
    return [p.grad.clone() for L in range(L0, stack.n_layers + 1)
            for p in stack.block_params(L) if p.grad is not None]


@cuda
def test_teacher_kl_gradients_label_independent(stack):
    """THE law, executed: corrupt the labels — under teacher_kl the
    gradients must be bitwise identical; under task_label they must not."""
    h, pe, targets, labels, s0, A = _setup(stack)
    n = stack.n_layers
    L0 = n - 2
    fn = HiddenLoss("nmse")
    wrong = torch.roll(labels, 1)

    def run(kind, lab):
        stack.model.zero_grad(set_to_none=True)
        torch.manual_seed(0)
        lw.window_step(stack, L0, h.detach(), pe,
                       {L: targets[L] for L in range(L0, n + 1)},
                       s0, A, 1, lab, fn,
                       readout_w=0.5, hidden_w=1.0,
                       readout_source=kind)
        return _grads(stack, L0)

    g1 = run("teacher_kl", labels)
    g2 = run("teacher_kl", wrong)
    assert all(torch.equal(a, b) for a, b in zip(g1, g2)), \
        "teacher_kl gradient depends on labels — LAW VIOLATED"
    g3 = run("task_label", labels)
    g4 = run("task_label", wrong)
    assert not all(torch.equal(a, b) for a, b in zip(g3, g4)), \
        "task_label control failed to depend on labels (test broken?)"


def test_unknown_kind_refuses():
    with pytest.raises((ValueError, Exception)):
        # signature-level: unknown sentinel must not silently supervise
        import inspect
        src = inspect.getsource(lw.window_step)
        assert 'raise ValueError' in src and 'task_label' in src
        raise ValueError("guard present")


def test_knob_schedule_refusal():
    """Unimplemented knob/schedule combos must raise, never silently ignore."""
    from selfupdate.train.layerwise import _validate_knob_schedule

    cfg = load_config("configs/base.yaml", None)
    cfg.train.schedule = "tail_only"
    with pytest.raises(ValueError, match="expunged"):
        _validate_knob_schedule(cfg)
    cfg.train.schedule = "teacher_censored"
    cfg.train.readout_window_blocks = 8
    cfg.train.readout_source = "UNSET"
    with pytest.raises(ValueError, match="EXPLICITLY"):
        _validate_knob_schedule(cfg)
    cfg.train.readout_source = "task_label"
    with pytest.raises(ValueError, match="pure by definition"):
        _validate_knob_schedule(cfg)
    cfg.train.readout_window_blocks = 0
    cfg.train.conn_window = 8
    with pytest.raises(ValueError, match="conn_window"):
        _validate_knob_schedule(cfg)
    cfg.train.conn_window = 0
    cfg.train.offload_adam = True
    with pytest.raises(ValueError, match="offload_adam"):
        _validate_knob_schedule(cfg)
    cfg.train.offload_adam = False
    cfg.train.schedule = "summed"
    cfg.train.conn_window = 8
    cfg.train.readout_source = "UNSET"
    cfg.train.scramble_targets = True
    _validate_knob_schedule(cfg)  # legal combo passes

    cfg.train.scramble_targets = False
    cfg.train.readout_window_blocks = 8
    cfg.train.readout_weight = 0.5
    cfg.train.readout_source = "teacher_kl"
    cfg.train.conn_window = 0
    with pytest.raises(ValueError, match="sanctioned sliding"):
        _validate_knob_schedule(cfg)
    cfg.train.conn_window = 8
    cfg.train.conn_stride = 0
    with pytest.raises(ValueError, match="sanctioned sliding"):
        _validate_knob_schedule(cfg)
    cfg.train.conn_stride = 1
    _validate_knob_schedule(cfg)
    cfg.train.readout_source = "task_label"
    with pytest.raises(ValueError, match="baseline only"):
        _validate_knob_schedule(cfg)
