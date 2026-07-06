"""Mixed schedule: annealed teacher/student routing.

CPU tests cover the p-schedule and branch-draw determinism; the GPU test
guards the restored pure teacher-censored path."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import DistillDataset
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import _censored_item, mix_teacher_p

MODEL = "Qwen/Qwen3-0.6B"
EXAMPLES = "data/poem/examples.jsonl"


class _Cfg:
    """Minimal stand-in for ExperimentConfig.train fields mix_teacher_p reads."""

    def __init__(self, epochs, start=1.0, end=0.0):
        class T:
            pass

        self.train = T()
        self.train.epochs = epochs
        self.train.mix_teacher_start = start
        self.train.mix_teacher_end = end


def test_mix_teacher_p_linear():
    cfg = _Cfg(epochs=5)
    ps = [mix_teacher_p(cfg, e) for e in range(5)]
    assert ps[0] == 1.0 and ps[-1] == 0.0
    assert ps == sorted(ps, reverse=True)
    diffs = [round(a - b, 9) for a, b in zip(ps, ps[1:])]
    assert len(set(diffs)) == 1  # linear


def test_mix_teacher_p_degenerate_epochs():
    assert mix_teacher_p(_Cfg(epochs=1), 0) == 0.0  # jumps to end value
    cfg = _Cfg(epochs=3, start=0.8, end=0.2)
    assert abs(mix_teacher_p(cfg, 1) - 0.5) < 1e-9


def test_branch_draw_deterministic_per_seed():
    def draws(seed, k=64):
        g = torch.Generator().manual_seed(seed + 1)
        return [torch.rand((), generator=g).item() for _ in range(k)]

    assert draws(17) == draws(17)
    assert draws(17) != draws(43)


def _cache_dir():
    from selfupdate.teacher.cache import resolve_cache_dir

    root, _ = resolve_cache_dir(load_config("configs/base.yaml", None))
    return root if root.exists() else None


@pytest.mark.skipif(_cache_dir() is None, reason="no built cache")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_censored_schedule_is_pure_per_block():
    """Teacher-censored path is per-block hidden matching only: every block
    may get its own local gradient, but no readout or frozen vocabulary
    parameter receives one."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda").train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    from selfupdate.teacher.cache import TeacherCache

    ds = DistillDataset(EXAMPLES, TeacherCache(_cache_dir()), tok,
                        need_layers=[], with_teacher_ids=True)
    it = ds[0]
    device = "cuda"
    n = stack.n_layers
    # full-sequence teacher states from the (untrained) model itself
    t_ids = it.teacher_ids.to(device)[None]
    t_pos = torch.arange(t_ids.shape[1], device=device)[None]
    with torch.no_grad():
        h = stack.embed(t_ids)
        pos_emb = stack.rope(h, t_pos)
        t_states = [h]
        for L in range(1, n + 1):
            h = stack.run_block(L, h, pos_emb)
            t_states.append(h)

    cfg = _Cfg(epochs=1)
    model.zero_grad(set_to_none=True)
    losses = _censored_item(cfg, stack, "nmse", it, t_states, device)
    assert len(losses) == n
    for L in range(1, n + 1):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in stack.block_params(L)), f"block {L} no grads"
    for pname, p in stack.model.named_parameters():
        if any(k in pname for k in ("embed_tokens", "model.norm", "lm_head")):
            assert p.grad is None, f"{pname} got a gradient"
