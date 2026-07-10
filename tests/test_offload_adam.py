"""offload_adam: paging Adam moments to CPU must not change the math."""

import pytest
import torch

from selfupdate.config import ExperimentConfig
from selfupdate.train.layerwise import _move_opt_state
from selfupdate.train.runtime import OptimizerPlan

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@cuda
def test_paged_steps_bitwise_match_resident():
    torch.manual_seed(11)
    w0 = torch.randn(64, 64, device="cuda")
    a = torch.nn.Parameter(w0.clone())
    b = torch.nn.Parameter(w0.clone())
    oa = torch.optim.AdamW([a], lr=1e-3)
    ob = torch.optim.AdamW([b], lr=1e-3)
    for i in range(4):
        g = torch.randn(64, 64, device="cuda")
        a.grad = g.clone()
        b.grad = g.clone()
        oa.step(); oa.zero_grad(set_to_none=True)
        _move_opt_state(ob, "cuda")
        ob.step(); ob.zero_grad(set_to_none=True)
        _move_opt_state(ob, "cpu")
        # moments really left the GPU
        for st in ob.state.values():
            assert all(v.device.type == "cpu" for v in st.values()
                       if torch.is_tensor(v) and v.dim() > 0)
    assert torch.equal(a.detach(), b.detach())


class _Blocks(torch.nn.Module):
    """Minimal block_params provider for OptimizerPlan."""

    def __init__(self, n=3, d=32):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(d, d) for _ in range(n)])
        self.n_layers = n

    def block_params(self, L):
        return list(self.blocks[L - 1].parameters())


@cuda
def test_streamed_offload_plan_matches_resident_bitwise():
    torch.manual_seed(23)
    base = _Blocks().to("cuda")
    import copy

    resident_m = copy.deepcopy(base)
    offload_m = copy.deepcopy(base)
    cfg_r, cfg_o = ExperimentConfig(), ExperimentConfig()
    cfg_o.train.offload_adam = True
    resident = OptimizerPlan.build(resident_m, cfg_r)
    offload = OptimizerPlan.build(offload_m, cfg_o)
    assert resident.kind == "full_resident" and offload.kind == "full_offload"

    for step in range(5):
        for L in range(1, base.n_layers + 1):
            for pr, po in zip(resident_m.block_params(L),
                              offload_m.block_params(L)):
                g = torch.randn(pr.shape, device="cuda",
                                generator=torch.Generator("cuda").manual_seed(
                                    1000 * step + L * 10 + pr.dim()))
                pr.grad = g.clone()
                po.grad = g.clone()
        resident.step()
        offload.step()
        # moments really live in pinned host memory between steps
        for opt in offload.optimizers:
            for st in opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v) and v.dim() > 0:
                        assert v.device.type == "cpu" and v.is_pinned()

    for L in range(1, base.n_layers + 1):
        for pr, po in zip(resident_m.block_params(L),
                          offload_m.block_params(L)):
            assert torch.equal(pr.detach(), po.detach()), f"block {L} diverged"
