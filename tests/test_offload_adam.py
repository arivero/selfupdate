"""offload_adam: paging Adam moments to CPU must not change the math."""

import pytest
import torch

from selfupdate.train.layerwise import _move_opt_state

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
