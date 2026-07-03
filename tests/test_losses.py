import torch

from selfupdate.train.losses import hidden_match


def test_hidden_match_scale_invariance_of_nmse():
    torch.manual_seed(3)
    hs, ht = torch.randn(10, 32), torch.randn(10, 32)
    a = hidden_match(hs, ht, "nmse")
    b = hidden_match(hs * 10, ht * 10, "nmse")
    assert torch.allclose(a, b, rtol=1e-4)
    assert hidden_match(ht, ht, "nmse").item() < 1e-10
    assert hidden_match(ht, ht, "l2mse").item() < 1e-10
