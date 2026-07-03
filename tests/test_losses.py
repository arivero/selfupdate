import pytest
import torch
import torch.nn as nn

from selfupdate.train.losses import HiddenLoss, hidden_match

H, V = 32, 97


def _vocab_loss(kind):
    torch.manual_seed(5)
    norm = nn.RMSNorm(H)
    head = nn.Linear(H, V, bias=False)
    return HiddenLoss(kind, final_norm=norm, lm_head=head)


def test_hidden_match_scale_invariance_of_nmse():
    torch.manual_seed(3)
    hs, ht = torch.randn(10, 32), torch.randn(10, 32)
    a = hidden_match(hs, ht, "nmse")
    b = hidden_match(hs * 10, ht * 10, "nmse")
    assert torch.allclose(a, b, rtol=1e-4)
    assert hidden_match(ht, ht, "nmse").item() < 1e-10
    assert hidden_match(ht, ht, "l2mse").item() < 1e-10


def test_hidden_match_huber_scale_invariant_zero_at_identity():
    torch.manual_seed(4)
    hs, ht = torch.randn(10, 32), torch.randn(10, 32)
    a = hidden_match(hs, ht, "huber")
    b = hidden_match(hs * 10, ht * 10, "huber")
    assert torch.allclose(a, b, rtol=1e-4)
    assert hidden_match(ht, ht, "huber").item() < 1e-10
    assert a.item() > 0


def test_hidden_match_cosine():
    torch.manual_seed(4)
    ht = torch.randn(10, 32)
    assert hidden_match(ht, ht, "cosine").item() < 1e-6
    # per-row positive rescaling of the student is invisible to cosine
    scaled = ht * torch.rand(10, 1).clamp_min(0.1)
    assert hidden_match(scaled, ht, "cosine").item() < 1e-6
    assert abs(hidden_match(-ht, ht, "cosine").item() - 2.0) < 1e-6
    assert hidden_match(torch.randn(10, 32), ht, "cosine").item() > 0


def test_hidden_match_unknown_kind_raises():
    t = torch.randn(4, 8)
    with pytest.raises(ValueError):
        hidden_match(t, t, "cka")
    with pytest.raises(ValueError):
        HiddenLoss("cka")


def test_vocab_kinds_require_norm_and_head():
    with pytest.raises(ValueError):
        HiddenLoss("vocab_mse")
    with pytest.raises(ValueError):
        HiddenLoss("lens_kl", final_norm=nn.RMSNorm(H))


def test_vocab_mse_matches_logit_space_mse():
    """Gram-matrix path == direct MSE in logit space (same normalizer)."""
    loss_fn = _vocab_loss("vocab_mse")
    torch.manual_seed(7)
    hs, ht = torch.randn(10, H), torch.randn(10, H)
    got = loss_fn(hs, ht, normed=True)
    W = loss_fn.lm_head.weight
    d = (hs - ht) @ W.T
    t = ht @ W.T
    want = d.pow(2).sum(-1).mean() / t.pow(2).sum(-1).mean()
    assert torch.allclose(got, want, rtol=1e-4)


def test_vocab_mse_joint_scale_invariant_zero_at_identity():
    loss_fn = _vocab_loss("vocab_mse")
    torch.manual_seed(8)
    hs, ht = torch.randn(10, H), torch.randn(10, H)
    a = loss_fn(hs, ht, normed=True)
    b = loss_fn(hs * 10, ht * 10, normed=True)
    assert torch.allclose(a, b, rtol=1e-4)
    assert loss_fn(ht, ht, normed=True).item() < 1e-10
    assert loss_fn(ht, ht, normed=False).item() < 1e-10


def test_lens_kl_zero_at_identity_positive_otherwise():
    loss_fn = _vocab_loss("lens_kl")
    torch.manual_seed(9)
    hs, ht = torch.randn(10, H), torch.randn(10, H)
    assert loss_fn(ht, ht, normed=True).item() < 1e-8
    assert loss_fn(ht.clone(), ht, normed=False).item() < 1e-8
    assert loss_fn(hs, ht, normed=True).item() > 0


def test_vocab_kinds_gradient_reaches_student_not_vocab():
    """Gradient flows to the student slice but never into the frozen
    norm/head params — the frozen-vocabulary principle at the loss level."""
    for kind in ("vocab_mse", "lens_kl"):
        loss_fn = _vocab_loss(kind)
        loss_fn.final_norm.requires_grad_(False)
        loss_fn.lm_head.requires_grad_(False)
        torch.manual_seed(10)
        hs = torch.randn(10, H, requires_grad=True)
        ht = torch.randn(10, H)
        loss_fn(hs, ht, normed=False).backward()
        assert hs.grad is not None and hs.grad.abs().sum() > 0
        assert loss_fn.lm_head.weight.grad is None
        for p in loss_fn.final_norm.parameters():
            assert p.grad is None


def test_hidden_loss_geometric_delegates():
    loss_fn = HiddenLoss("nmse")
    torch.manual_seed(11)
    hs, ht = torch.randn(10, H), torch.randn(10, H)
    assert torch.allclose(loss_fn(hs, ht), hidden_match(hs, ht, "nmse"))
