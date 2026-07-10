import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    with pytest.raises(ValueError):
        HiddenLoss("delta_vocab_cos", final_norm=nn.RMSNorm(H))


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


def test_vocab_fisher_zero_at_identity_positive_otherwise():
    loss = _vocab_loss("vocab_fisher")
    torch.manual_seed(7)
    ht = torch.randn(9, H)
    assert loss(ht.clone(), ht).item() < 1e-10
    hs = ht + 0.3 * torch.randn(9, H)
    assert loss(hs, ht).item() > 0


def test_vocab_fisher_differs_from_vocab_mse_and_grads_flow():
    fisher = _vocab_loss("vocab_fisher")
    vmse = _vocab_loss("vocab_mse")
    torch.manual_seed(8)
    ht = torch.randn(7, H)
    hs = (ht + 0.5 * torch.randn(7, H)).requires_grad_(True)
    lf = fisher(hs, ht)
    lv = vmse(hs, ht)
    # same family, different metric: both positive, not equal
    assert lf.item() > 0 and lv.item() > 0
    assert abs(lf.item() - lv.item()) > 1e-6
    lf.backward()
    assert hs.grad is not None and torch.isfinite(hs.grad).all()
    # frozen head/norm receive no gradient
    assert all(p.grad is None for p in fisher.lm_head.parameters())


def test_vocab_fisher_weights_toward_teacher_support():
    """An error along a HIGH-probability token's unembedding row must cost
    more than the same-size error along a LOW-probability row."""
    torch.manual_seed(9)
    norm = torch.nn.Identity()
    head = torch.nn.Linear(H, V, bias=False)
    loss = HiddenLoss("vocab_fisher", final_norm=norm, lm_head=head)
    with torch.no_grad():
        head.weight[:] = torch.randn(V, H)
        ht = 3.0 * head.weight[0] / head.weight[0].norm()  # teacher peaks token 0
        ht = ht[None]
        logits = head(ht)
        lo = logits[0].argmin().item()
        e_hi = head.weight[0] / head.weight[0].norm()
        e_lo = head.weight[lo] / head.weight[lo].norm()
    l_hi = loss(ht + 0.1 * e_hi[None], ht)
    l_lo = loss(ht + 0.1 * e_lo[None], ht)
    assert l_hi.item() > l_lo.item()


def test_delta_losses_match_identical_updates_and_stop_previous_gradient():
    torch.manual_seed(21)
    prev = torch.randn(9, H, requires_grad=True)
    teacher_prev = torch.randn(9, H)
    teacher_delta = torch.randn(9, H)
    teacher = teacher_prev + teacher_delta
    student = (prev.detach() + teacher_delta).requires_grad_(True)
    for kind in ("delta_nmse", "delta_cosine"):
        loss = HiddenLoss(kind).delta(student, prev, teacher, teacher_prev)
        assert loss.item() < 1e-6, kind
    # A perturbed update has a gradient only through the newly produced state;
    # ``prev`` is deliberately a stop-gradient subtraction operand.
    loss = HiddenLoss("delta_nmse").delta(
        student + 0.2 * torch.randn_like(student), prev, teacher, teacher_prev,
    )
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert prev.grad is None


def test_delta_vocab_cos_matches_explicit_centred_vocab_scores():
    torch.manual_seed(22)
    norm = nn.RMSNorm(H)
    head = nn.Linear(H, V, bias=False)
    loss = HiddenLoss("delta_vocab_cos", final_norm=norm, lm_head=head)
    s_prev = torch.randn(7, H, requires_grad=True)
    t_prev = torch.randn(7, H)
    s = (s_prev.detach() + torch.randn(7, H)).requires_grad_(True)
    t = t_prev + torch.randn(7, H)

    got = loss.delta(s, s_prev, t, t_prev)
    W = head.weight.detach()
    ds = (s - s_prev.detach()) @ W.T
    dt = (t - t_prev) @ W.T
    ds = ds - ds.mean(dim=-1, keepdim=True)
    dt = dt - dt.mean(dim=-1, keepdim=True)
    want = 1.0 - F.cosine_similarity(ds, dt, dim=-1).mean()
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-6)

    got.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert s_prev.grad is None  # the delta's preceding state is stop-gradient
    assert all(p.grad is None for p in head.parameters())


def test_delta_vocab_cos_uses_vocab_mse_at_cache_boundaries():
    """The cache's absent h0 and post-norm h_n retain a meaningful state loss."""
    torch.manual_seed(23)
    norm = nn.RMSNorm(H)
    head = nn.Linear(H, V, bias=False)
    delta = HiddenLoss("delta_vocab_cos", final_norm=norm, lm_head=head)
    state = HiddenLoss("vocab_mse", final_norm=norm, lm_head=head)
    hs, ht = torch.randn(6, H), torch.randn(6, H)
    assert torch.allclose(delta(hs, ht, normed=True),
                          state(hs, ht, normed=True), rtol=1e-6)
