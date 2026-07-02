import torch
import torch.nn.functional as F

from selfupdate.train.losses import hidden_match, kd_topk_kl


def full_kl(t_logits, s_logits, T=1.0):
    p = F.log_softmax(t_logits / T, -1)
    q = F.log_softmax(s_logits / T, -1)
    return (p.exp() * (p - q)).sum(-1).mean()


def test_kd_topk_kl_exact_when_k_is_vocab():
    torch.manual_seed(0)
    N, V = 7, 50
    t = torch.randn(N, V) * 3
    s = torch.randn(N, V) * 3
    v, i = t.topk(V - 1, dim=-1)  # leave one token in the tail bucket
    logz = torch.logsumexp(t, -1)
    approx = kd_topk_kl(s, v, i, logz)
    # with k = V-1 the "tail" is a single real token, so the bucketed KL is
    # exactly the full KL
    assert torch.allclose(approx, full_kl(t, s), atol=1e-5)


def test_kd_topk_kl_zero_for_identical():
    torch.manual_seed(1)
    N, V, k = 5, 100, 16
    t = torch.randn(N, V) * 2
    v, i = t.topk(k, -1)
    logz = torch.logsumexp(t, -1)
    assert kd_topk_kl(t.clone(), v, i, logz).abs().item() < 1e-6


def test_kd_topk_kl_close_to_full_kl_for_peaked_teacher():
    torch.manual_seed(2)
    N, V, k = 5, 1000, 64
    t = torch.randn(N, V)
    t[:, :3] += 12  # peaked teacher: top-k captures nearly all mass
    s = torch.randn(N, V)
    v, i = t.topk(k, -1)
    logz = torch.logsumexp(t, -1)
    assert abs(kd_topk_kl(s, v, i, logz).item() - full_kl(t, s).item()) < 0.05


def test_hidden_match_scale_invariance_of_nmse():
    torch.manual_seed(3)
    hs, ht = torch.randn(10, 32), torch.randn(10, 32)
    a = hidden_match(hs, ht, "nmse")
    b = hidden_match(hs * 10, ht * 10, "nmse")
    assert torch.allclose(a, b, rtol=1e-4)
    assert hidden_match(ht, ht, "nmse").item() < 1e-10
    assert hidden_match(ht, ht, "l2mse").item() < 1e-10


def test_kd_gradients_flow():
    torch.manual_seed(4)
    N, V, k = 3, 60, 8
    t = torch.randn(N, V)
    s = torch.randn(N, V, requires_grad=True)
    v, i = t.topk(k, -1)
    loss = kd_topk_kl(s, v, i, torch.logsumexp(t, -1))
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
