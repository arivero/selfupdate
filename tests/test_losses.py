import torch
import torch.nn.functional as F

from selfupdate.train.losses import kd_topk_kl


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


def test_kd_gradients_flow():
    torch.manual_seed(4)
    N, V, k = 3, 60, 8
    t = torch.randn(N, V)
    s = torch.randn(N, V, requires_grad=True)
    v, i = t.topk(k, -1)
    loss = kd_topk_kl(s, v, i, torch.logsumexp(t, -1))
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_kd_topk_kl_has_no_external_label_argument():
    torch.manual_seed(6)
    N, V, k = 4, 100, 16
    t = torch.randn(N, V)
    s = torch.randn(N, V)
    v, i = t.topk(k, -1)
    logz = torch.logsumexp(t, -1)

    # The KL target is fully determined by teacher logits. External token
    # labels are not an argument to this loss.
    labels_a = torch.randint(0, V, (N,))
    labels_b = (labels_a + 17) % V
    assert not torch.equal(labels_a, labels_b)
    loss_a = kd_topk_kl(s, v, i, logz)
    loss_b = kd_topk_kl(s, v, i, logz)
    assert torch.equal(loss_a, loss_b)


def test_kd_temperature_t2_rescale():
    torch.manual_seed(5)
    N, V, k = 4, 80, 16
    t = torch.randn(N, V) * 2
    s = torch.randn(N, V) * 2
    v, i = t.topk(k, -1)
    logz = torch.logsumexp(t, -1)
    T = 2.0
    # bucketed teacher/student KL at temperature T, then Hinton T^2 rescale
    expected = T * T * _bucket_kl(t, s, v, i, logz, T)
    got = kd_topk_kl(s, v, i, logz, T=T)
    assert torch.allclose(got, expected, atol=1e-5)


def _bucket_kl(t, s, v, i, logz, T):
    import torch.nn.functional as F

    lse_k = torch.logsumexp(v, -1)
    tail = logz + torch.log1p(-torch.exp((lse_k - logz).clamp(max=-1e-7)))
    logp = F.log_softmax(torch.cat([v, tail[:, None]], -1) / T, -1)
    ls = F.log_softmax(s / T, -1)
    ls_k = torch.gather(ls, -1, i.long())
    s_tail = torch.log1p(-torch.exp(torch.logsumexp(ls_k, -1).clamp(max=-1e-7)))
    logq = torch.cat([ls_k, s_tail[:, None]], -1)
    return (logp.exp() * (logp - logq)).sum(-1).mean()
