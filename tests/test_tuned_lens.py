"""Tuned lens: identity-init equals raw lens, round-trip, KL decreases,
vocabulary stays frozen."""

import pytest
import torch

from selfupdate.train.tuned_lens import (apply_translator, lens_kl_step,
                                         load_translators, make_translators,
                                         save_translators)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@pytest.fixture(scope="module")
def model_and_tok():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                                 dtype=torch.bfloat16)
    model.to("cuda").eval()
    model.requires_grad_(False)
    return model, tok


def test_identity_init_is_exact_passthrough():
    tr = make_translators(16, 4, device="cpu")
    h = torch.randn(5, 16, dtype=torch.bfloat16)
    for L in (1, 2, 3):
        assert torch.equal(apply_translator(tr, L, h), h)
    # final layer has no translator; passthrough by construction
    assert torch.equal(apply_translator(tr, 4, h), h)


def test_round_trip(tmp_path):
    tr = make_translators(8, 3, device="cpu")
    with torch.no_grad():
        for lin in tr.values():
            lin.weight.uniform_(-0.1, 0.1)
            lin.bias.uniform_(-0.1, 0.1)
    save_translators(tr, tmp_path / "t.safetensors", meta={"model": "test"})
    back = load_translators(tmp_path / "t.safetensors", device="cpu")
    assert set(back.keys()) == set(tr.keys())
    for k in tr:
        assert torch.equal(back[k].weight, tr[k].weight)
        assert torch.equal(back[k].bias, tr[k].bias)
    assert (tmp_path / "meta.json").exists()


@cuda
def test_kl_decreases_and_vocab_frozen(model_and_tok):
    model, tok = model_and_tok
    ids = torch.tensor([tok.encode(
        "The library opened at nine and the reading room filled slowly "
        "with students, retirees, and one man who came only for the "
        "newspapers, which he read standing up beside the window.",
        add_special_tokens=False)], device="cuda")
    n = model.config.num_hidden_layers
    tr = make_translators(model.config.hidden_size, n, device="cuda")
    opt = torch.optim.AdamW(tr.parameters(), lr=1e-3)
    head_sum0 = model.lm_head.weight.double().sum().item()

    kl0 = lens_kl_step(model, tr, ids)
    opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(7):
        kl = lens_kl_step(model, tr, ids)
        opt.step(); opt.zero_grad(set_to_none=True)
    mean0 = sum(kl0.values()) / len(kl0)
    mean1 = sum(kl.values()) / len(kl)
    assert mean1 < mean0, (mean0, mean1)
    # no gradient ever reached the frozen model
    assert all(p.grad is None for p in model.parameters())
    assert model.lm_head.weight.double().sum().item() == head_sum0


@cuda
def test_zero_translators_reproduce_raw_lens_on_model(model_and_tok):
    model, tok = model_and_tok
    from selfupdate.data.dataset import load_jsonl
    from selfupdate.eval.logit_lens import gold_logprob_by_layer
    from selfupdate.masking import ContextMasker, SegmentedExample

    masker = ContextMasker(tok)
    pairs = [masker.build(SegmentedExample.from_record(r))
             for r in load_jsonl("data/poem/examples_v4.jsonl")[:2]]
    tr = make_translators(model.config.hidden_size,
                          model.config.num_hidden_layers, device="cuda")
    raw = gold_logprob_by_layer(model, tok, pairs, limit=2)
    tuned = gold_logprob_by_layer(model, tok, pairs, limit=2, translators=tr)
    assert raw == tuned
