"""Anti-intrusion anchor: gradient confinement + data hygiene."""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.eval.general import PROBE_TEXTS
from selfupdate.eval.recite import normalize_verse
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import AnchorBank, anchor_step

ANCHORS = Path("data/anchors_es.txt")


def test_anchor_file_disjoint_from_probes_and_poem():
    """Anchor fragments must not overlap the forgetting probes (they would
    train the eval) or the poem (they would be more poem data)."""
    frags = [normalize_verse(t) for t in ANCHORS.read_text(encoding="utf-8").split("\n\n")
             if t.strip()]
    assert len(frags) >= 4
    probes = " ".join(normalize_verse(p).replace("\n", " ") for p in PROBE_TEXTS)
    poem = normalize_verse(Path("data/poem/raw.txt").read_text(encoding="utf-8"))
    poem_lines = {l for l in poem.split("\n") if len(l) > 12}
    for f in frags:
        for line in f.split("\n"):
            if len(line) > 12:
                assert line not in probes, f"anchor line overlaps a probe: {line!r}"
                assert line not in poem_lines, f"anchor line overlaps the poem: {line!r}"


def test_anchor_bank_cycles():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    bank = AnchorBank(ANCHORS, tok, "cpu")
    n = len(bank.ids)
    first, base = bank.next()
    assert base is None
    for _ in range(n - 1):
        bank.next()
    again, _ = bank.next()
    assert torch.equal(first, again)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_anchor_grads_confined_to_window():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32).to("cuda").train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    bank = AnchorBank(ANCHORS, tok, "cuda")
    n = stack.n_layers
    L0 = n - 4 + 1
    model.zero_grad(set_to_none=True)
    a_ids, _ = bank.next()
    pos = torch.arange(len(a_ids), device="cuda")[None]
    with torch.no_grad():
        h = stack.embed(a_ids[None])
        pe = stack.rope(h, pos)
        for L in range(1, n + 1):
            h = stack.run_block(L, h, pe)
        base_logits = stack.lm_head(stack.final_norm(h))[0].detach()
    kl = float(anchor_step(
        stack, L0, a_ids, w=0.5, base_logits=base_logits, autocast=False,
    ).detach().cpu())
    assert kl >= 0
    for L in range(1, L0):
        assert all(p.grad is None for p in stack.block_params(L)), f"L{L} leaked"
    for L in range(L0, n + 1):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in stack.block_params(L)), f"window L{L} no grads"
    for pname, p in stack.model.named_parameters():
        if any(k in pname for k in ("embed_tokens", "model.norm", "lm_head")):
            assert p.grad is None, f"{pname} got a gradient"
