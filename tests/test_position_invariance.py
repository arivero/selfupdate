"""RoPE position-offset invariance.

Qwen3 uses pure rotary position embeddings with full attention: shifting all
position_ids by a constant leaves every pairwise relative distance unchanged,
so the model's outputs are (mathematically) identical. This justifies NOT
rebasing student position_ids to match the teacher's absolute positions: the
teacher/student hidden-state difference at aligned positions is then purely
attention-into-the-privileged-block — the signal we distill.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    return model, tok


def test_constant_position_offset_is_invariant(model_and_tok):
    model, tok = model_and_tok
    ids = tok.encode("En la feria de Berlanga prendóse de una doncella,")
    input_ids = torch.tensor([ids])
    n = input_ids.shape[1]

    base_pos = torch.arange(n).unsqueeze(0)
    with torch.no_grad():
        out0 = model(input_ids, position_ids=base_pos, use_cache=False)
        out50 = model(input_ids, position_ids=base_pos + 50, use_cache=False)

    diff = (out0.logits - out50.logits).abs().max().item()
    assert diff < 1e-3, f"position offset changed logits by {diff}"
