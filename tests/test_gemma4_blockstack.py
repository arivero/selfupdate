import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from selfupdate.train.blocks import BlockStack


def test_gemma4_blockstack_matches_text_forward_cpu():
    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=8,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
    )
    model = Gemma4ForCausalLM(cfg).eval()
    stack = BlockStack(model)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    pos = torch.arange(ids.shape[1])[None]

    with torch.no_grad():
        ref = model.model(input_ids=ids, position_ids=pos).last_hidden_state
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for layer in range(1, stack.n_layers + 1):
            h = stack.run_block(layer, h, pos_emb)
        got = stack.final_norm(h)

    assert stack.needs_gemma4_masks is True
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
