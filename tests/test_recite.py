import torch

from selfupdate.eval import recite


class _DummyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        n = max(1, len(text.split()))
        return list(range(1, n + 1))

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "answer"


class _DummyModel:
    device = torch.device("cpu")

    def __init__(self):
        self.seen_input_ids = None
        self.seen_attention_mask = None

    def generate(self, input_ids, attention_mask=None, **kwargs):
        del kwargs
        self.seen_input_ids = input_ids.clone()
        self.seen_attention_mask = attention_mask.clone()
        return torch.cat([input_ids, torch.tensor([[42]], device=input_ids.device)], dim=1)


def test_recite_one_single_prompt_passes_attention_mask(monkeypatch):
    monkeypatch.setattr(recite, "stop_token_id", lambda tokenizer: tokenizer.eos_token_id)
    model = _DummyModel()
    record = {
        "example_id": "x",
        "shared_prefix": "question",
        "student_stub": "",
        "shared_mid": " answer:",
        "answer_text": "answer",
    }

    got = recite.recite_one(model, _DummyTokenizer(), record)

    assert got["example_id"] == "x"
    assert torch.equal(model.seen_attention_mask, torch.ones_like(model.seen_input_ids))
