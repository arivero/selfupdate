"""Regression tests for checkpoint and recall evaluation integrity."""

import importlib.util
from pathlib import Path

import pytest
import torch

from selfupdate.config import load_config
from selfupdate.eval import tasks


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()


class _Tokenizer:
    eos_token_id = 91

    def convert_tokens_to_ids(self, _token):
        return 0  # SentencePiece-style unknown id for <|im_end|>

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def encode(self, _text, **_kwargs):
        return [3, 4]

    def decode(self, _ids, **_kwargs):
        return ""


def test_tasks_eval_uses_chatfmt_stop_and_restores_training(tmp_path, monkeypatch):
    poem = tmp_path / "tiny.txt"
    poem.write_text("uno\ndos\n\ntres\ncuatro\n", encoding="utf-8")
    model = _Model().train()
    seen = []

    monkeypatch.setattr(tasks, "stop_token_id", lambda _tok: 73)
    from selfupdate.eval import recite

    def fake_generate(_model, _ids, _pos, *, max_new_tokens, eos_id):
        seen.append((max_new_tokens, eos_id))
        return []

    monkeypatch.setattr(recite, "greedy_generate_positions", fake_generate)
    tasks.tasks_eval(model, _Tokenizer(), str(poem), n_per_task=1,
                     keep_examples=0)

    assert model.eval_calls == 1
    assert model.training  # training caller remains ready for the next epoch
    assert seen and {eos for _, eos in seen} == {73}


def test_layer_residual_config_adopts_saved_data_and_mask():
    evaluate = _script("evaluate.py")
    cfg = load_config("configs/base.yaml")
    saved = {
        "model": {"name": "Qwen/Qwen3-1.7B"},
        "data": {
            "poem_path": "data/quijote/raw_ch8.txt",
            "examples_path": "data/quijote/examples_ch8.jsonl",
            "paraphrase": True,
            "long_windows": [24, 48],
        },
        "mask": {"mode": "rag", "compaction": "stub_gap"},
    }
    got = evaluate._adopt_checkpoint_eval_config(
        cfg, saved, require_geometry=True)
    assert got.model.name == "Qwen/Qwen3-1.7B"
    assert got.data.examples_path.endswith("examples_ch8.jsonl")
    assert got.data.paraphrase and got.data.long_windows == [24, 48]
    assert got.mask.compaction == "stub_gap"


def test_layer_residual_config_refuses_missing_geometry():
    evaluate = _script("evaluate.py")
    cfg = load_config("configs/base.yaml")
    with pytest.raises(ValueError, match="refuses to guess checkpoint geometry"):
        evaluate._adopt_checkpoint_eval_config(
            cfg, {"model": {"name": "Qwen/Qwen3-1.7B"}},
            require_geometry=True)
