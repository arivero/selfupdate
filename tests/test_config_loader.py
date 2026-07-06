from pathlib import Path
from types import SimpleNamespace

import pytest

from selfupdate.config import load_config
from selfupdate.train import layerwise


def test_load_config_reports_yaml_path_for_extra_document_end(tmp_path: Path):
    base = tmp_path / "base.yaml"
    bad = tmp_path / "bad.yaml"
    base.write_text("run_name: dev\n", encoding="utf-8")
    bad.write_text(
        "run_name: bad\n"
        "model:\n"
        "  name: Qwen/Qwen3-0.6B\n"
        "...\n"
        "data:\n"
        "  examples_path: data/poem/examples.jsonl\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_config(base, bad)

    message = str(excinfo.value)
    assert str(bad) in message
    assert "failed to parse YAML config" in message


def test_pipeline_splits_map_blocks_across_visible_devices(monkeypatch):
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_name):
            return SimpleNamespace(
                model_type="qwen3",
                num_hidden_layers=8,
                tie_word_embeddings=False,
            )

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="fake/model",
            pipeline_split=0,
            pipeline_splits=[2, 5, 7],
        )
    )
    monkeypatch.setattr(layerwise, "AutoConfig", FakeAutoConfig, raising=False)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", FakeAutoConfig.from_pretrained)
    monkeypatch.setattr(layerwise.torch.cuda, "device_count", lambda: 4)

    dm = layerwise._pp_device_map(cfg)

    assert dm["model.embed_tokens"] == 0
    assert dm["model.norm"] == 3
    assert dm["lm_head"] == 3
    assert [dm[f"model.layers.{i}"] for i in range(8)] == [0, 0, 1, 1, 1, 2, 2, 3]
