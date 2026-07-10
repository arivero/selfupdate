"""Teacher-cache dtype is a storage contract, not merely a hash label."""

import json

import pytest
import torch

from selfupdate.config import ExperimentConfig
from selfupdate.teacher.cache import (TeacherCache, TeacherCacheWriter,
                                      resolve_cache_dir)


def test_cache_writer_honors_bfloat16_and_records_it(tmp_path):
    root = tmp_path / "cache"
    writer = TeacherCacheWriter(root, "hash", hidden_dtype="bfloat16")
    writer.add("item", {1: torch.tensor([[1.25, -2.5]])}, {"shard": 0})
    writer.finalize()

    assert json.loads((root / "index.json").read_text())["hidden_dtype"] == "bfloat16"
    assert TeacherCache(root).hidden("item", 1).dtype == torch.bfloat16


def test_cache_writer_refuses_nonfinite_storage(tmp_path):
    writer = TeacherCacheWriter(tmp_path / "cache", "hash", hidden_dtype="float16")
    with pytest.raises(FloatingPointError, match="non-finite"):
        writer.add("item", {1: torch.tensor([[float("inf")]])}, {"shard": 0})


def test_bfloat16_cache_identity_is_new_and_dtype_is_valid(tmp_path):
    cfg = ExperimentConfig()
    cfg.cache.root = str(tmp_path)
    fp16_root, _ = resolve_cache_dir(cfg)
    cfg.cache.hidden_dtype = "bfloat16"
    bf16_root, _ = resolve_cache_dir(cfg)
    assert fp16_root != bf16_root

    cfg.cache.hidden_dtype = "float32"
    with pytest.raises(ValueError, match="cache.hidden_dtype"):
        resolve_cache_dir(cfg)
