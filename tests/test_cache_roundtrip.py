"""Cache round-trip: recompute teacher logits and compare to stored tensors.

Requires a built cache (scripts/build_teacher_cache.py); skipped otherwise.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.masking import ContextMasker, SegmentedExample
from selfupdate.teacher.cache import TeacherCache

MODEL = "Qwen/Qwen3-0.6B"
EXAMPLES = "data/poem/examples.jsonl"


def _cache_dir():
    """The cache whose config-hash matches base.yaml exactly — a glob would
    also match caches of other datasets (v2) for the same model+mask."""
    from selfupdate.config import load_config
    from selfupdate.teacher.cache import resolve_cache_dir

    root, _ = resolve_cache_dir(load_config("configs/base.yaml", None))
    return root if root.exists() else None


pytestmark = pytest.mark.skipif(_cache_dir() is None, reason="no built cache")


@pytest.fixture(scope="module")
def setup():
    cache = TeacherCache(_cache_dir())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda").eval()
    examples = {}
    for line in Path(EXAMPLES).read_text(encoding="utf-8").splitlines():
        ex = SegmentedExample.from_record(json.loads(line))
        examples[ex.example_id] = ex
    return cache, tok, model, examples


def test_roundtrip_five_examples(setup):
    cache, tok, model, examples = setup
    masker = ContextMasker(tok)
    ids = cache.example_ids
    picks = [ids[i] for i in (0, len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4, -1)]
    for ex_id in picks:
        pair = masker.build(examples[ex_id])
        span = cache.span(ex_id)
        assert span["t0"] == pair.t_aligned.start
        assert span["A"] == pair.aligned_len
        with torch.no_grad():
            out = model(torch.tensor([pair.teacher_ids], device="cuda"),
                        use_cache=False)
        sl = pair.t_aligned
        topk_v, topk_i, logz = cache.logits(ex_id)
        logits = out.logits[0, sl.start:sl.stop].float().cpu()
        fresh_z = torch.logsumexp(logits, -1)
        assert (fresh_z - logz).abs().max().item() < 1e-3
        fresh_v = torch.gather(logits, 1, topk_i.long())
        assert (fresh_v - topk_v.float()).abs().max().item() < 0.05
