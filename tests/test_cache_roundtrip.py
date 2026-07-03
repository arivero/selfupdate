"""Cache round-trip: recompute teacher forwards and compare to stored tensors.

Requires a built cache (scripts/build_teacher_cache.py); skipped otherwise.
Also pins the layer-index convention: h{L} == output_hidden_states[L], with
h{n_layers} being post-final-RMSNorm.
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
    n_layers = model.config.num_hidden_layers
    ids = cache.example_ids
    picks = [ids[i] for i in (0, len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4, -1)]
    for ex_id in picks:
        pair = masker.build(examples[ex_id])
        span = cache.span(ex_id)
        assert span["t0"] == pair.t_aligned.start
        assert span["A"] == pair.aligned_len
        with torch.no_grad():
            out = model(torch.tensor([pair.teacher_ids], device="cuda"),
                        output_hidden_states=True, use_cache=False)
        sl = pair.t_aligned
        for L in (1, n_layers // 2, n_layers):
            fresh = out.hidden_states[L][0, sl.start:sl.stop].cpu()
            stored = cache.hidden(ex_id, L).float()
            err = (fresh - stored).abs().max().item()
            scale = fresh.abs().max().item()
            assert err <= max(2e-3 * scale, 1e-2), f"{ex_id} h{L}: err {err} scale {scale}"
        topk_v, topk_i, logz = cache.logits(ex_id)
        logits = out.logits[0, sl.start:sl.stop].float().cpu()
        fresh_z = torch.logsumexp(logits, -1)
        assert (fresh_z - logz).abs().max().item() < 1e-3
        fresh_v = torch.gather(logits, 1, topk_i.long())
        assert (fresh_v - topk_v.float()).abs().max().item() < 0.05


def test_last_layer_is_post_norm(setup):
    """h{n_layers} must equal final RMSNorm applied to the last block output —
    the convention student-side losses rely on."""
    cache, tok, model, examples = setup
    n_layers = model.config.num_hidden_layers
    ex_id = cache.example_ids[0]
    pair = ContextMasker(tok).build(examples[ex_id])

    raw = {}

    def hook(_m, _i, out):
        raw["h"] = out[0] if isinstance(out, tuple) else out

    h = model.model.layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(torch.tensor([pair.teacher_ids], device="cuda"), use_cache=False)
    h.remove()

    sl = pair.t_aligned
    normed = model.model.norm(raw["h"])[0, sl.start:sl.stop].cpu()
    stored = cache.hidden(ex_id, n_layers).float()
    err = (normed - stored).abs().max().item()
    scale = normed.abs().max().item()
    assert err <= max(2e-3 * scale, 1e-2), f"err {err} scale {scale}"
