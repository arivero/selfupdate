"""Online-teacher equivalence: with LoRA attached and adapters disabled,
per-step teacher hidden targets must match the disk cache.

The cache is what the fp32 build wrote; the online teacher recomputes under
bf16 autocast, so tolerances are bf16-scale.
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import LoraConfig
from selfupdate.data.dataset import DistillDataset
from selfupdate.teacher.cache import TeacherCache
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import _online_targets
from selfupdate.train.lora import attach_lora

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


def test_online_targets_match_cache():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda")
    peft_model = attach_lora(model, LoraConfig(enabled=True))
    base = peft_model.get_base_model()
    stack = BlockStack(base)
    cache = TeacherCache(_cache_dir())

    n = stack.n_layers
    ds = DistillDataset(EXAMPLES, cache, tok,
                        need_layers=[1, n // 2, n],
                        with_teacher_ids=True)
    it = ds[5]
    targets = _online_targets(stack, peft_model, it, "cuda")

    for L in (1, n // 2, n):
        cached = it.hidden[L].to("cuda", torch.float32)
        online = targets[L].float()
        err = (online - cached).abs().max().item()
        scale = cached.abs().max().item()
        # bf16 autocast online vs fp16-stored fp32 cache. Current kernels land
        # around 2.5% max-relative at some middle layers; keep the guard tight
        # but not tied to one cache build.
        assert err <= max(3e-2 * scale, 0.05), f"h{L}: err {err} scale {scale}"

def test_gap_decoder_matches_generate_at_gap_zero():
    """The manual position-aware greedy decoder must reproduce HF generate
    exactly when the gap is zero (contiguous positions)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from selfupdate.eval.recite import greedy_generate_positions

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to("cuda").eval()
    prompt = "La capital de Francia es"
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device="cuda")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=16, do_sample=False,
                             eos_token_id=eos, pad_token_id=tok.eos_token_id)
    got = greedy_generate_positions(
        model, ids, torch.arange(ids.shape[1], device="cuda")[None],
        max_new_tokens=16, eos_id=eos,
    )
    assert got == ref[0, ids.shape[1]:].tolist()
