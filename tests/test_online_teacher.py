"""Online-teacher equivalence: with LoRA attached (B=0, forward unchanged)
and adapters disabled, per-step teacher targets must match the disk cache.

Validates both paths at once: the cache is what the fp32 build wrote; the
online teacher recomputes under bf16 autocast, so tolerances are bf16-scale.
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import LoraConfig
from selfupdate.data.dataset import DistillDataset
from selfupdate.teacher.cache import TeacherCache
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
    cache = TeacherCache(_cache_dir())

    base = peft_model.get_base_model()
    ds = DistillDataset(EXAMPLES, cache, tok, need_logits=True, with_teacher_ids=True)
    it = ds[5]

    # Online top-k should assign the same top-1 token almost everywhere and
    # close values at cached top-k indices.
    with torch.no_grad(), peft_model.disable_adapter(), \
            torch.autocast("cuda", dtype=torch.bfloat16):
        t_h = base.model(input_ids=it.teacher_ids.to("cuda")[None],
                         use_cache=False).last_hidden_state[0]
        t_logits = base.lm_head(t_h[it.t0: it.t0 + it.A - 1]).float()
    cached_v, cached_i = it.topk_v[:-1].to("cuda"), it.topk_i[:-1].to("cuda")
    online_at_cached = torch.gather(t_logits, 1, cached_i.long())
    rel = (online_at_cached - cached_v.float()).abs().mean().item()
    assert rel < 0.25, f"mean |logit diff| at cached top-k: {rel}"
    top1_match = (t_logits.argmax(-1) == cached_i[:, 0].long()).float().mean().item()
    assert top1_match > 0.95, f"top-1 agreement only {top1_match:.2%}"


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
