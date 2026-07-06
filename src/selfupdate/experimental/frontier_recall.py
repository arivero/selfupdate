"""Epoch-0 recall of a frontier release at its NATIVE (possibly quantised) width.

The one question: before any of our training, how much of the poem does the
released model already reproduce from the same recitation prompt we train
against? This is the teacher-reference / epoch-0 number, but for models too
large (or too exotically quantised) to run through the layerwise trainer.

Deliberately minimal and quarantined (see package docstring):
- NO BlockStack, NO windows, NO adapters — just HF ``generate`` via the
  shared ``recite_eval``. Hybrid-attention layers (GatedDeltaNet) and
  mixed-bit MoE are the model's own business on the generate path.
- The model is loaded EXACTLY as released: its own ``quantization_config``
  from the checkpoint is honored (we never re-quantize or dequantize), so
  the recall number is the release-width number. ``device_map`` defaults to
  ``auto`` because these are multi-card loads.
- ``trust_remote_code`` is opt-in per model (DeepSeek V4 / GLM 5.2 ship
  custom modeling code); it is OFF unless the config asks for it.

Output mirrors scripts/evaluate.py --base so results slot into the same
teacher-reference tables, tagged ``teacher_epoch0_native_no_rag``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..data.dataset import load_jsonl
from ..eval.recite import recite_eval


def _load_any_lm(model_name, **kw):
    """Load a generate-able LM regardless of head class. Some frontier
    releases (Mistral-Medium-3.5 = mistral3) register ONLY as
    image-text-to-text and are absent from the causal-LM auto-map, so
    ``AutoModelForCausalLM`` raises. Fall back to ImageTextToText, whose
    ``.generate`` is the same text path recite_eval drives."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_name, **kw)


def load_frontier(model_name: str, *, trust_remote_code: bool = False,
                  device_map: str = "auto", max_memory: dict | None = None):
    """Load a released frontier model at native width. The checkpoint's own
    quantization_config is used verbatim — we pass NO quantization override,
    so a block-fp8 / mixed-bit release loads at release precision. dtype is
    left to the checkpoint (``dtype='auto'``) so an fp8/nvfp4 release is not
    silently upcast by a hardcoded bf16."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = _load_any_lm(
        model_name,
        dtype="auto",
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def epoch0_recall(model_name: str, examples_path: str, *,
                  trust_remote_code: bool = False,
                  device_map: str = "auto",
                  limit: int | None = None,
                  batch_size: int = 1,
                  max_extra_tokens: int = 48,
                  score_workers: int | None = None,
                  shuffle_seed: int | None = None,
                  out_dir: str | Path | None = None) -> dict:
    model, tok = load_frontier(model_name, trust_remote_code=trust_remote_code,
                               device_map=device_map)
    records = load_jsonl(examples_path)
    r = recite_eval(model, tok, records, limit=limit,
                    batch_size=batch_size, max_extra_tokens=max_extra_tokens,
                    score_workers=score_workers, shuffle_seed=shuffle_seed)
    r["model"] = model_name
    r["examples_path"] = examples_path
    r["teacher_reference_kind"] = "teacher_epoch0_native_no_rag"
    r["native_width"] = True
    qc = getattr(model.config, "quantization_config", None)
    if qc is not None:
        r["quantization"] = (qc if isinstance(qc, dict)
                             else getattr(qc, "quant_method", str(type(qc).__name__)))
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "recall.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    return r
