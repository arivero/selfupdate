"""Thinking-trace harvesting for the thinking-hiding mode.

The teacher generates its <think> trace greedily from the canonical thinking
prompt; the gold poem answer is then teacher-forced after ``</think>`` when the
cache is built, so the distillation target is the correct recitation
conditioned on the model's own reasoning. Traces are generated once and frozen
into examples.jsonl for reproducibility.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from ..data.poem import TaskSpec
from ..masking import (
    SegmentedExample,
    render_rag_mayeutic,
    render_rag_thinking,
    render_thinking,
)


def harvest_traces(
    model,
    tokenizer,
    specs: list[TaskSpec],
    *,
    max_think_tokens: int = 512,
    system: str | None = None,
    student_stub: str = "",
) -> list[SegmentedExample]:
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if think_end is None or think_end == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(
            "thinking mode needs a tokenizer with a </think> special token "
            "(Qwen3/R1 family); this model family has none — use rag mode"
        )
    examples = []
    for spec in tqdm(specs, desc="harvest <think> traces"):
        kwargs = {} if system is None else {"system": system}
        # render with an empty trace to get the teacher prompt ending at "<think>\n"
        proto = render_thinking(spec.task_id, spec.question, "", spec.answer, **kwargs)
        prompt_ids = tokenizer.encode(proto.shared_prefix, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_think_tokens,
                do_sample=False,
                eos_token_id=think_end,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, len(prompt_ids):].tolist()
        if gen and gen[-1] == think_end:
            gen = gen[:-1]
        trace = tokenizer.decode(gen).strip()
        examples.append(
            render_thinking(
                spec.task_id, spec.question, trace, spec.answer,
                **kwargs, student_stub=student_stub,
            )
        )
    return examples


def harvest_rag_thinking_traces(
    model,
    tokenizer,
    specs: list[TaskSpec],
    *,
    max_think_tokens: int = 512,
    system: str | None = None,
    student_stub: str = "",
) -> list[SegmentedExample]:
    """Harvest visible traces for mixed RAG+thinking mode.

    Unlike ``harvest_traces`` for thinking-hiding, only the RAG passage is
    privileged. The generated trace is stored in the aligned answer segment,
    so the student is trained to reproduce both the trace and the poem text.
    """
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if think_end is None or think_end == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(
            "rag_thinking mode needs a tokenizer with a </think> special token "
            "(Qwen3/R1 family)"
        )
    examples = []
    for spec in tqdm(specs, desc="harvest RAG-conditioned <think> traces"):
        kwargs = {} if system is None else {"system": system}
        proto = render_rag_thinking(
            spec.task_id, spec.question, spec.passage, "", spec.answer, **kwargs
        )
        prompt = proto.shared_prefix + proto.privileged + proto.shared_mid
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_think_tokens,
                do_sample=False,
                eos_token_id=think_end,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, len(prompt_ids):].tolist()
        if gen and gen[-1] == think_end:
            gen = gen[:-1]
        trace = tokenizer.decode(gen).strip()
        examples.append(
            render_rag_thinking(
                spec.task_id, spec.question, spec.passage, trace, spec.answer,
                **kwargs, student_stub=student_stub,
            )
        )
    return examples


def harvest_rag_mayeutic_traces(
    model,
    tokenizer,
    specs: list[TaskSpec],
    *,
    max_think_tokens: int = 512,
    system: str | None = None,
    student_stub: str = "",
) -> list[SegmentedExample]:
    """Harvest visible mayeutic traces for mixed RAG+thinking mode."""
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if think_end is None or think_end == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(
            "rag_mayeutic mode needs a tokenizer with a </think> special token "
            "(Qwen3/R1 family)"
        )
    examples = []
    for spec in tqdm(specs, desc="harvest RAG-conditioned mayeutic traces"):
        kwargs = {} if system is None else {"system": system}
        proto = render_rag_mayeutic(
            spec.task_id, spec.question, spec.passage, "", spec.answer, **kwargs
        )
        prompt = proto.shared_prefix + proto.privileged + proto.shared_mid
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_think_tokens,
                do_sample=False,
                eos_token_id=think_end,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, len(prompt_ids):].tolist()
        if gen and gen[-1] == think_end:
            gen = gen[:-1]
        trace = tokenizer.decode(gen).strip()
        examples.append(
            render_rag_mayeutic(
                spec.task_id, spec.question, spec.passage, trace, spec.answer,
                **kwargs, student_stub=student_stub,
            )
        )
    return examples
