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
from ..masking import SegmentedExample, render_thinking


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
