"""Template-agnostic chat rendering: derive segment pieces from ANY tokenizer.

masking.py renders Qwen3's chat template by hand because examples must be
split INSIDE turns (teacher/student differ mid-turn) — something
``apply_chat_template`` cannot express. That manual rendering is the single
Qwen-ism blocking other model families (Llama/SmolLM/Gemma tickets in
issues.md).

This module removes it generically: render the template twice with a sentinel
in place of the variable content and split the *string* output around it.
Whatever BOS, role markers or turn closers a family uses land in the derived
``pre``/``mid``/``answer_close`` pieces automatically, and segment-wise
``add_special_tokens=False`` encoding stays faithful because transformers v5
templates carry all specials in the rendered string.

Entry points:
- ``adapt_records(records, tokenizer)``: load-time choke point. Records built
  by scripts/build_dataset.py store canonical Qwen text plus the raw
  ``question``/``answer_text`` fields; when the run's tokenizer uses a
  different template, segments are re-rendered from the raw fields (RAG-mode
  ``privileged`` is pure text, so this is lossless). For Qwen tokenizers this
  is an exact identity — asserted in tests/test_chatfmt.py.
- ``render_rag_for(tokenizer, ...)``: generic counterpart of
  ``masking.render_rag`` for code that builds fresh prompts (recite_long).
- ``stop_token_id(tokenizer)``: the turn-terminator token to stop generation
  on (``<|im_end|>`` for Qwen, ``<|eot_id|>`` for Llama-3, ...), falling back
  to ``eos_token_id`` when the closer is not a single token.

Thinking mode and the rag_tool arm splice into family-specific syntax
(``<think>`` blocks, Hermes tool turns); they stay Qwen-only and are rejected
here with explicit errors rather than silently mis-rendered.
"""

from __future__ import annotations

from dataclasses import dataclass

from .masking import DEFAULT_SYSTEM, IM_END, IM_START, SegmentedExample

_SENTINEL = "\x00CUT\x00"  # passes through jinja string rendering verbatim

# privileged-block wrapper for RAG mode — mode-owned text, template-free
# (must match masking.render_rag exactly; test_chatfmt asserts equivalence)
RAG_PRIV_WRAP = "\n\nDocumento recuperado:\n{passage}"


@dataclass(frozen=True)
class TemplatePieces:
    """Chat-template fragments around the variable content.

    student prompt = ``pre + question + mid`` (generation-ready);
    teacher prompt inserts ``privileged`` between question and ``mid``;
    a stored answer segment is ``answer_text + answer_close``.
    """

    pre: str  # up to and including the opening of the user content
    mid: str  # user-turn close + assistant generation prompt
    answer_close: str  # assistant turn terminator (e.g. "<|im_end|>")


_pieces_cache: dict[tuple[str, str], TemplatePieces] = {}


def _render(tokenizer, msgs, **kw) -> str:
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, enable_thinking=False, **kw
    )


def template_pieces(tokenizer, system: str = DEFAULT_SYSTEM) -> TemplatePieces:
    key = (str(getattr(tokenizer, "name_or_path", id(tokenizer))), system)
    if key in _pieces_cache:
        return _pieces_cache[key]

    gen = _render(
        tokenizer,
        [{"role": "system", "content": system},
         {"role": "user", "content": _SENTINEL}],
        add_generation_prompt=True,
    )
    assert gen.count(_SENTINEL) == 1, (
        "chat template duplicated/transformed the user content; cannot derive "
        "segment pieces for this tokenizer"
    )
    pre, mid = gen.split(_SENTINEL)

    closed = _render(
        tokenizer,
        [{"role": "system", "content": system},
         {"role": "user", "content": "q"},
         {"role": "assistant", "content": _SENTINEL}],
        add_generation_prompt=False,
    )
    assert closed.count(_SENTINEL) == 1, (
        "chat template duplicated/transformed the assistant content"
    )
    # everything after the answer, minus cosmetic trailing newlines
    answer_close = closed.split(_SENTINEL)[1].rstrip("\n")

    pieces = TemplatePieces(pre=pre, mid=mid, answer_close=answer_close)
    _pieces_cache[key] = pieces
    return pieces


def stop_token_id(tokenizer) -> int:
    """Turn-terminator id for generation stops. Single-token closers (all
    known families) are used directly; otherwise fall back to eos."""
    close = template_pieces(tokenizer).answer_close
    ids = tokenizer.encode(close, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else tokenizer.eos_token_id


def render_rag_for(
    tokenizer,
    example_id: str,
    question: str,
    passage: str,
    answer: str,
    system: str = DEFAULT_SYSTEM,
    student_stub: str = "",
) -> SegmentedExample:
    """Template-agnostic masking.render_rag (identical output on Qwen3)."""
    p = template_pieces(tokenizer, system)
    return SegmentedExample(
        example_id,
        p.pre + question,
        RAG_PRIV_WRAP.format(passage=passage) if passage else "",
        p.mid,
        f"{answer}{p.answer_close}",
        student_stub,
    )


def _matches(record: dict, p: TemplatePieces) -> bool:
    return (
        record["shared_prefix"] == p.pre + record["question"]
        and record["shared_mid"] == p.mid
        and record["answer"] == record["answer_text"] + p.answer_close
    )


def _qwen_native_template(p: TemplatePieces) -> bool:
    return p.pre.startswith(f"{IM_START}system\n") and p.answer_close == IM_END


def _qwen_native_record(record: dict) -> bool:
    return (
        record.get("shared_prefix", "").startswith(f"{IM_START}system\n")
        and record.get("answer", "").endswith(IM_END)
    )


def adapt_records(
    records: list[dict], tokenizer, system: str = DEFAULT_SYSTEM
) -> list[dict]:
    """Re-render stored segments for this tokenizer's template if needed.

    Identity (same list object) when the stored rendering already matches —
    the Qwen fast path, so existing runs are byte-for-byte unaffected.
    """
    if not records:
        return records
    p = template_pieces(tokenizer, system)
    if _matches(records[0], p):
        return records
    if _qwen_native_template(p) and _qwen_native_record(records[0]):
        return records

    adapted = []
    for r in records:
        # thinking-mode records split INSIDE the think block: the prefix ends
        # at the opened "<think>" tag. (RAG records contain a closed, empty
        # think block in shared_mid — that re-renders fine.)
        if r.get("shared_prefix", "").rstrip().endswith("<think>"):
            raise ValueError(
                f"{r.get('example_id')}: thinking-mode records splice into the "
                "<think> block — Qwen/R1-family only; re-harvest traces for "
                "this model family (scripts/build_dataset.py, mask.mode=thinking)"
            )
        if "<|im_start|>" in r.get("privileged", "") or "<tool_response>" in r.get("privileged", ""):
            raise ValueError(
                f"{r.get('example_id')}: rag_tool records embed Qwen's native "
                "tool protocol; rebuild the dataset for this family "
                "(scripts/build_dataset.py, mask.mode=rag_tool)"
            )
        if "question" not in r or "answer_text" not in r:
            raise ValueError(
                f"{r.get('example_id')}: record lacks raw question/answer_text "
                "fields; rebuild examples.jsonl with current build_dataset.py"
            )
        adapted.append({
            **r,
            "shared_prefix": p.pre + r["question"],
            "shared_mid": p.mid,
            "answer": r["answer_text"] + p.answer_close,
        })
    return adapted
