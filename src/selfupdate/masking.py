"""The generic context-masking abstraction.

Every example is four text segments:

    shared_prefix | privileged | shared_mid | answer

The teacher sees all four; the student skips ``privileged``. RAG mode puts the
privileged block (a retrieved passage) inside the user turn; thinking mode puts
it inside the assistant ``<think>`` block. Both modes produce the same
:class:`AlignedPair` contract, so every trainer and eval downstream is
mode-agnostic.

Token identity between teacher and student on shared segments is guaranteed by
construction: each segment is tokenized separately (``add_special_tokens=False``)
and the ID lists are concatenated. Never tokenize concatenated strings — BPE
boundary merges would silently break the alignment.

The aligned span is ``shared_mid + answer``: the position that predicts the
first answer token is the last ``shared_mid`` token, and that position carries
the core "recall without context" signal, so it must be inside the matched
region. Classical KD logit losses apply at ``[s0, s0+A-1)`` predicting tokens
``[s0+1, s0+A)``. This branch does not train on hidden-state losses.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass


@dataclass
class SegmentedExample:
    """Rendered chat-text segments for one training example.

    ``student_stub`` is what the student sees in place of ``privileged``:
    "" (default) removes the block entirely ("compact to zero size"); a short
    uninformative placeholder keeps a positional marker where the hidden
    context lived — a compared research axis (see MaskConfig.compaction).
    """

    example_id: str
    shared_prefix: str
    privileged: str  # teacher-only block; "" makes teacher == student
    shared_mid: str
    answer: str
    student_stub: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SegmentedExample":
        return cls(**d)

    @classmethod
    def from_record(cls, d: dict) -> "SegmentedExample":
        """Build from an examples.jsonl record, ignoring extra keys
        (answer_text, question, ...). The single place that knows which
        record keys are segment fields."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class AlignedPair:
    example_id: str
    teacher_ids: list[int]
    student_ids: list[int]
    t_aligned: slice  # shared_mid + answer span in the teacher sequence
    s_aligned: slice  # same span in the student sequence
    t_answer: slice  # answer-only span (eval / CE)
    s_answer: slice
    position_gap: int = 0  # extra positions the teacher's aligned span sits at

    @property
    def aligned_len(self) -> int:
        return self.s_aligned.stop - self.s_aligned.start

    def student_position_ids(self, rebase_gap: bool = False) -> list[int]:
        """Position ids for the student forward.

        With ``rebase_gap=True`` the aligned span is shifted by
        ``position_gap`` so every relative distance matches the teacher's RoPE
        geometry exactly (valid because a constant offset is output-invariant,
        see tests/test_position_invariance.py); the divergence then reduces to
        the missing attention targets alone.
        """
        n = len(self.student_ids)
        if not rebase_gap or self.position_gap == 0:
            return list(range(n))
        s0 = self.s_aligned.start
        return list(range(s0)) + [p + self.position_gap for p in range(s0, n)]


# Qwen3 chat-template pieces, rendered manually so we can split inside turns.
# tests/test_alignment.py checks these against tokenizer.apply_chat_template.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"

DEFAULT_SYSTEM = "Eres un experto en poesía española. Respondes recitando de memoria, con exactitud literal."


# uninformative placeholder shown to the student under "stub" compaction
RAG_STUB = "\n\nDocumento recuperado:\n[no disponible]"
THINK_STUB = "..."


def render_rag(
    example_id: str,
    question: str,
    passage: str,
    answer: str,
    system: str = DEFAULT_SYSTEM,
    student_stub: str = "",
) -> SegmentedExample:
    """RAG mode: privileged = retrieved passage appended to the user turn."""
    prefix = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{question}"
    )
    privileged = f"\n\nDocumento recuperado:\n{passage}" if passage else ""
    mid = f"{IM_END}\n{IM_START}assistant\n{EMPTY_THINK}"
    return SegmentedExample(
        example_id, prefix, privileged, mid, f"{answer}{IM_END}", student_stub
    )


def render_rag_tool(
    example_id: str,
    question: str,
    passage: str,
    answer: str,
    system: str = DEFAULT_SYSTEM,
    student_stub: str = "",
) -> SegmentedExample:
    """RAG via Qwen3's NATIVE tool protocol: the retrieved passage arrives as
    a Hermes-style <tool_response> block in its own tool turn (how the model
    saw retrieval during training). The entire tool turn is the privileged
    segment, so the student's censored view is a canonical no-tool
    conversation — same alignment contract as render_rag."""
    prefix = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{question}{IM_END}\n"
    )
    privileged = (
        f"{IM_START}user\n<tool_response>\n{passage}\n</tool_response>{IM_END}\n"
        if passage else ""
    )
    mid = f"{IM_START}assistant\n{EMPTY_THINK}"
    return SegmentedExample(
        example_id, prefix, privileged, mid, f"{answer}{IM_END}", student_stub
    )


def render_thinking(
    example_id: str,
    question: str,
    trace: str,
    answer: str,
    system: str = DEFAULT_SYSTEM,
    student_stub: str = "",
) -> SegmentedExample:
    """Thinking mode: privileged = the model's own <think> trace body.

    Student text (prefix + mid) reproduces Qwen3's enable_thinking=False
    rendering exactly: ``<think>\\n\\n</think>\\n\\n``.
    """
    prefix = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{question}{IM_END}\n"
        f"{IM_START}assistant\n<think>\n"
    )
    privileged = trace.strip()
    mid = "\n</think>\n\n"
    return SegmentedExample(
        example_id, prefix, privileged, mid, f"{answer}{IM_END}", student_stub
    )


def render_rag_thinking(
    example_id: str,
    question: str,
    passage: str,
    trace: str,
    answer: str,
    system: str = DEFAULT_SYSTEM,
    student_stub: str = "",
) -> SegmentedExample:
    """Mixed RAG + visible thinking.

    The teacher sees the retrieved passage in the user turn, then writes a
    visible ``<think>`` trace and the answer. The student sees the same prompt
    with only the RAG passage removed; the trace remains part of the aligned
    target and must be reproduced.
    """
    prefix = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{question}"
    )
    privileged = f"\n\nDocumento recuperado:\n{passage}" if passage else ""
    mid = f"{IM_END}\n{IM_START}assistant\n<think>\n"
    visible = f"{trace.strip()}\n</think>\n\n" if trace.strip() else "\n</think>\n\n"
    return SegmentedExample(
        example_id, prefix, privileged, mid, f"{visible}{answer}{IM_END}", student_stub
    )


class ContextMasker:
    """Tokenizes SegmentedExamples into aligned teacher/student ID pairs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def build(self, ex: SegmentedExample) -> AlignedPair:
        prefix = self._encode(ex.shared_prefix)
        priv = self._encode(ex.privileged)
        stub = self._encode(ex.student_stub)
        mid = self._encode(ex.shared_mid)
        answer = self._encode(ex.answer)

        teacher_ids = prefix + priv + mid + answer
        student_ids = prefix + stub + mid + answer

        t0 = len(prefix) + len(priv)
        s0 = len(prefix) + len(stub)
        pair = AlignedPair(
            example_id=ex.example_id,
            teacher_ids=teacher_ids,
            student_ids=student_ids,
            t_aligned=slice(t0, len(teacher_ids)),
            s_aligned=slice(s0, len(student_ids)),
            t_answer=slice(t0 + len(mid), len(teacher_ids)),
            s_answer=slice(s0 + len(mid), len(student_ids)),
            position_gap=len(priv) - len(stub),
        )
        assert (
            teacher_ids[pair.t_aligned] == student_ids[pair.s_aligned]
        ), f"aligned-span token mismatch in {ex.example_id}"
        assert pair.aligned_len == len(mid) + len(answer)
        return pair
