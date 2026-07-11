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
region. Hidden-state losses apply at aligned positions ``[s0, s0+A)``; logit
losses apply at ``[s0, s0+A-1)`` predicting tokens ``[s0+1, s0+A)``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
import string
from dataclasses import asdict, dataclass, field


@dataclass
class SegmentedExample:
    """Rendered chat-text segments for one training example.

    ``student_stub`` is what the student sees in place of ``privileged``:
    "" (default) removes the block entirely ("compact to zero size"); a short
    uninformative placeholder keeps a positional marker where the hidden
    context lived — a compared research axis (see MaskConfig.compaction).

    ``interleaved`` (thinking_selective mode): a list of ``[text,
    is_privileged]`` runs that REPLACES the single ``privileged`` block —
    the teacher sees every run, the student only the non-privileged ones
    (the model's own free deduction survives censoring; only the verbatim
    retrieved spans are hidden). When set, ``privileged`` and
    ``student_stub`` must be empty; compaction is remove-only.
    """

    example_id: str
    shared_prefix: str
    privileged: str  # teacher-only block; "" makes teacher == student
    shared_mid: str
    answer: str
    student_stub: str = ""
    interleaved: list | None = None

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
    # teacher-coordinate (start, stop) of each privileged run (interleaved
    # mode; a single-block example leaves this empty and the block is
    # implicitly [s_aligned.start, t_aligned.start))
    t_privileged: list = field(default_factory=list)

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


_FILL_VOCAB_CACHE: dict[int, list[int]] = {}


def random_fill_ids(tokenizer, key: str, n: int) -> list[int]:
    """Seeded length-``n`` sample of DISTINCT ordinary-vocabulary token ids
    (special/control ids banned) — the pad_random censor fill, shared by
    ``ContextMasker`` (training) and the eval battery's padded floor.
    Deterministic per ``key``."""
    vocab = _FILL_VOCAB_CACHE.get(id(tokenizer))
    if vocab is None:
        banned = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
        vocab = [i for i in range(tokenizer.vocab_size) if i not in banned]
        _FILL_VOCAB_CACHE[id(tokenizer)] = vocab
    if n > len(vocab):
        raise ValueError(
            f"{key}: fill of {n} tokens exceeds the ordinary vocabulary "
            f"({len(vocab)})")
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    return random.Random(seed).sample(vocab, n)


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
    open_answer: bool = False,
) -> SegmentedExample:
    """RAG via Qwen3's NATIVE tool protocol: the retrieved passage arrives as
    a Hermes-style <tool_response> block in its own tool turn (how the model
    saw retrieval during training). The entire tool turn is the privileged
    segment, so the student's censored view is a canonical no-tool
    conversation — same alignment contract as render_rag.

    ``open_answer=True`` (v5 question-only records) leaves the answer
    segment EMPTY — no text, no ``<|im_end|>``: the teacher stage appends
    its own generation plus the terminator."""
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
        example_id, prefix, privileged, mid,
        "" if open_answer else f"{answer}{IM_END}", student_stub
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


_SPAN_PUNCT = string.punctuation + "¡¿«»“”‘’—–"
# One-or-more whitespace/punctuation between verse words: a trace that quotes
# a verse with an inserted comma, or wraps it in quotes, must still count as
# a verbatim retrieval — censoring false negatives leak privileged content.
_WORD_SEP = r"[\s" + re.escape(_SPAN_PUNCT) + r"]+"


def find_poem_spans(trace: str, verses: list[str],
                    min_words: int = 3) -> list[tuple[int, int]]:
    """Char spans of whole-verse quotations inside a think trace.

    Whitespace-normalized, punctuation-tolerant, and case-insensitive: the
    model quotes verses with arbitrary wrapping/casing/inserted punctuation.
    Word-boundary anchored, so a verse cannot match starting or ending
    mid-word (e.g. "el que mira y sueña" inside "aquel que mira y sueña").
    Only whole verses count — a shared common word is deduction, a verbatim
    verse is retrieval. Verses under ``min_words`` words are skipped (too
    easy to emit by chance). Overlapping/adjacent matches are merged. Pure
    function."""
    spans = []
    for verse in verses:
        words = [w.strip(_SPAN_PUNCT) for w in verse.split()]
        words = [w for w in words if w]
        if len(words) < min_words:
            continue
        core = _WORD_SEP.join(re.escape(w) for w in words)
        pat = r"\b" + core + r"\b"
        for m in re.finditer(pat, trace, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def censor_spans(trace: str, spans: list[tuple[int, int]]) -> list[list]:
    """Split a trace into ``[text, is_privileged]`` runs from char spans."""
    runs: list[list] = []
    cur = 0
    for a, b in spans:
        if a > cur:
            runs.append([trace[cur:a], False])
        runs.append([trace[a:b], True])
        cur = b
    if cur < len(trace):
        runs.append([trace[cur:], False])
    return runs


def render_thinking_selective(
    example_id: str,
    question: str,
    trace: str,
    answer: str,
    verses: list[str],
    system: str = DEFAULT_SYSTEM,
) -> SegmentedExample:
    """Selective thinking mode: censor ONLY the whole-verse quotations
    inside the think trace; the model's free deduction stays visible to the
    student. This is the reasoning-family attack: whole-think censoring
    (render_thinking) deletes the channel the answer actually routes
    through; selective censoring deletes just the retrieved content."""
    prefix = (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{question}{IM_END}\n"
        f"{IM_START}assistant\n<think>\n"
    )
    trace = trace.strip()
    runs = censor_spans(trace, find_poem_spans(trace, verses))
    mid = "\n</think>\n\n"
    return SegmentedExample(
        example_id, prefix, "", mid, f"{answer}{IM_END}",
        interleaved=runs,
    )


class ContextMasker:
    """Tokenizes SegmentedExamples into aligned teacher/student ID pairs.

    ``pad_random`` (mask.compaction=pad_random, owner 2026-07-12): the
    student's view of the privileged block is a LENGTH-MATCHED random fill —
    every token distinct, drawn from ordinary vocabulary only (no special or
    control ids: a fixed pad token or any repeated filler is an attendable
    attractor the student could learn to key on). Length preservation makes
    the position gap zero, so student RoPE geometry matches the teacher's
    with no rebase. The fill is seeded per example_id: deterministic across
    epochs, runs, and cache/training stages.
    """

    def __init__(self, tokenizer, pad_random: bool = False):
        self.tokenizer = tokenizer
        self.pad_random = pad_random

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def _fill_ids(self, example_id: str, n: int) -> list[int]:
        return random_fill_ids(self.tokenizer, example_id, n)

    def build(self, ex: SegmentedExample,
              answer_ids: list[int] | None = None) -> AlignedPair:
        """``answer_ids`` (v5 open-answer records): the teacher's GENERATED
        answer, injected as token ids — never re-tokenized text, so the
        teacher-forced pass runs over exactly the ids the teacher produced.
        Requires an empty answer segment; every slice below follows from the
        injected length unchanged."""
        prefix = self._encode(ex.shared_prefix)
        mid = self._encode(ex.shared_mid)
        if answer_ids is not None:
            assert not ex.answer, (
                f"{ex.example_id}: answer_ids injection needs an open-answer "
                "record (empty answer segment)")
            answer = list(answer_ids)
        else:
            answer = self._encode(ex.answer)

        if ex.interleaved is not None:
            assert not ex.privileged and not ex.student_stub, (
                f"{ex.example_id}: interleaved replaces privileged/stub")
            t_runs: list[int] = []
            s_runs: list[int] = []
            t_priv: list[tuple[int, int]] = []
            cursor = len(prefix)
            for text, is_priv in ex.interleaved:
                ids = self._encode(text)
                t_runs += ids
                if is_priv:
                    t_priv.append((cursor, cursor + len(ids)))
                else:
                    s_runs += ids  # same id list: kept-run identity by construction
                cursor += len(ids)
            teacher_ids = prefix + t_runs + mid + answer
            student_ids = prefix + s_runs + mid + answer
            t0 = len(prefix) + len(t_runs)
            s0 = len(prefix) + len(s_runs)
        else:
            priv = self._encode(ex.privileged)
            stub = self._encode(ex.student_stub)
            if self.pad_random and priv:
                assert not stub, (
                    f"{ex.example_id}: pad_random requires an empty "
                    "student_stub (the fill replaces the block wholesale)")
                stub = self._fill_ids(ex.example_id, len(priv))
            teacher_ids = prefix + priv + mid + answer
            student_ids = prefix + stub + mid + answer
            t0 = len(prefix) + len(priv)
            s0 = len(prefix) + len(stub)
            t_priv = []

        pair = AlignedPair(
            example_id=ex.example_id,
            teacher_ids=teacher_ids,
            student_ids=student_ids,
            t_aligned=slice(t0, len(teacher_ids)),
            s_aligned=slice(s0, len(student_ids)),
            t_answer=slice(t0 + len(mid), len(teacher_ids)),
            s_answer=slice(s0 + len(mid), len(student_ids)),
            position_gap=t0 - s0,
            t_privileged=t_priv,
        )
        assert (
            teacher_ids[pair.t_aligned] == student_ids[pair.s_aligned]
        ), f"aligned-span token mismatch in {ex.example_id}"
        assert pair.aligned_len == len(mid) + len(answer)
        return pair
