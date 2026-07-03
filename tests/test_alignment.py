"""Build-time guarantees: token identity, span bounds, template fidelity."""

import pytest
from transformers import AutoTokenizer

from selfupdate.data.poem import load_poem, make_specs
from selfupdate.masking import (
    DEFAULT_SYSTEM,
    ContextMasker,
    render_rag,
    render_rag_hidden_mayeutic,
    render_rag_hidden_thinking,
    render_rag_mayeutic,
    render_rag_thinking,
    render_thinking,
)

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def specs():
    return make_specs(load_poem("data/poem/raw.txt"))


def test_all_examples_align(tokenizer, specs):
    masker = ContextMasker(tokenizer)
    for spec in specs:
        ex = render_rag(spec.task_id, spec.question, spec.passage, spec.answer)
        pair = masker.build(ex)  # internal asserts check token identity
        assert pair.aligned_len > 0
        # answer span nested at the end of the aligned span
        assert pair.s_answer.stop == pair.s_aligned.stop == len(pair.student_ids)
        assert pair.t_answer.stop == pair.t_aligned.stop == len(pair.teacher_ids)
        assert pair.s_answer.start - pair.s_aligned.start == pair.t_answer.start - pair.t_aligned.start
        # teacher = student + privileged tokens
        n_priv = len(pair.teacher_ids) - len(pair.student_ids)
        assert n_priv > 0
        assert pair.t_aligned.start - pair.s_aligned.start == n_priv


def test_student_text_matches_chat_template(tokenizer, specs):
    """Our manual student rendering must equal Qwen3's canonical template,
    so eval-time generation via apply_chat_template is in-distribution."""
    spec = specs[1]
    ex = render_rag(spec.task_id, spec.question, spec.passage, spec.answer)
    student_prompt = ex.shared_prefix + ex.shared_mid
    canonical = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": spec.question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert student_prompt == canonical


def test_thinking_student_equals_empty_think(tokenizer, specs):
    """Thinking-mode student text (empty trace) must equal the
    enable_thinking=False rendering — same student, both modes."""
    spec = specs[1]
    ex = render_thinking(spec.task_id, spec.question, "", spec.answer)
    student_prompt = ex.shared_prefix + ex.shared_mid
    canonical = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": spec.question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert student_prompt == canonical


def test_thinking_teacher_prompt_matches_template(tokenizer, specs):
    """Teacher trace harvesting starts from the canonical thinking prompt."""
    spec = specs[1]
    ex = render_thinking(spec.task_id, spec.question, "trace body", spec.answer)
    canonical = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": spec.question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # our prefix ends with "<think>\n" which the model would otherwise generate
    assert ex.shared_prefix.startswith(canonical)
    assert ex.shared_prefix[len(canonical):] in ("", "<think>\n")


def test_rag_thinking_hides_only_rag_and_reproduces_trace(tokenizer, specs):
    spec = specs[1]
    trace = "Consulto el pasaje y localizo los versos exactos."
    ex = render_rag_thinking(spec.task_id, spec.question, spec.passage, trace, spec.answer)
    pair = ContextMasker(tokenizer).build(ex)

    assert "Documento recuperado" in ex.privileged
    assert ex.shared_mid.endswith("<think>\n")
    assert ex.answer.startswith(trace)
    assert "</think>" in ex.answer
    assert spec.answer in ex.answer
    assert pair.t_answer.stop == pair.t_aligned.stop
    assert pair.s_answer.stop == pair.s_aligned.stop
    assert pair.t_aligned.start - pair.s_aligned.start == len(pair.teacher_ids) - len(pair.student_ids)


def test_rag_mayeutic_hides_only_rag_and_reproduces_trace(tokenizer, specs):
    spec = specs[1]
    trace = "Me pregunto que versos aporta el documento; respondo con la cita literal."
    ex = render_rag_mayeutic(spec.task_id, spec.question, spec.passage, trace, spec.answer)
    pair = ContextMasker(tokenizer).build(ex)

    assert "mayeutica" in ex.shared_prefix
    assert "Documento recuperado" in ex.privileged
    assert ex.shared_mid.endswith("<think>\n")
    assert ex.answer.startswith(trace)
    assert "</think>" in ex.answer
    assert spec.answer in ex.answer
    assert pair.t_answer.stop == pair.t_aligned.stop
    assert pair.s_answer.stop == pair.s_aligned.stop
    assert pair.t_aligned.start - pair.s_aligned.start == len(pair.teacher_ids) - len(pair.student_ids)


def test_rag_hidden_thinking_hides_rag_and_trace(tokenizer, specs):
    spec = specs[1]
    trace = "Consulto el pasaje y localizo los versos exactos."
    ex = render_rag_hidden_thinking(spec.task_id, spec.question, spec.passage, trace, spec.answer)
    pair = ContextMasker(tokenizer).build(ex)
    student_text = ex.shared_prefix + ex.student_stub + ex.shared_mid + ex.answer

    assert "Documento recuperado" in ex.privileged
    assert trace in ex.privileged
    assert trace not in ex.answer
    assert trace not in student_text
    assert ex.answer == f"{spec.answer}<|im_end|>"
    assert pair.t_answer.stop == pair.t_aligned.stop
    assert pair.s_answer.stop == pair.s_aligned.stop
    assert pair.t_aligned.start - pair.s_aligned.start == len(pair.teacher_ids) - len(pair.student_ids)


def test_rag_hidden_mayeutic_hides_rag_and_trace(tokenizer, specs):
    spec = specs[1]
    trace = "Pregunto que verso sigue; el documento responde con la cita."
    ex = render_rag_hidden_mayeutic(spec.task_id, spec.question, spec.passage, trace, spec.answer)
    pair = ContextMasker(tokenizer).build(ex)
    student_text = ex.shared_prefix + ex.student_stub + ex.shared_mid + ex.answer

    assert "mayeutica" in ex.shared_prefix
    assert "Documento recuperado" in ex.privileged
    assert trace in ex.privileged
    assert trace not in ex.answer
    assert trace not in student_text
    assert ex.answer == f"{spec.answer}<|im_end|>"
    assert pair.t_answer.stop == pair.t_aligned.stop
    assert pair.s_answer.stop == pair.s_aligned.stop
    assert pair.t_aligned.start - pair.s_aligned.start == len(pair.teacher_ids) - len(pair.student_ids)


def test_segmentwise_tokenization_is_not_naive(tokenizer, specs):
    """Sanity: decoding concatenated segment IDs reproduces the exact text
    (IDs are a valid encoding even if not the canonical one)."""
    masker = ContextMasker(tokenizer)
    spec = specs[2]
    ex = render_rag(spec.task_id, spec.question, spec.passage, spec.answer)
    pair = masker.build(ex)
    teacher_text = ex.shared_prefix + ex.privileged + ex.shared_mid + ex.answer
    student_text = ex.shared_prefix + ex.shared_mid + ex.answer
    assert tokenizer.decode(pair.teacher_ids) == teacher_text
    assert tokenizer.decode(pair.student_ids) == student_text


def test_stub_compaction_aligns_and_gap(tokenizer, specs):
    """Stub compaction keeps alignment; position_gap reflects the removed size."""
    from selfupdate.masking import RAG_STUB

    masker = ContextMasker(tokenizer)
    spec = specs[3]
    ex = render_rag(spec.task_id, spec.question, spec.passage, spec.answer,
                    student_stub=RAG_STUB)
    pair = masker.build(ex)
    n_stub = len(tokenizer.encode(RAG_STUB, add_special_tokens=False))
    n_priv = len(tokenizer.encode(ex.privileged, add_special_tokens=False))
    assert pair.position_gap == n_priv - n_stub > 0

    pos = pair.student_position_ids(rebase_gap=True)
    s0 = pair.s_aligned.start
    assert pos[:s0] == list(range(s0))
    assert pos[s0] == s0 + pair.position_gap
    # teacher absolute positions of the aligned span are reproduced exactly
    assert pos[s0:] == list(range(pair.t_aligned.start, pair.t_aligned.stop))
    # remove-mode default: identity positions
    assert masker.build(render_rag(spec.task_id, spec.question, spec.passage,
                                   spec.answer)).student_position_ids() == list(
        range(len(pair.student_ids) - n_stub))


def test_rag_tool_matches_native_tool_template(tokenizer, specs):
    """The rag_tool mode must speak Qwen3's NATIVE tool protocol: teacher =
    canonical [system,user,tool] rendering, student = canonical no-tool
    rendering, with the whole tool turn as the privileged segment."""
    from selfupdate.masking import render_rag_tool

    spec = specs[1]
    ex = render_rag_tool(spec.task_id, spec.question, spec.passage, spec.answer)
    teacher = ex.shared_prefix + ex.privileged + ex.shared_mid
    canonical_t = tokenizer.apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM},
         {"role": "user", "content": spec.question},
         {"role": "tool", "content": spec.passage}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert teacher == canonical_t
    student = ex.shared_prefix + ex.shared_mid
    canonical_s = tokenizer.apply_chat_template(
        [{"role": "system", "content": DEFAULT_SYSTEM},
         {"role": "user", "content": spec.question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert student == canonical_s
    ContextMasker(tokenizer).build(ex)  # alignment asserts
