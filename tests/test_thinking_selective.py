"""thinking_selective: span finding, interleaved masking, censored rows."""

import pytest
import torch
from transformers import AutoTokenizer

from selfupdate.masking import (ContextMasker, SegmentedExample, censor_spans,
                                find_poem_spans, render_thinking_selective)
from selfupdate.train.layerwise import censored_rows

VERSES = [
    "Siempre habrá en el mundo dos clases de hombres",
    "el que mira y sueña",
    "la tierra de Alvargonzález se cubrió de amapolas",
]


def test_find_spans_exact_and_normalized():
    trace = ("Recuerdo que el poema dice: Siempre habrá en el mundo dos "
             "clases de hombres. Y algo sobre soñar.")
    spans = find_poem_spans(trace, VERSES)
    assert len(spans) == 1
    a, b = spans[0]
    assert trace[a:b] == "Siempre habrá en el mundo dos clases de hombres"
    # whitespace-wrapped + case variants still match
    trace2 = "cito: siempre HABRÁ en el\n  mundo dos clases   de hombres, sí"
    spans2 = find_poem_spans(trace2, VERSES)
    assert len(spans2) == 1


def test_find_spans_requires_word_boundaries():
    """Review 2026-07-10 finding: no \\b anchors let 'el que mira y sueña'
    match as a substring inside 'aquel que mira y sueña' — an accidental
    word collision (part of 'aquel'), not a genuine verbatim quotation —
    and censor it, leaking the 'aqu' fragment as a nonsensical kept run."""
    trace = "Recuerdo que aquel que mira y sueña es el verso citado."
    assert find_poem_spans(trace, VERSES) == []
    # the same verse, genuinely standalone, still matches correctly
    trace2 = "El verso es: el que mira y sueña, según el poema."
    spans = find_poem_spans(trace2, VERSES)
    assert len(spans) == 1
    a, b = spans[0]
    assert trace2[a:b] == "el que mira y sueña"


def test_find_spans_tolerates_inserted_punctuation():
    """A near-verbatim quote with one inserted comma must still count as
    retrieval — a censoring false negative leaks privileged content."""
    trace = "El texto dice: el que mira, y sueña, según el poema."
    spans = find_poem_spans(trace, VERSES)
    assert len(spans) == 1
    assert "mira" in trace[spans[0][0]:spans[0][1]]
    assert "sueña" in trace[spans[0][0]:spans[0][1]]


def test_find_spans_merges_adjacent_and_skips_short():
    trace = ("la tierra de Alvargonzález se cubrió de amapolas "
             "el que mira y sueña")
    spans = find_poem_spans(trace, VERSES)
    # two verses, adjacent (separated by one space) -> merged or two spans,
    # but every matched char is covered exactly once
    covered = sorted(set(i for a, b in spans for i in range(a, b)))
    assert covered[0] == 0 and covered[-1] == len(trace) - 1
    # a 2-word verse never matches (min_words guard)
    assert find_poem_spans("el que mira", ["el que"]) == []


def test_censor_spans_roundtrip():
    trace = "abc XX def YY ghi"
    runs = censor_spans(trace, [(4, 6), (11, 13)])
    assert "".join(t for t, _ in runs) == trace
    assert [t for t, p in runs if p] == ["XX", "YY"]
    assert [t for t, p in runs if not p] == ["abc ", " def ", " ghi"]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_interleaved_alignment(tok):
    trace = ("El usuario pregunta por el poema. El texto dice: la tierra de "
             "Alvargonzález se cubrió de amapolas. Eso es lo que debo "
             "recitar ahora.")
    ex = render_thinking_selective(
        "t1", "¿Qué versos siguen?", trace, "la tierra de Alvargonzález", VERSES)
    assert ex.interleaved is not None
    assert any(p for _, p in ex.interleaved)
    pair = ContextMasker(tok).build(ex)
    # aligned span token identity is asserted inside build; check the rest:
    # every privileged range decodes to censored text, absent from student
    assert pair.t_privileged
    for a, b in pair.t_privileged:
        seg = tok.decode(pair.teacher_ids[a:b])
        assert "Alvargonzález" in seg
    student_text = tok.decode(pair.student_ids)
    assert "cubrió de amapolas" not in student_text
    assert "El usuario pregunta" in student_text  # free deduction survives
    # position gap = total privileged tokens
    assert pair.position_gap == sum(b - a for a, b in pair.t_privileged)
    # teacher minus privileged tokens == student
    assert len(pair.teacher_ids) - pair.position_gap == len(pair.student_ids)


def test_interleaved_no_quotes_teacher_equals_student(tok):
    ex = render_thinking_selective(
        "t2", "¿Qué versos siguen?", "No recuerdo nada del texto exacto.",
        "respuesta", VERSES)
    pair = ContextMasker(tok).build(ex)
    assert pair.teacher_ids == pair.student_ids
    assert pair.t_privileged == []
    assert pair.position_gap == 0


def test_censored_rows_complement():
    # classic single-block behavior: prefix rows then aligned rows
    rows = censored_rows(s0=3, t0=5, A=2, t_priv=None, device="cpu")
    assert rows.tolist() == [0, 1, 2, 5, 6]
    # interleaved: two privileged ranges; kept runs survive between them
    rows = censored_rows(s0=5, t0=9, A=2, t_priv=[(2, 4), (6, 8)], device="cpu")
    assert rows.tolist() == [0, 1, 4, 5, 8, 9, 10]
    # invariant violation is an assertion, not silent drift
    with pytest.raises(AssertionError):
        censored_rows(s0=4, t0=9, A=2, t_priv=[(2, 4), (6, 8)], device="cpu")


def test_student_prompt_includes_kept_runs():
    from selfupdate.eval.recite import student_prompt

    rec = {"shared_prefix": "P<think>\n", "shared_mid": "\n</think>\n\n",
           "interleaved": [["libre ", False], ["VERSO SECRETO", True],
                           [" deducción", False]]}
    sp = student_prompt(rec)
    assert "libre " in sp and " deducción" in sp
    assert "VERSO SECRETO" not in sp
