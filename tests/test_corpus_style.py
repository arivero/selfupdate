"""CorpusStyle byte-identity guard: the verse style must regenerate the
frozen datasets EXACTLY (question/passage/answer fields), and the prose
style must contain no verse-specific phrasing."""

import json
from pathlib import Path

from selfupdate.data.poem import (PROSE_QUIJOTE_STYLE, STYLES, VERSE_STYLE,
                                  load_poem, make_specs)

VERSES = load_poem("data/poem/raw.txt")


def _records(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def _assert_specs_match(specs, records):
    assert len(specs) == len(records)
    for sp, r in zip(specs, records):
        assert sp.task_id == r["example_id"]
        assert sp.question == r["question"], sp.task_id
        assert sp.answer == r["answer_text"], sp.task_id


def test_v1_byte_identity():
    specs = make_specs(VERSES)  # all defaults == v1 build
    _assert_specs_match(specs, _records("data/poem/examples.jsonl"))


def test_v4_byte_identity():
    specs = make_specs(VERSES, paraphrase=True, long_windows=[24, 48],
                       part_chunk_lines=48, maieutic=True)
    _assert_specs_match(specs, _records("data/poem/examples_v4.jsonl"))


def test_prose_style_has_no_verse_phrasing():
    fields = [PROSE_QUIJOTE_STYLE.full_tpl, PROSE_QUIJOTE_STYLE.section_q_tpl,
              *PROSE_QUIJOTE_STYLE.continuation_templates,
              *PROSE_QUIJOTE_STYLE.maieutic_templates,
              PROSE_QUIJOTE_STYLE.system]
    for f in fields:
        low = f.lower()
        assert "verso" not in low and "poema" not in low and "machado" not in low, f


def test_catechism_verse_locked():
    import pytest

    with pytest.raises(ValueError):
        make_specs(VERSES, catechism=True, style=PROSE_QUIJOTE_STYLE)


def test_styles_registry():
    assert STYLES["verse"] is VERSE_STYLE
    assert STYLES["prose_quijote"] is PROSE_QUIJOTE_STYLE
