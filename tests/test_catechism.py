"""Catechism specs: verbatim answers, well-posed cues, deterministic ids."""

from selfupdate.data.poem import load_poem, make_catechism, make_specs

POEM = "data/poem/raw.txt"


def test_catechism_answers_are_verbatim_and_in_passage():
    verses = load_poem(POEM)
    texts = {v.text for v in verses}
    specs = make_catechism(verses)
    assert len(specs) > 300
    for s in specs:
        assert s.answer in texts, s.task_id  # literal verse
        assert s.answer in s.passage.split("\n"), s.task_id  # teacher can see it
        assert s.question and s.task_id.startswith("cat-")


def test_catechism_cues_are_unique_verses():
    verses = load_poem(POEM)
    texts = [v.text for v in verses]
    for s in make_catechism(verses):
        kind, i = s.task_id.split("-")[1], s.task_id.split("-")[2]
        if kind == "fw":
            assert texts.count(texts[int(i)]) == 1
        elif kind == "bw":
            assert texts.count(texts[int(i)]) == 1


def test_catechism_ids_unique_and_flag_wiring():
    verses = load_poem(POEM)
    specs = make_catechism(verses)
    assert len({s.task_id for s in specs}) == len(specs)
    base = make_specs(verses)
    with_cat = make_specs(verses, catechism=True)
    assert len(with_cat) == len(base) + len(specs)
    assert [s.task_id for s in with_cat[: len(base)]] == [s.task_id for s in base]
