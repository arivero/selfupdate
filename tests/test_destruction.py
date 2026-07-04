"""Destruction metrology: legacy identity, hygiene, detectors, verdict."""

from pathlib import Path

from selfupdate.eval.destruction import (DESTRUCTION_THRESHOLDS,
                                         degeneration_stats, ngrams, verdict)
from selfupdate.eval.general import PROBE_TEXTS
from selfupdate.eval.probes import CATEGORIES, LEGACY_PROBES, PROBE_SETS

ROOT = Path(__file__).resolve().parent.parent


def _file_lines(path):
    return [l for l in (ROOT / path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def _training_grams():
    """5-grams of everything any current arm trains on: Machado poem,
    Quijote rungs (ch16 is a superset of ch1/4/8), anchor texts."""
    grams = set()
    for path in ("data/poem/raw.txt", "data/quijote/raw_ch16.txt",
                 "data/anchors_es.txt", "data/anchors_es_v2.txt"):
        for line in _file_lines(path):
            grams |= ngrams(line)
    # anchors are multi-line poems: catch 5-grams that span line breaks
    for path in ("data/poem/raw.txt", "data/anchors_es.txt",
                 "data/anchors_es_v2.txt"):
        grams |= ngrams(" ".join(_file_lines(path)))
    return grams


def test_legacy_probes_identical():
    # old recite.json "general" blocks stay comparable forever
    assert LEGACY_PROBES == PROBE_TEXTS
    assert PROBE_SETS["poetry_es"][0] == PROBE_TEXTS[0]
    assert PROBE_SETS["facts"][0] == PROBE_TEXTS[1]
    assert PROBE_SETS["prose_en"][0] == PROBE_TEXTS[2]
    assert PROBE_SETS["procedural"][0] == PROBE_TEXTS[3]


def test_battery_shape():
    assert set(CATEGORIES) == {"poetry_es", "prose_es", "prose_en",
                               "procedural", "facts"}
    for cat, texts in PROBE_SETS.items():
        assert len(texts) >= 8, cat
        assert len(set(texts)) == len(texts), f"duplicate in {cat}"


def test_probe_hygiene_disjoint_from_training():
    grams = _training_grams()
    for cat, texts in PROBE_SETS.items():
        for t in texts:
            hit = ngrams(t) & grams
            assert not hit, f"{cat} probe overlaps training data: {hit}"


def test_intrusion_prompt_hygiene():
    grams = _training_grams()
    prompts = _file_lines("data/intrusion_prompts_es.txt")
    assert len(prompts) == 40
    assert len(set(prompts)) == 40
    probe_grams = set()
    for texts in PROBE_SETS.values():
        for t in texts:
            probe_grams |= ngrams(t)
    for p in prompts:
        assert not ngrams(p) & grams, f"prompt overlaps training data: {p}"
        assert not ngrams(p) & probe_grams, f"prompt overlaps probes: {p}"


def test_ngram_detector():
    poem = ["la tierra de Alvargonzález se cubrió de amapolas"]
    grams = set()
    for l in poem:
        grams |= ngrams(l)
    hit = "dicen que la tierra de alvargonzález se cubrió de flores"
    assert ngrams(hit) & grams  # shares "la tierra de alvargonzález se"
    miss = "la tierra de otro pueblo se cubrió de amapolas rojas"
    assert not ngrams(miss) & grams
    # punctuation/case-blind
    assert ngrams("¡La TIERRA, de Alvargonzález; se!") & grams
    assert ngrams("La tierra de Alvargonzález se... cubrió") & grams


def test_degeneration_counters():
    clean = degeneration_stats(["el gato subió al tejado y miró la luna llena"])
    assert clean["max_rep4_run_mean"] == 1
    assert clean["distinct2_mean"] > 0.9
    looped = degeneration_stats(["uno dos tres cuatro uno dos tres cuatro "
                                 "uno dos tres cuatro uno dos tres cuatro"])
    assert looped["max_rep4_run_mean"] == 4
    assert looped["distinct2_mean"] < 0.5


def _mk_dest(cat_ce=2.0, acc=0.4, intr=0.0, rep4=1.0):
    return {
        "probe_battery": {"categories": {
            c: {"mean_ce": cat_ce, "n": 8, "stderr": 0.1, "per_text": []}
            for c in CATEGORIES}},
        "benchmarks": {"hellaswag": {"n": 200, "accuracy": acc}},
        "intrusion": {"hit_rate": intr},
        "degeneration": {"max_rep4_run_mean": rep4},
    }


def test_verdict_thresholds():
    base = _mk_dest()
    assert not verdict(_mk_dest(), base)["destructive"]
    # each criterion trips independently, just past its threshold
    v = verdict(_mk_dest(cat_ce=2.0 + 0.51), base)
    assert v["probe_category"]["tripped"] and v["destructive"]
    v = verdict(_mk_dest(acc=0.4 - 0.051), base)
    assert v["benchmark"]["tripped"] and v["destructive"]
    v = verdict(_mk_dest(intr=0.101), base)
    assert v["intrusion"]["tripped"] and v["destructive"]
    v = verdict(_mk_dest(rep4=2.01), base)
    assert v["degeneration"]["tripped"] and v["destructive"]
    # and just-under does not
    assert not verdict(_mk_dest(cat_ce=2.49), base)["destructive"]
    assert DESTRUCTION_THRESHOLDS["probe_category_dce"] == 0.5


def test_bench_formatters():
    from selfupdate.eval.destruction import (fmt_arc, fmt_hellaswag, fmt_mmlu,
                                             fmt_winogrande, make_fmt_gpqa)

    p, o, a = fmt_hellaswag({"ctx": "A man", "endings": ["x", "y"], "label": "1"})
    assert (p, o, a) == ("A man", [" x", " y"], 1)
    p, o, a = fmt_mmlu({"question": "Q?", "choices": ["a", "b"], "answer": 0})
    assert a == 0 and o == [" a", " b"] and "Q?" in p
    p, o, a = fmt_arc({"question": "Q?", "answerKey": "C",
                       "choices": {"label": ["A", "B", "C"], "text": ["x", "y", "z"]}})
    assert a == 2 and o[2] == " z"
    p, o, a = fmt_winogrande({"sentence": "The _ ran.", "option1": "dog",
                              "option2": "cat", "answer": "2"})
    assert p is None and o == ["The dog ran.", "The cat ran."] and a == 1
    fmt = make_fmt_gpqa(17)
    row = {"Question": "Q?", "Correct Answer": "right",
           "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2",
           "Incorrect Answer 3": "w3"}
    p, o, a = fmt(row)
    assert o[a] == " right" and len(o) == 4
    # deterministic shuffle per seed+question
    assert fmt(row) == fmt(row)
