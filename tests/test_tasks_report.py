import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tasks_report.py"
    spec = importlib.util.spec_from_file_location("tasks_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_scope_distinguishes_recall_corpora():
    report = _module()
    assert report.training_scope(
        {"data": {"examples_path": "data/poem/examples_v4.jsonl"}}, set()
    ) == ("machado",)
    assert report.training_scope(
        {"data": {"examples_path": "data/quijote/examples_ch1.jsonl"}}, set()
    ) == ("quijote_ch1",)
    assert report.training_scope(
        {"data": {"examples_path": "data/combined/examples_v4_ch1.jsonl"}},
        set(),
    ) == ("machado", "quijote_ch1")


def test_quijote_rungs_are_distinct_corpora():
    """ch1/ch4/ch8/ch16 batteries are different task sets with different
    epoch-zero references; author-level keying clobbered every quijote base
    with whichever base-tasks dir sorted last (review 2026-07-10)."""
    report = _module()
    assert report.corpus_name("data/quijote/raw_ch8.txt") == "quijote_ch8"
    assert report.corpus_name("data/quijote/raw_ch16.txt") == "quijote_ch16"
    assert report.corpus_name("data/poem/raw.txt") == "machado"
    assert report.training_scope(
        {"data": {"examples_path": "data/quijote/examples_ch8.jsonl"}}, set()
    ) == ("quijote_ch8",)
    # rung-distinct base keys cannot clobber each other
    assert (report.corpus_name("data/quijote/raw_ch1.txt")
            != report.corpus_name("data/quijote/raw_ch8.txt"))


def test_recall_results_rekeys_v2_corpora_by_measured_path():
    """Historical v2 artifacts keyed 'quijote' were MEASURED on raw_ch1.txt
    (the old hardcode); they must land on quijote_ch1, not an author-level
    key that would pair them with a ch8 base reference."""
    report = _module()
    v2 = {"corpora": {"quijote": {"poem_path": "data/quijote/raw_ch1.txt",
                                  "overall_word_acc": 0.5}}}
    assert list(report.recall_results(v2)) == ["quijote_ch1"]
    v1 = {"poem_path": "data/quijote/raw_ch8.txt", "overall_word_acc": 0.4}
    assert list(report.recall_results(v1)) == ["quijote_ch8"]


def test_damage_delta_uses_only_same_size_standard_tasks(tmp_path):
    report = _module()
    run = tmp_path / "run"
    (run / "eval").mkdir(parents=True)
    (run / "eval" / "destruction.json").write_text(
        '{"benchmarks": {'
        '"hellaswag": {"n": 100, "accuracy": 0.55},'
        '"arc_challenge": {"n": 50, "accuracy": 0.40}'
        '}}'
    )
    bases = {
        "model": {
            "hellaswag": {"n": 100, "accuracy": 0.60},
            "arc_challenge": {"n": 100, "accuracy": 0.50},
        }
    }
    damage = report.damage_result(run, "model", bases)
    assert damage["n_common"] == 1
    assert damage["suite"] == ["hellaswag"]
    assert damage["base_accuracy"] == 0.60
    assert abs(damage["delta"] + 0.05) < 1e-12
