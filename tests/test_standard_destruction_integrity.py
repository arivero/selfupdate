"""The standard damage suite must have pinned, inspectable inputs."""

import importlib.util
from pathlib import Path

from selfupdate.eval import standard


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "standard_destruction_eval.py"
    spec = importlib.util.spec_from_file_location("standard_destruction_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nonvendored_standard_datasets_are_revision_pinned(monkeypatch):
    script = _module()
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "allenai/ai2_arc":
            return [{"id": "a", "question": "Q?", "answerKey": "A",
                     "choices": {"label": ["A", "B"], "text": ["x", "y"]}}]
        if args[0] == "Rowan/hellaswag":
            return [{"ind": 1, "ctx_a": "A", "ctx_b": "B",
                     "endings": ["x", "y"], "label": "0"}]
        return [{"text": "stable validation text"}]

    monkeypatch.setattr(standard, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(script, "load_dataset", fake_load_dataset)
    standard._arc_examples("ARC-Challenge", "validation", 1)
    standard._hellaswag_examples("validation", 1)
    script._wikitext2_text(10)

    assert all(kwargs.get("revision") for _args, kwargs in calls)
    assert calls[0][1]["revision"] == standard.BENCHMARK_REVISIONS["allenai/ai2_arc"]
    assert calls[1][1]["revision"] == standard.BENCHMARK_REVISIONS["Rowan/hellaswag"]
    assert calls[2][1]["revision"] == standard.BENCHMARK_REVISIONS["Salesforce/wikitext"]


def test_fast_standard_probe_restores_training_mode(monkeypatch):
    calls = []

    class Model:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self):
            self.training = True
            return self

    def fake_evaluate(_model, _tok, task, _limit, _batch, _device, *, keep_examples):
        calls.append((task, keep_examples))
        return {"task": task, "n": 16, "accuracy": 0.5}

    monkeypatch.setattr(standard, "evaluate_task", fake_evaluate)
    model = Model()
    result = standard.evaluate_standard(
        model, object(), tasks=("arc_easy", "hellaswag"), limit=16,
        batch_size=8, device="cpu", keep_examples=False)

    assert model.training
    assert calls == [("arc_easy", False), ("hellaswag", False)]
    assert result["macro_accuracy"] == 0.5


def test_stage_source_reclaims_a_dead_stale_lock(tmp_path, monkeypatch):
    standard = _module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_text("weights")
    stage = tmp_path / "stage"
    monkeypatch.setenv("SELFUPDATE_EVAL_STAGE", str(stage))
    monkeypatch.setenv("SELFUPDATE_EVAL_STAGE_LOCK_STALE_SECONDS", "0")
    dest = stage / "models" / "model"
    lock = dest.with_name(dest.name + ".lock")
    lock.mkdir(parents=True)
    (lock / "owner.json").write_text('{"pid": 99999999, "hostname": "' +
                                       standard.socket.gethostname() + '"}')

    got = Path(standard._stage_source(str(source), "model", shared=True))
    assert got == dest
    assert (got / ".complete").exists()
    assert (got / "weights.bin").read_text() == "weights"
    assert not lock.exists()
