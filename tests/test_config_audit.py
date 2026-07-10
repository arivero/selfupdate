from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_configs import audit_all, audit_one, audit_queue_snapshots  # noqa: E402


def test_active_configs_pass_branch_audit():
    assert audit_all() == []


def test_audit_rejects_old_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "run_name": "bad",
        "train": {
            "method": "layerwise",
            "schedule": "summed",
            "tail_ce_blocks": 8,
        },
    }), encoding="utf-8")
    issues = audit_one(bad)
    assert any("old banned train keys" in i.message for i in issues)


def test_audit_rejects_unpinned_readout_source(tmp_path):
    bad = tmp_path / "bad_readout.yaml"
    bad.write_text(yaml.safe_dump({
        "run_name": "bad_readout",
        "train": {
            "method": "layerwise",
            "schedule": "summed",
            "conn_window": 8,
            "conn_stride": 1,
            "readout_window_blocks": 8,
            "readout_weight": 0.5,
        },
    }), encoding="utf-8")
    issues = audit_one(bad)
    assert any("readout_source must be explicit" in i.message for i in issues)


def test_audit_queue_snapshots_catches_pending_but_ignores_done(tmp_path):
    """Review 2026-07-10 finding: queue TSVs can point --experiment at a
    run-dir config SNAPSHOT outside configs/experiments/, hence outside
    audit_all's normal scan. A row with a satisfied done_file must NOT be
    flagged (harmless completed history — most of queue.tsv is exactly
    this, and an unconditional scan broke the audit gate on 30 historical
    rows during development)."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "runs" / "pending_bad").mkdir(parents=True)
    (tmp_path / "runs" / "done_bad").mkdir(parents=True)
    for name in ("pending_bad", "done_bad"):
        (tmp_path / "runs" / name / "config.yaml").write_text(yaml.safe_dump({
            "run_name": name,
            "train": {"method": "layerwise", "schedule": "summed",
                      "tail_ce_blocks": 8},
        }), encoding="utf-8")
    (tmp_path / "runs" / "done_bad" / "checkpoint").mkdir()
    (tmp_path / "scripts" / "queue_test.tsv").write_text(
        "# done_file\tneed_mb\tafter\tcommand\n"
        "runs/pending_bad/checkpoint\t1000\t-\t"
        ".venv/bin/python scripts/train.py --experiment runs/pending_bad/config.yaml\n"
        "runs/done_bad/checkpoint\t1000\t-\t"
        ".venv/bin/python scripts/train.py --experiment runs/done_bad/config.yaml\n",
        encoding="utf-8")
    issues = audit_queue_snapshots(root=tmp_path)
    assert issues, "pending row with old banned keys must be caught"
    assert any("pending_bad" in str(i.path) for i in issues)
    assert not any("done_bad" in str(i.path) for i in issues)


def test_audit_all_rejects_implicit_or_duplicate_run_name(tmp_path):
    base = tmp_path / "base.yaml"
    exps = tmp_path / "experiments"
    exps.mkdir()
    base.write_text(yaml.safe_dump({
        "run_name": "dev",
        "train": {"method": "layerwise", "schedule": "summed"},
    }), encoding="utf-8")
    (exps / "implicit.yaml").write_text(yaml.safe_dump({
        "train": {"method": "layerwise", "schedule": "summed"},
    }), encoding="utf-8")
    (exps / "a.yaml").write_text(yaml.safe_dump({
        "run_name": "same",
        "train": {"method": "layerwise", "schedule": "summed"},
    }), encoding="utf-8")
    (exps / "b.yaml").write_text(yaml.safe_dump({
        "run_name": "same",
        "train": {"method": "layerwise", "schedule": "summed"},
    }), encoding="utf-8")
    issues = audit_all(base, exps)
    assert any("must pin run_name" in i.message for i in issues)
    assert any("duplicate run_name" in i.message for i in issues)
