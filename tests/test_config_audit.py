from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_configs import audit_all, audit_one  # noqa: E402


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
