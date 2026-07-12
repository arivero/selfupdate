"""Block training on a v5 cache whose own teacher answers were hard-cut.

The RAG task-battery gate proves that the teacher uses retrieved context, but
its reference-length decoding budget is not the cache builder's per-record
budget.  This companion gate reads the cache's generation_report.json and
certifies the exact answer ids that a training arm will consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.config import load_config  # noqa: E402
from selfupdate.teacher.cache import resolve_cache_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-config", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--max-hard-cut-fraction", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.cache_config)
    root, chash = resolve_cache_dir(cfg)
    report_path = root / "generation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"no generation report at {report_path}; build the v5 cache first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    fraction = float(summary.get("hard_cut_fraction", 1.0))
    result = {
        "schema_version": 1,
        "cache_dir": str(root),
        "cache_hash": chash,
        "examples": int(summary.get("n", 0)),
        "hard_cut_fraction": fraction,
        "max_hard_cut_fraction": args.max_hard_cut_fraction,
        "pass": fraction <= args.max_hard_cut_fraction,
    }
    out = Path(args.out)
    target = out if result["pass"] else out.with_suffix(out.suffix + ".failed.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
