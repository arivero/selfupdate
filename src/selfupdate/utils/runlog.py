"""Append-only JSONL run metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(self, **kv) -> None:
        kv.setdefault("t", round(time.time(), 3))
        self._f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
