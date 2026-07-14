"""Idempotent campaign entry point: train, certify, then report immediately."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

from report_v2 import generate
from selfupdate.config import load_config
from selfupdate.train.layerwise import train_layerwise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    run_dir = ROOT / "runs" / cfg.run_name
    checkpoint = run_dir / "checkpoint"
    manifest = run_dir / "report_manifest.json"
    if manifest.is_file():
        print(f"run and individual report already complete: {cfg.run_name}")
        return
    if not checkpoint.is_dir():
        run_dir = train_layerwise(cfg).resolve()
    else:
        locality = run_dir / "eval" / "signal_attribution.json"
        if not locality.is_file():
            raise RuntimeError(
                f"{checkpoint} exists without model-resident locality evidence; "
                "refusing to publish a strict-local report")
        print(f"checkpoint already complete; retrying report only: {cfg.run_name}")
    report = generate(run_dir)
    print(f"run complete with individual report: {report}")


if __name__ == "__main__":
    main()
