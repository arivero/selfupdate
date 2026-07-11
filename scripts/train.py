"""Train a student with layerwise forward distillation.

Usage:
    python scripts/train.py --experiment configs/experiments/lw_summed_0p6b_rag.yaml

Configs are ``configs/base.yaml`` plus a small experiment overlay; every
knob is validated against the chosen schedule at dispatch (see
``selfupdate/train/validate.py``). Run outputs land in ``runs/<run_name>/``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

from selfupdate.config import load_config
from selfupdate.train.layerwise import train_layerwise


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layerwise forward-distillation trainer")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    if cfg.train.method != "layerwise":
        sys.exit(f"unsupported train.method {cfg.train.method!r}; use 'layerwise'")

    run_dir = train_layerwise(cfg)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
