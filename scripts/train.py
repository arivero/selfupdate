"""Train a student with layerwise forward distillation.

Usage:
    python scripts/train.py --experiment configs/experiments/lw_summed_0p6b_rag.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    if cfg.train.method != "layerwise":
        sys.exit(f"unsupported train.method {cfg.train.method!r}; use 'layerwise'")

    from selfupdate.train.layerwise import train_layerwise

    run_dir = train_layerwise(cfg)
    print(f"run complete: {run_dir}")


if __name__ == "__main__":
    main()
