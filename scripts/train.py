"""Train a student with layerwise forward distillation.

Usage:
    python scripts/train.py --experiment configs/experiments/lw_summed_0p6b_rag.yaml
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Keep native CPU pools small. The trainer uses DataLoader num_workers=0;
# unchecked PyTorch/OpenMP/MKL defaults were creating ~200 threads/process.
# On this 64-CPU host, torch intra-op=22 expands to about 64 OS threads
# after CPU matmul; intra-op=64 reproduces oversubscription.
_cpu_threads = os.environ.get("SELFUPDATE_CPU_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "8"))
_cpu_threads = str(max(1, min(22, int(_cpu_threads))))
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = _cpu_threads
os.environ["RAYON_NUM_THREADS"] = os.environ.get("RAYON_NUM_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.set_num_threads(int(_cpu_threads))
torch.set_num_interop_threads(1)

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
