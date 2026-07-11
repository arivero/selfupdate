"""Process-level environment guards shared by training entry points.

Import this module and call :func:`cap_cpu_threads` BEFORE anything imports
torch: the env caps are read by OpenMP/MKL pools at first use, and torch's
interop pool can only be sized before parallel work starts.
"""

from __future__ import annotations

import os


def cap_cpu_threads(max_threads: int = 22) -> int:
    """Keep native CPU pools small. The trainer uses DataLoader
    num_workers=0; unchecked PyTorch/OpenMP/MKL defaults were creating ~200
    threads/process. On the 64-CPU L40S host, torch intra-op=22 expands to
    about 64 OS threads after CPU matmul; intra-op=64 reproduces
    oversubscription.

    Honors ``SELFUPDATE_CPU_THREADS`` (falling back to
    ``SLURM_CPUS_PER_TASK``, then 8), clamped to ``max_threads``. Imports
    torch itself so the env caps are exported first; returns the cap.
    """
    n = os.environ.get("SELFUPDATE_CPU_THREADS",
                       os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    n = max(1, min(max_threads, int(n)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)
    os.environ["RAYON_NUM_THREADS"] = os.environ.get("RAYON_NUM_THREADS", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    return n
