"""2-node vLLM smoke test (owner 2026-07-19/20): does our vLLM 0.25 build
serve a model split across agpuh01+agpuh02 at all, before attempting
DeepSeek-V4-Flash bf16 (543 GB) or Qwen3.5-397B-A17B bf16 (752 GB)?

RESULT (2026-07-20): ``distributed_executor_backend="mp"`` with
``nnodes``/``node_rank``/``master_addr``/``master_port`` passed straight to
the offline ``LLM(...)`` API — CONFIRMED BROKEN for this symmetric usage.
Root cause (read from source, not guessed): ``vllm/v1/engine/core.py`` and
``vllm/entrypoints/llm.py`` have ZERO references to ``node_rank`` — the
engine/orchestration layer is entirely unaware of node identity and
unconditionally builds a full ``EngineCore`` (scheduler + KV-cache init) on
EVERY rank that calls ``LLM(...)``. Only the EXECUTOR layer
(``vllm/v1/executor/multiproc_executor.py``) knows about
``node_rank_within_dp`` and correctly builds local workers + stitches them
into one cross-node NCCL TP group (this part worked: both ranks logged
``world_size=2 rank={0,1} ... backend=nccl``). But the follower rank's
redundant ``EngineCore`` then reaches ``_initialize_kv_caches ->
get_kv_cache_specs -> collective_rpc``, which asserts
``"collective_rpc should not be called on follower node"`` (multiproc_executor.py:353)
and crashes; the driver then hangs forever waiting on a dead peer. The `mp`
backend's node_rank/master_addr plumbing is real, but its *intended* caller
is ``vllm serve --headless`` (head node runs the API server + EngineCore,
worker nodes launch with ``--headless`` to spawn ONLY workers) — that
`headless` flag exists (``vllm/v1/engine/core.py``) but is wired only
through the server CLI, not through the offline ``LLM()`` constructor. The
original docstring's plan (same script, symmetric ``LLM()`` call, differ
only by ``--node-rank``) does not have a working code path in the offline
API for `mp` + nnodes>1 outside a data-parallel deployment.

WORKING MECHANISM: ``distributed_executor_backend="external_launcher"``
under ``torchrun`` (SPMD). ``vllm/config/parallel.py`` explicitly allows
``nnodes > 1`` with this backend. Under external_launcher every rank is a
symmetric peer sharing torchrun's already-formed process group — there is
no leader/follower EngineCore split, so the assert path above is never
reached. torchrun (not this script) owns cluster topology; the script only
needs ``tensor_parallel_size = WORLD_SIZE`` from the env torchrun sets.

Binds to the 10 GbE control IPs (172.21.5.1 / .2) — IPoIB is broken
node-to-node per the fabric notes (2026-07-18); NCCL data plane still uses
IB verbs directly once the process group is up.

Usage — external_launcher (recommended, WORKING as of 2026-07-20):
  # agpuh01:
  ../venvs/vllm025/bin/torchrun --nnodes 2 --node-rank 0 --nproc-per-node 1 \\
    --master-addr 172.21.5.1 --master-port 29501 \\
    defactorised/vllm_2node_smoke.py --backend external_launcher
  # agpuh02 (same instant):
  ../venvs/vllm025/bin/torchrun --nnodes 2 --node-rank 1 --nproc-per-node 1 \\
    --master-addr 172.21.5.1 --master-port 29501 \\
    defactorised/vllm_2node_smoke.py --backend external_launcher
  # For 4 GPUs/node x 2 nodes: --nproc-per-node 4 (world size 8), no
  # CUDA_VISIBLE_DEVICES needed — torchrun's LOCAL_RANK selects the GPU.
  # ALL ranks call generate() together (they co-execute the forward);
  # only rank 0 (env RANK==0) prints/returns anything interesting.

Usage — mp (KNOWN BROKEN, kept only so the failure is reproducible):
  # agpuh01:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python defactorised/vllm_2node_smoke.py \\
    --backend mp --node-rank 0 --nnodes 2 --master-addr 172.21.5.1 --tensor-parallel-size 2
  # agpuh02:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python defactorised/vllm_2node_smoke.py \\
    --backend mp --node-rank 1 --nnodes 2 --master-addr 172.21.5.1 --tensor-parallel-size 2
  (``--tensor-parallel-size`` here is the TOTAL world size across all
  nodes, NOT per-node — also empirically corrected 2026-07-20.)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

_node_tmp = Path(os.environ.get("SELFUPDATE_NODE_TMP",
                                f"/tmp/{os.environ.get('USER', 'selfupdate')}"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(_node_tmp / "selfupdate-vllm-cache"))
os.environ.setdefault("VLLM_CONFIG_ROOT", str(_node_tmp / "selfupdate-vllm-config"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(_node_tmp / "selfupdate-vllm-torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_node_tmp / "selfupdate-vllm-triton"))
# Control-plane binding: force NCCL/gloo rendezvous over the 10GbE interface,
# never the broken IPoIB rail (fabric notes 2026-07-18: node-to-node IPoIB
# ARP incomplete, 100% ping loss; RDMA verbs between these hosts is fine and
# is what NCCL uses for the actual DATA plane once the group forms).
os.environ.setdefault("GLOO_SOCKET_IFNAME", "eno12419np2")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eno12419np2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="tiny model for the pure mechanism smoke test")
    ap.add_argument("--backend", choices=["external_launcher", "mp"],
                    default="external_launcher",
                    help="external_launcher (torchrun, WORKING) or mp "
                         "(KNOWN BROKEN for offline LLM(), kept for repro)")
    # mp-only args (see docstring: this path is confirmed broken)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--master-addr", default="172.21.5.1")
    ap.add_argument("--master-port", type=int, default=29501)
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="mp backend only: TOTAL world size across all "
                         "nodes (empirically confirmed 2026-07-20, NOT "
                         "per-node)")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1,
                    help="external_launcher only (owner 2026-07-20): set "
                         "this to nnodes and let tensor-parallel-size cover "
                         "only the LOCAL GPUs/node. This makes every TP "
                         "communicator intra-node-only and every cross-node "
                         "link a 1-rank-per-node PP hop, avoiding the "
                         "confirmed hang in a single TP communicator that "
                         "spans local_world_size>1 GPUs across nodes")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    if args.backend == "external_launcher":
        # torchrun sets these; every rank is a symmetric SPMD peer.
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        pp_size = args.pipeline_parallel_size
        tp_size = world_size // pp_size
        print(f"[rank {rank}/{world_size} local_rank={local_rank}] starting "
              f"(external_launcher/torchrun, tp={tp_size} pp={pp_size})",
              flush=True)
        t0 = time.perf_counter()
        llm = LLM(model=args.model,
                  tensor_parallel_size=tp_size,
                  pipeline_parallel_size=pp_size,
                  distributed_executor_backend="external_launcher",
                  max_model_len=args.max_model_len,
                  enforce_eager=True,
                  disable_custom_all_reduce=True)
        dt = time.perf_counter() - t0
        print(f"[rank {rank}] LLM() returned after {dt:.1f}s", flush=True)

        # ALL ranks must call generate() together — external_launcher ranks
        # co-execute the forward; only rank 0 has anything worth printing.
        out = llm.generate(["The capital of France is"],
                           SamplingParams(temperature=0.0, max_tokens=8))
        if rank == 0:
            print("GENERATE OK:", out[0].outputs[0].text, flush=True)
            print("2NODE_SMOKE_PASS", flush=True)
        return

    # --- mp backend: KNOWN BROKEN, see docstring. Kept for reproducibility. ---
    if args.tensor_parallel_size is None:
        raise SystemExit("--tensor-parallel-size is required for --backend mp")
    print(f"[rank {args.node_rank}/{args.nnodes}] starting, "
          f"master={args.master_addr}:{args.master_port} (mp, KNOWN BROKEN)",
          flush=True)
    t0 = time.perf_counter()
    llm = LLM(model=args.model,
              tensor_parallel_size=args.tensor_parallel_size,
              nnodes=args.nnodes, node_rank=args.node_rank,
              master_addr=args.master_addr, master_port=args.master_port,
              max_model_len=args.max_model_len,
              distributed_executor_backend="mp",
              enforce_eager=True)
    dt = time.perf_counter() - t0
    role = "DRIVER (expected)" if args.node_rank == 0 else \
        "returned on a non-zero rank (UNEXPECTED — confirm this is fine)"
    print(f"[rank {args.node_rank}] LLM() returned after {dt:.1f}s — {role}",
          flush=True)

    if args.node_rank == 0:
        out = llm.generate(["The capital of France is"],
                           SamplingParams(temperature=0.0, max_tokens=8))
        print("GENERATE OK:", out[0].outputs[0].text, flush=True)
        print("2NODE_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
