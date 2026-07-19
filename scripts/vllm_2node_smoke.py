"""2-node vLLM smoke test (owner 2026-07-19): does our vLLM 0.25 build serve
a model split across agpuh01+agpuh02 at all, before attempting DeepSeek-V4-
Flash bf16 (543 GB) or Qwen3.5-397B-FP8?

vLLM 0.25 multi-node is NATIVE (no Ray): ``EngineArgs`` exposes
``nnodes``/``node_rank``/``master_addr``/``master_port`` directly, and with
``nnodes > 1`` the backend defaults to ``mp`` (confirmed by reading
vllm/config/parallel.py — ray is NOT installed in this env and is not
needed). ``MultiprocExecutor.__init__`` branches on
``node_rank_within_dp == 0`` — i.e. the SAME script runs symmetrically on
BOTH nodes, differing only in ``--node-rank``; rank 0 is the driver that
returns from ``LLM(...)`` usable for ``.generate()``, other ranks block
inside the constructor as workers (to be confirmed empirically here).

Binds to the 10 GbE control IPs (172.21.5.1 / .2) — IPoIB is broken
node-to-node per the fabric notes (2026-07-18); NCCL data plane still uses
IB verbs directly once the process group is up.

Usage (run on BOTH nodes at the same time, only rank 0 generates/prints):
  # agpuh01:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python scripts/vllm_2node_smoke.py \
    --node-rank 0 --nnodes 2 --master-addr 172.21.5.1
  # agpuh02:
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python scripts/vllm_2node_smoke.py \
    --node-rank 1 --nnodes 2 --master-addr 172.21.5.1
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
    ap.add_argument("--node-rank", type=int, required=True)
    ap.add_argument("--nnodes", type=int, required=True)
    ap.add_argument("--master-addr", required=True)
    ap.add_argument("--master-port", type=int, default=29501)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="GPUs PER NODE; world_size = this * nnodes")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    print(f"[rank {args.node_rank}/{args.nnodes}] starting, "
          f"master={args.master_addr}:{args.master_port}", flush=True)
    from vllm import LLM, SamplingParams
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
