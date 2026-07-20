"""vLLM's NATIVE teacher-forced verification: symmetric speed comparator.

Owner ask (2026-07-19): "vLLM prediction of the very last token of the
answers, with all the answer as input, compared with the same input in our
PPPn architecture." This is vLLM playing OUR verifier's role: input =
prompt + answer[:-1] (every token but the last), ask for exactly ONE
generated token (max_tokens=1) — vLLM's own prefill forward predicts the
final answer position. No autoregression on either side; both engines do
ONE forward pass over (almost) the full sequence. This is the fair
counterpart to our trainer's teacher_argmax_acceptance (which scores EVERY
answer position from one such forward, at no extra wall-clock cost in our
architecture — a batched forward doesn't get cheaper by wanting fewer output
rows, so "last token only" and "every position" cost the same on our side).

Reports: wall time for the whole batch (2071 items = one full epoch), and whether vLLM's own
prefill prediction matches its OWN earlier greedy answer (a self-consistency
sanity check: it must, since greedy decoding IS iterated argmax).

Run in the vLLM env (single-node, unchanged since Phase 1):
  CUDA_VISIBLE_DEVICES=0 ../venvs/vllm025/bin/python compressed/vllm_prefill_verify.py \
    --model Qwen/Qwen3.6-27B \
    --responses runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl \
    --out runs/spec_verify/27b_vllm_prefill_verify.json   # no --limit = full 2071-item epoch

Multi-node (owner 2026-07-20, mirrors compressed/vllm_2node_smoke.py): driven by
``torchrun``, NOT by CLI nnodes/node-rank args on this script — that `mp`-
backend, manual-node-rank pattern was tried first and is CONFIRMED BROKEN
for the offline ``LLM()`` API (see vllm_2node_smoke.py's docstring for the
full root cause: the engine/orchestration layer is node-rank-unaware and
builds a redundant EngineCore on every rank, which crashes with
``AssertionError: collective_rpc should not be called on follower node``).
The WORKING mechanism is ``distributed_executor_backend="external_launcher"``
under torchrun (SPMD): every rank is a symmetric peer sharing torchrun's
process group, so there is no leader/follower EngineCore split. Run the SAME
command symmetrically on both nodes at the same time; torchrun sets
RANK/WORLD_SIZE/LOCAL_RANK, this script reads them, ALL ranks call
generate() together (they co-execute the forward — do NOT early-return on
rank!=0, that would hang the collective), and only rank 0 (RANK==0) prints
the summary and writes --out.

  # agpuh01 (4 GPUs/node x 2 nodes = world size 8):
  ../venvs/vllm025/bin/torchrun --nnodes 2 --node-rank 0 --nproc-per-node 4 \
    --master-addr 172.21.5.1 --master-port 29501 \
    compressed/vllm_prefill_verify.py --model <path> --responses <jsonl> \
    --multi-node --pipeline-parallel-size 2 --out runs/spec_verify/foo.json
  # agpuh02 (same instant, --node-rank 1, everything else identical):
  ../venvs/vllm025/bin/torchrun --nnodes 2 --node-rank 1 --nproc-per-node 4 \
    --master-addr 172.21.5.1 --master-port 29501 \
    compressed/vllm_prefill_verify.py --model <path> --responses <jsonl> \
    --multi-node --pipeline-parallel-size 2 --out runs/spec_verify/foo.json

IMPORTANT (owner 2026-07-20, empirical): plain ``--multi-node`` with a single
TP communicator spanning ALL 8 GPUs (tensor_parallel_size=WORLD_SIZE,
pipeline_parallel_size=1, i.e. local_world_size=4 GPUs/node in ONE TP group
across nodes) HANGS — reproduced 3x (once on the real 397B model, twice on
the tiny 0.6B model with only 2 GPUs/node), always stuck immediately after
"vLLM is using nccl==2.28.9" / before any NCCL INFO output even under
NCCL_DEBUG=INFO, i.e. before ncclCommInitRank. Three NCCL env toggles
(NCCL_IB_DISABLE=1, NCCL_P2P_DISABLE=1, NCCL_CUMEM_ENABLE=0) were each tried
and did NOT clear it. The FIX is ``--pipeline-parallel-size <nnodes>``: this
makes tensor_parallel_size = WORLD_SIZE / pipeline_parallel_size (GPUs per
node only), so every TP communicator is intra-node-only (the proven Phase 1
TP4 case) and every cross-node link is a 1-rank-per-node PP hop (the proven
2-way smoke case) — confirmed working at PP2xTP4 (world size 8) on the tiny
model 2026-07-20. Always pass ``--pipeline-parallel-size <nnodes>`` for any
--multi-node run with more than 1 GPU per node.

``--multi-node`` (default off) is the ONLY new flag that changes single-node
behavior; when absent this script is byte-identical to the published
single-node Phase 1 TP4 runs. Do NOT set CUDA_VISIBLE_DEVICES under
torchrun — LOCAL_RANK selects the GPU. ``--tensor-parallel-size`` is ignored
in --multi-node mode
(world size comes from torchrun's WORLD_SIZE env var instead).
``--kv-cache-dtype`` (e.g. fp8) and ``--cpu-offload-gb`` are passthroughs to
vLLM's own EngineArgs, needed for the 8-card DeepSeek-V4-Flash escalation.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--responses", required=True,
                    help="vLLM responses jsonl (prompt_token_ids + token_ids)")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole file (2071 items = one full epoch)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="single-node only; ignored under --multi-node "
                         "(world size comes from torchrun's WORLD_SIZE)")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--out", default=None)
    # multi-node (owner 2026-07-20): launched via torchrun, see docstring.
    # Default off keeps this script byte-identical to the published
    # single-node Phase 1 TP4 runs.
    ap.add_argument("--multi-node", action="store_true",
                    help="use distributed_executor_backend=external_launcher "
                         "and read RANK/WORLD_SIZE/LOCAL_RANK from the "
                         "torchrun-set environment (confirmed working "
                         "2026-07-20; the earlier mp/node-rank path is not)")
    ap.add_argument("--pipeline-parallel-size", type=int, default=1,
                    help="--multi-node only (owner 2026-07-20): set this to "
                         "nnodes for the 8-card escalation. A single TP "
                         "communicator spanning local_world_size>1 GPUs "
                         "ACROSS nodes hangs (confirmed: 3 NCCL env toggles "
                         "tried, none cleared it — see vllm_2node_smoke.py). "
                         "pipeline_parallel_size=nnodes with "
                         "tensor_parallel_size=WORLD_SIZE/nnodes keeps every "
                         "TP communicator intra-node-only (proven: Phase 1 "
                         "TP4) and makes every cross-node link a "
                         "1-rank-per-node PP hop (proven: the 2-way smoke) "
                         "— confirmed working at PP2xTP4 (world size 8) on "
                         "the tiny model 2026-07-20.")
    ap.add_argument("--kv-cache-dtype", default=None,
                    help="vLLM EngineArgs passthrough, e.g. fp8; default "
                         "None leaves vLLM's own default (auto)")
    ap.add_argument("--cpu-offload-gb", type=float, default=None,
                    help="GiB of weights offloaded to pinned CPU RAM PER "
                         "GPU, vLLM EngineArgs passthrough")
    args = ap.parse_args()

    rank = 0
    world_size = args.tensor_parallel_size
    pp_size = 1
    if args.multi_node:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        pp_size = args.pipeline_parallel_size
        # Control-plane binding: force NCCL/gloo rendezvous over the 10GbE
        # interface, never the broken IPoIB rail (see vllm_2node_smoke.py).
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "eno12419np2")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eno12419np2")

    rows = []
    with open(args.responses) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt_token_ids") and r.get("token_ids"):
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} responses from {args.responses}", flush=True)

    from vllm import LLM, SamplingParams
    t_load = time.perf_counter()
    # enforce_eager=True (owner, 2026-07-19): skips torch.compile/CUDA-graph
    # capture entirely. Measured cause of multi-minute-plus hangs on TP4 for
    # 26B/31B (MoE + dense) — one case hit vLLM's OWN internal cancellation
    # at ~356s, another ran 1150s+ with a single thread pegged at ~100% CPU
    # (real Triton/Inductor codegen, not a stuck loop, but never converged in
    # reasonable time). A completed eager-mode measurement is more valuable
    # than a compiled-mode attempt that may never finish; eager will report
    # slower per-token throughput than compiled mode would — label results
    # accordingly, this is a real methodology difference, not a bug.
    # disable_custom_all_reduce (owner, 2026-07-19): 35B DIED outright with
    # "CUDA error: an illegal memory access" inside determine_available_memory
    # (KV-cache profiling) — a FATAL instance of the same CUDASymmetricMemory
    # exception that appeared (survivably) in 26B's log earlier. Forces the
    # standard NCCL allreduce instead of vLLM's custom symmetric-memory path,
    # which is the implicated code (compilation_config showed
    # disable_custom_all_reduce=False as vLLM's own default, i.e. the fragile
    # path is opt-out, not opt-in).
    tp_size = world_size // pp_size if args.multi_node else world_size
    llm_kw = dict(model=args.model, max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 tensor_parallel_size=tp_size,
                 enforce_eager=True,
                 disable_custom_all_reduce=True)
    if pp_size > 1:
        llm_kw["pipeline_parallel_size"] = pp_size
    if args.max_num_seqs:
        llm_kw["max_num_seqs"] = args.max_num_seqs
    if args.kv_cache_dtype:
        llm_kw["kv_cache_dtype"] = args.kv_cache_dtype
    if args.cpu_offload_gb:
        llm_kw["cpu_offload_gb"] = args.cpu_offload_gb
    if args.multi_node:
        llm_kw["distributed_executor_backend"] = "external_launcher"
    llm = LLM(**llm_kw)
    load_s = time.perf_counter() - t_load
    print(f"[rank {rank}] loaded in {load_s:.1f}s", flush=True)

    # input = prompt + answer[:-1]; ask for exactly the last answer token.
    prompts, golds, drop = [], [], 0
    for r in rows:
        p_ids = list(r["prompt_token_ids"])
        a_ids = list(r["token_ids"])
        if len(a_ids) < 1:
            drop += 1
            continue
        prompts.append({"prompt_token_ids": p_ids + a_ids[:-1]})
        golds.append(a_ids[-1])
    if drop:
        print(f"dropped {drop} empty-answer rows", flush=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)
    t0 = time.perf_counter()
    # --multi-node: ALL ranks must call generate() together (external_launcher
    # ranks co-execute the forward as symmetric peers); early-returning on
    # rank!=0 here would hang the collective on the other ranks forever.
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    if args.multi_node and rank != 0:
        # Non-driver rank: participated in the collective generate() above,
        # nothing further to do (matches vllm_2node_smoke.py's convention).
        print(f"[rank {rank}] done (non-driver, no output written)", flush=True)
        return

    match = sum(int(o.outputs[0].token_ids[0] == g)
                for o, g in zip(outs, golds))
    n = len(golds)
    seq_tokens = sum(len(p["prompt_token_ids"]) for p in prompts)
    summary = {
        "model": args.model,
        "responses": args.responses,
        "enforce_eager": True,
        "items": n,
        "self_consistency_match_rate": round(match / max(n, 1), 4),
        "seconds": round(dt, 3),
        "items_per_s": round(n / max(dt, 1e-9), 2),
        "total_context_tokens": seq_tokens,
        "context_tok_per_s": round(seq_tokens / max(dt, 1e-9), 1),
        "load_seconds": round(load_s, 2),
        "note": ("ONE forward pass per item (max_tokens=1), no autoregression "
                "— the symmetric counterpart to our trainer's single "
                "teacher-forced forward. self_consistency should be ~1.0: "
                "greedy decoding IS iterated argmax, so vLLM's own prefill "
                "here must reproduce its own earlier generated last token."),
    }
    if args.multi_node:
        # Only added for multi-node runs so the single-node JSON schema
        # stays byte-identical to the published Phase 1 TP4 results.
        summary["distributed_executor_backend"] = "external_launcher"
        summary["world_size"] = world_size
        summary["tensor_parallel_size"] = tp_size
        summary["pipeline_parallel_size"] = pp_size
    if args.kv_cache_dtype:
        summary["kv_cache_dtype"] = args.kv_cache_dtype
    if args.cpu_offload_gb:
        summary["cpu_offload_gb"] = args.cpu_offload_gb
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
