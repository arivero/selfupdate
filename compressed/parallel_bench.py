"""Three-way parallelism timing: single-card vs PP2 (device_map pipeline)
vs TP2 (torch tensor parallel), on the layerwise training workload.

Workload: N items of the summed walk — per-block forward+backward with
nmse hidden loss against precomputed targets, optimizer step every 8 —
the trainer's hot loop, minus logging.

Usage:
  single / pp2 :  python compressed/parallel_bench.py --mode single|pp2 [--items 30]
  tp2          :  torchrun --nproc-per-node=2 compressed/parallel_bench.py --mode tp2

At 0.6B the EXPECTED result is TP2 losing badly (all-reduce per linear
on tiny matmuls); the measurement pins the crossover story and any
DTensor pitfalls go in the risk register (docs/windows.md / issues.md).
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import json
import os
import sys
import time
from pathlib import Path


import torch
from transformers import AutoModelForCausalLM

from selfupdate.train.blocks import BlockStack
from selfupdate.train.losses import HiddenLoss


def _local_tensor(tensor):
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(tensor, DTensor):
            return tensor.to_local()
    except Exception:
        pass
    return tensor


def _parameter_sample(tensor, limit=1024):
    """Bounded CPU sample for update certification without cloning weights."""
    flat = _local_tensor(tensor.detach()).reshape(-1)
    if flat.numel() > limit:
        index = torch.linspace(0, flat.numel() - 1, limit,
                               device=flat.device).long()
        flat = flat[index]
    return flat.float().cpu().clone()


def build_model(mode, name):
    if mode == "single":
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        return m.to("cuda")
    if mode == "pp2":
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(name).num_hidden_layers
        split = n // 2
        tied = getattr(AutoConfig.from_pretrained(name), "tie_word_embeddings", False)
        vocab_dev = 0 if tied else 1
        dm = {"model.embed_tokens": 0, "model.rotary_emb": 0,
              "model.norm": vocab_dev, "lm_head": vocab_dev}
        for i in range(n):
            dm[f"model.layers.{i}"] = 0 if i < split else 1
        return AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                    device_map=dm)
    if mode == "tp2":
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                 tp_plan="auto")
        return m
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single", "pp2", "tp2"])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--seqlen", type=int, default=600)
    ap.add_argument("--loss", default="nmse", choices=["nmse", "vocab_mse"])
    ap.add_argument("--check", action="store_true",
                    help="correctness probe: fixed seed, report loss curve + weight signature (+ cross-rank sync under tp2)")
    ap.add_argument("--out", default=None,
                    help="write machine-readable timing/correctness JSON (rank 0)")
    ap.add_argument("--reference", default=None,
                    help="single-mode JSON to compare against; exits nonzero on drift")
    ap.add_argument("--loss-rtol", type=float, default=5e-3)
    ap.add_argument("--update-rtol", type=float, default=5e-2)
    args = ap.parse_args()

    model = build_model(args.mode, args.model)
    # match the trainer: explicit pipeline placement walks hook-free
    stack = BlockStack(model, hook_free_walk=args.mode == "pp2")
    stack.freeze_non_blocks()
    torch.manual_seed(1234)
    loss_fn = (HiddenLoss("vocab_mse", stack.final_norm, stack.lm_head)
               if args.loss == "vocab_mse" else HiddenLoss("nmse"))
    dev = "cuda:0" if args.mode != "tp2" else f"cuda:{os.environ['LOCAL_RANK']}"

    ids = torch.randint(10, 1000, (1, args.seqlen), device=dev)
    pos = torch.arange(args.seqlen, device=dev)[None]
    h0 = stack.embed(ids)
    pe = stack.rope(h0, pos)
    with torch.no_grad():
        t = h0
        targets = {}
        for L in range(1, stack.n_layers + 1):
            t = stack.run_block(L, t, pe)
            target = stack.loss_view(L, t)[0].detach()
            # A self-generated target is exactly equal to the initial student
            # and produces no update. Add a deterministic, scale-aware offset
            # so the probe exercises backward, optimizer state, and parameter
            # synchronization in every mode.
            g = torch.Generator(device=target.device).manual_seed(10_000 + L)
            noise = torch.randn(target.shape, generator=g, device=target.device,
                                dtype=target.dtype)
            targets[L] = (target + 0.01 * target.float().std().to(target.dtype) * noise).to(
                dtype=torch.bfloat16, device="cpu")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5,
        foreach=args.mode != "tp2")
    initial = {name: _parameter_sample(p) for name, p in model.named_parameters()
               if p.requires_grad}

    def one_item():
        h = h0
        vals = []
        for L in range(1, stack.n_layers + 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = stack.run_block(L, h.detach(), pe)
                tgt = targets[L]
                lv = stack.loss_view(L, out)[0]
                loss = loss_fn(lv, tgt.to(lv.device), normed=(L == stack.n_layers))
            loss.backward()
            vals.append(loss.detach())
            h = out.detach()
        return vals

    for _ in range(3):
        one_item()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.items):
        one_item()
        if (i + 1) % 8 == 0:
            opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.items * 1000
    rank = int(os.environ.get("RANK", 0))
    result = {
        "mode": args.mode, "model": args.model, "items": args.items,
        "seqlen": args.seqlen, "loss": args.loss, "ms_per_item": dt,
    }
    if args.check or args.reference:
        last = [float(v) for v in one_item()[-3:]]
        update_sq = 0.0
        for name, p_ in model.named_parameters():
            if name not in initial:
                continue
            delta = _parameter_sample(p_) - initial[name]
            update_sq += float(delta.double().square().sum())
        if args.mode == "tp2":
            import torch.distributed as dist
            metric = torch.tensor([update_sq], device=f"cuda:{os.environ['LOCAL_RANK']}")
            dist.all_reduce(metric)
            update_sq = float(metric)
        result.update(last3_losses=last, sampled_update_l2=update_sq ** 0.5)
        if rank == 0:
            print(f"CHECK mode={args.mode} loss={args.loss}: last3={last} "
                  f"sampled_update_l2={result['sampled_update_l2']:.8f}")
    if rank == 0 and args.reference:
        reference = json.loads(Path(args.reference).read_text())
        for got, expected in zip(result["last3_losses"], reference["last3_losses"]):
            if not torch.isclose(torch.tensor(got), torch.tensor(expected),
                                 rtol=args.loss_rtol, atol=1e-7):
                raise SystemExit(f"loss drift: {got} vs reference {expected}")
        if not torch.isclose(torch.tensor(result["sampled_update_l2"]),
                             torch.tensor(reference["sampled_update_l2"]),
                             rtol=args.update_rtol, atol=1e-7):
            raise SystemExit("parameter-update drift: "
                             f"{result['sampled_update_l2']} vs reference "
                             f"{reference['sampled_update_l2']}")
        result["reference"] = str(args.reference)
        result["certified"] = True
    if rank == 0:
        print(f"BENCH mode={args.mode} model={args.model.split('/')[-1]} "
              f"seq={args.seqlen}: {dt:.1f} ms/item")
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
