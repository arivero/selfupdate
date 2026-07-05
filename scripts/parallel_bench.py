"""Three-way parallelism timing: single-card vs PP2 (device_map pipeline)
vs TP2 (torch tensor parallel), on the layerwise training workload.

Workload: N items of the summed walk — per-block forward+backward with
nmse hidden loss against precomputed targets, optimizer step every 8 —
the trainer's hot loop, minus logging.

Usage:
  single / pp2 :  python scripts/parallel_bench.py --mode single|pp2 [--items 30]
  tp2          :  torchrun --nproc-per-node=2 scripts/parallel_bench.py --mode tp2

At 0.6B the EXPECTED result is TP2 losing badly (all-reduce per linear
on tiny matmuls); the measurement pins the crossover story and any
DTensor pitfalls go in the risk register (docs/windows.md / issues.md).
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM

from selfupdate.train.blocks import BlockStack
from selfupdate.train.losses import HiddenLoss


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
    args = ap.parse_args()

    model = build_model(args.mode, args.model)
    stack = BlockStack(model)
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
            targets[L] = stack.loss_view(L, t)[0].detach()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5)

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
    if args.check:
        last = [round(float(v), 6) for v in one_item()[-3:]]
        sig = 0.0
        for p_ in model.parameters():
            t_ = p_.detach()
            try:
                from torch.distributed.tensor import DTensor
                if isinstance(t_, DTensor):
                    t_ = t_.to_local()
            except Exception:
                pass
            sig += float(t_.double().sum())
        if args.mode == "tp2":
            import torch.distributed as dist
            s_ = torch.tensor([sig], device=f"cuda:{os.environ['LOCAL_RANK']}")
            gathered = [torch.zeros_like(s_) for _ in range(2)]
            dist.all_gather(gathered, s_)
            tot = float(gathered[0] + gathered[1])
            if rank == 0:
                print(f"CHECK mode=tp2 loss={args.loss}: last3={last} "
                      f"sig_total={tot:.6f} rank_sigs={[float(g) for g in gathered]}")
        else:
            print(f"CHECK mode={args.mode} loss={args.loss}: last3={last} sig_total={sig:.6f}")
    if rank == 0:
        print(f"BENCH mode={args.mode} model={args.model.split('/')[-1]} "
              f"seq={args.seqlen}: {dt:.1f} ms/item")


if __name__ == "__main__":
    main()
