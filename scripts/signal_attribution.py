"""Signal attribution: what fraction of the training gradient comes from
the per-layer hidden losses vs the behavioral auxiliaries (lens-CE /
tail-CE)?

The naming contract (CLAUDE.md) requires this number: a project called
"layerwise distillation" must show the hidden matching is the primary
signal — not a 100%-weight auxiliary wearing a costume. For each block
we backward the hidden loss and the auxiliary SEPARATELY (retain_graph)
and report per-block and aggregate gradient-norm shares.

Usage:
    signal_attribution.py --experiment configs/experiments/X.yaml \
        --checkpoint runs/X/checkpoint [--items 16] [--device cuda]
Writes runs/<run>/eval/signal_attribution.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import DistillDataset
from selfupdate.train.blocks import BlockStack
from selfupdate.train.layerwise import OnlineTeacherSource
from selfupdate.train.losses import HiddenLoss, answer_ce


def grad_norm2(stack, L):
    return sum(float((p.grad ** 2).sum()) for p in stack.block_params(L)
               if p.grad is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    dev = args.device

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint,
                                                 dtype=torch.bfloat16).to(dev)
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    t_model = AutoModelForCausalLM.from_pretrained(cfg.model.name,
                                                   dtype=torch.bfloat16).to(dev)
    t_model.eval().requires_grad_(False)
    teacher = OnlineTeacherSource(stack, frozen_stack=BlockStack(t_model))
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head)

    ds = DistillDataset(cfg.data.examples_path, None, tok, [],
                        with_teacher_ids=True)
    n = stack.n_layers
    tail0 = (n - cfg.train.tail_ce_blocks + 1
             if cfg.train.tail_ce_blocks > 0 else n + 1)
    hid2 = {L: 0.0 for L in range(1, n + 1)}
    aux2 = {L: 0.0 for L in range(1, n + 1)}

    for it in [ds[i] for i in range(0, len(ds), max(1, len(ds) // args.items))][: args.items]:
        targets = teacher.aligned_targets(it, dev)
        ids = it.student_ids.to(dev)[None]
        pos = it.position_ids.to(dev)[None]
        gold = ids[0, it.ans0: it.s0 + it.A]
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        L = 1
        while L <= n:
            if L == tail0:
                # connected window: attribute hidden-sum vs CE jointly
                with torch.autocast(dev, dtype=torch.bfloat16):
                    hh = h.detach()
                    losses = []
                    for LL in range(tail0, n + 1):
                        hh = stack.run_block(LL, hh, pos_emb)
                        losses.append(loss_fn(
                            stack.loss_view(LL, hh)[0, it.s0: it.s0 + it.A],
                            targets[LL], normed=(LL == n)))
                    logits = stack.lm_head(stack.final_norm(hh)[
                        0, it.ans0 - 1: it.s0 + it.A - 1])
                    ce = cfg.train.tail_ce_weight * answer_ce(logits, gold)
                    hid = cfg.train.tail_hidden_weight * sum(losses)
                model.zero_grad(set_to_none=True)
                hid.backward(retain_graph=True)
                for LL in range(tail0, n + 1):
                    hid2[LL] += grad_norm2(stack, LL)
                model.zero_grad(set_to_none=True)
                ce.backward()
                for LL in range(tail0, n + 1):
                    aux2[LL] += grad_norm2(stack, LL)
                break
            lens_w = (cfg.train.lens_ce_weight
                      if L >= cfg.train.lens_ce_from else 0.0)
            with torch.autocast(dev, dtype=torch.bfloat16):
                h_out = stack.run_block(L, h.detach(), pos_emb)
                hid = loss_fn(stack.loss_view(L, h_out)[0, it.s0: it.s0 + it.A],
                              targets[L], normed=(L == n))
                aux = None
                if lens_w > 0:
                    s_lens = stack.lm_head(stack.final_norm(h_out)[
                        0, it.ans0 - 1: it.s0 + it.A - 1])
                    aux = lens_w * answer_ce(s_lens, gold)
            model.zero_grad(set_to_none=True)
            hid.backward(retain_graph=aux is not None)
            hid2[L] += grad_norm2(stack, L)
            if aux is not None:
                model.zero_grad(set_to_none=True)
                aux.backward()
                aux2[L] += grad_norm2(stack, L)
            h = h_out.detach()
            L += 1

    tot_h = sum(hid2.values()) ** 0.5
    tot_a = sum(aux2.values()) ** 0.5
    share = tot_h / max(tot_h + tot_a, 1e-12)
    per_block = {L: {"hidden_gn": hid2[L] ** 0.5, "aux_gn": aux2[L] ** 0.5}
                 for L in range(1, n + 1)}
    out = {"run": cfg.run_name, "items": args.items,
           "hidden_grad_norm": tot_h, "aux_grad_norm": tot_a,
           "hidden_share": share, "per_block": per_block}
    out_dir = Path(args.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signal_attribution.json").write_text(json.dumps(out, indent=1))
    print(f"{cfg.run_name}: hidden share of total gradient norm = {share:.1%} "
          f"(hidden {tot_h:.3g} vs aux {tot_a:.3g})")
    top = sorted(per_block.items(), key=lambda kv: -(kv[1]["aux_gn"]))[:5]
    for L, v in top:
        h_, a_ = v["hidden_gn"], v["aux_gn"]
        print(f"  block {L:2d}: hidden {h_:.3g}  aux {a_:.3g}  "
              f"({h_ / max(h_ + a_, 1e-12):.0%} hidden)")


if __name__ == "__main__":
    main()
