"""Signal attribution: what fraction of the training gradient comes from
the per-layer hidden losses vs teacher-sourced behavioral readout?

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
from selfupdate.train.losses import HiddenLoss


def load_student(cfg, checkpoint: str, dev: str):
    ckpt = Path(checkpoint)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16).to(dev)
        peft_model = PeftModel.from_pretrained(
            base, checkpoint, is_trainable=True)
        model = peft_model.get_base_model()
        model.to(dev)
        return model, tok, peft_model
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16).to(dev)
    return model, tok, None


def grad_norm2(stack, L):
    return sum(float((p.grad ** 2).sum()) for p in stack.block_params(L)
               if p.grad is not None)


def backward_if_grad(loss, *, retain_graph=False):
    if loss is None or not loss.requires_grad:
        return False
    loss.backward(retain_graph=retain_graph)
    return True


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

    model, tok, peft_model = load_student(cfg, args.checkpoint, dev)
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    model.eval()
    if peft_model is not None:
        teacher = OnlineTeacherSource(stack, peft_model=peft_model)
    else:
        t_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16).to(dev)
        t_model.eval().requires_grad_(False)
        teacher = OnlineTeacherSource(stack, frozen_stack=BlockStack(t_model))
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head)

    ds = DistillDataset(cfg.data.examples_path, None, tok, [],
                        with_teacher_ids=True)
    n = stack.n_layers
    readout0 = (n - cfg.train.readout_window_blocks + 1
                if cfg.train.readout_window_blocks > 0 else n + 1)
    hid2 = {L: 0.0 for L in range(1, n + 1)}
    aux2 = {L: 0.0 for L in range(1, n + 1)}

    for it in [ds[i] for i in range(0, len(ds), max(1, len(ds) // args.items))][: args.items]:
        targets = teacher.aligned_targets(it, dev)
        ids = it.student_ids.to(dev)[None]
        pos = it.position_ids.to(dev)[None]
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        L = 1
        while L <= n:
            if L == readout0:
                # connected window: attribute hidden-sum vs readout jointly
                with torch.autocast(dev, dtype=torch.bfloat16):
                    hh = h.detach()
                    losses = []
                    for LL in range(readout0, n + 1):
                        hh = stack.run_block(LL, hh, pos_emb)
                        losses.append(loss_fn(
                            stack.loss_view(LL, hh)[0, it.s0: it.s0 + it.A],
                            targets[LL], normed=(LL == n)))
                    logits = stack.lm_head(stack.final_norm(hh)[
                        0, it.ans0 - 1: it.s0 + it.A - 1])
                    if cfg.train.readout_source != "teacher_kl":
                        raise ValueError(
                            f"unknown readout_source {cfg.train.readout_source!r}; "
                            "only teacher_kl is allowed"
                        )
                    with torch.no_grad():
                        t_logits = stack.lm_head(
                            targets[n][it.ans0 - it.s0 - 1: it.A - 1]
                            .to(logits.dtype)
                        )
                    readout = cfg.train.readout_weight * torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(logits.float(), dim=-1),
                        torch.nn.functional.log_softmax(t_logits.float(), dim=-1),
                        log_target=True, reduction="batchmean",
                    )
                    hid = cfg.train.window_hidden_weight * sum(losses)
                model.zero_grad(set_to_none=True)
                if backward_if_grad(hid, retain_graph=True):
                    for LL in range(readout0, n + 1):
                        hid2[LL] += grad_norm2(stack, LL)
                model.zero_grad(set_to_none=True)
                if backward_if_grad(readout):
                    for LL in range(readout0, n + 1):
                        aux2[LL] += grad_norm2(stack, LL)
                break
            with torch.autocast(dev, dtype=torch.bfloat16):
                h_out = stack.run_block(L, h.detach(), pos_emb)
                hid = loss_fn(stack.loss_view(L, h_out)[0, it.s0: it.s0 + it.A],
                              targets[L], normed=(L == n))
                aux = None
            model.zero_grad(set_to_none=True)
            if backward_if_grad(hid, retain_graph=aux is not None):
                hid2[L] += grad_norm2(stack, L)
            if aux is not None:
                model.zero_grad(set_to_none=True)
                if backward_if_grad(aux):
                    aux2[L] += grad_norm2(stack, L)
            h = h_out.detach()
            L += 1

    tot_h = sum(hid2.values()) ** 0.5
    tot_a = sum(aux2.values()) ** 0.5
    share = tot_h / max(tot_h + tot_a, 1e-12)
    per_block = {L: {"hidden_gn": hid2[L] ** 0.5, "aux_gn": aux2[L] ** 0.5}
                 for L in range(1, n + 1)}
    out = {"run": cfg.run_name, "items": args.items,
           "run_class": cfg.train.run_class,
           "readout_source": cfg.train.readout_source,
           "readout_window_blocks": cfg.train.readout_window_blocks,
           "readout_weight": cfg.train.readout_weight,
           "window_hidden_weight": cfg.train.window_hidden_weight,
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
