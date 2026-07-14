"""Certify and quantify strictly block-local training gradients.

For each sampled item and layer this reproduces the configured hidden-state
loss, backpropagates it in isolation, and records the intended block norm,
largest foreign-block norm, and frozen-vocabulary leakage.  The branch has no
behavioral readout or final-logit training component.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import DistillDataset
from selfupdate.teacher.cache import TeacherCache, resolve_cache_dir
from selfupdate.train.blocks import BlockStack
from selfupdate.train.losses import HiddenLoss


def load_student(cfg, checkpoint: str, dev: str):
    ckpt = Path(checkpoint)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(checkpoint)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, dtype=torch.bfloat16).to(dev)
        peft_model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
        model = peft_model.get_base_model()
        model.to(dev)
        return model, tok
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16).to(dev)
    return model, tok


def grad_norm2(params) -> float:
    return sum(float((p.grad.float() ** 2).sum())
               for p in params if p.grad is not None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    model, tok = load_student(cfg, args.checkpoint, args.device)
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    model.eval()
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head)
    n = stack.n_layers

    cache_root, cache_hash = resolve_cache_dir(cfg)
    cache = TeacherCache(cache_root, expect_hash=cache_hash)
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok, list(range(1, n + 1)),
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        pad_random=(cfg.mask.compaction == "pad_random"),
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
    )

    local2 = {L: 0.0 for L in range(1, n + 1)}
    foreign2 = {L: 0.0 for L in range(1, n + 1)}
    frozen2 = {L: 0.0 for L in range(1, n + 1)}
    stride = max(1, len(ds) // args.items)
    sampled = [ds[i] for i in range(0, len(ds), stride)][:args.items]

    for it in sampled:
        targets = {L: value.to(args.device) for L, value in it.hidden.items()}
        ids = it.student_ids.to(args.device)[None]
        pos = it.position_ids.to(args.device)[None]
        h = stack.embed(ids)
        pos_emb = stack.rope(h, pos)
        for L in range(1, n + 1):
            h_in = h.detach()
            with torch.autocast(args.device, dtype=torch.bfloat16):
                h_out = stack.run_block(L, h_in, pos_emb)
                if loss_fn.is_delta and 1 < L < n:
                    loss = loss_fn.delta(
                        h_out[0, it.s0:it.s0 + it.A],
                        h_in[0, it.s0:it.s0 + it.A],
                        targets[L], targets[L - 1])
                else:
                    loss = loss_fn(
                        stack.loss_view(L, h_out)[0, it.s0:it.s0 + it.A],
                        targets[L], normed=(L == n), layer=L)
            model.zero_grad(set_to_none=True)
            loss.backward()
            local2[L] += grad_norm2(stack.block_params(L))
            foreign2[L] += max(
                (grad_norm2(stack.block_params(other))
                 for other in range(1, n + 1) if other != L),
                default=0.0)
            frozen2[L] += grad_norm2(
                list(stack.embed_tokens.parameters())
                + list(stack.final_norm.parameters())
                + list(stack.lm_head.parameters()))
            h = h_out.detach()

    total_local = sum(local2.values()) ** 0.5
    total_foreign = sum(foreign2.values()) ** 0.5
    total_frozen = sum(frozen2.values()) ** 0.5
    passed = total_foreign == 0.0 and total_frozen == 0.0
    per_block = {
        str(L): {
            "local_grad_norm": local2[L] ** 0.5,
            "max_foreign_grad_norm": foreign2[L] ** 0.5,
            "frozen_vocab_grad_norm": frozen2[L] ** 0.5,
        }
        for L in range(1, n + 1)
    }
    out = {
        "schema_version": 2,
        "run": cfg.run_name,
        "items": len(sampled),
        "teacher_target_source": "pipeline_v2_disk_cache",
        "teacher_cache": str(cache_root),
        "teacher_cache_hash": cache_hash,
        "run_class": cfg.train.run_class,
        "hidden_loss": cfg.train.hidden_loss,
        "gradient_contract": "strict_block_local_hidden_state",
        "final_logit_training": False,
        "local_grad_norm": total_local,
        "cross_block_leak_grad_norm": total_foreign,
        "frozen_vocab_grad_norm": total_frozen,
        "passed": passed,
        "per_block": per_block,
    }
    out_dir = Path(args.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signal_attribution.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(f"{cfg.run_name}: locality={'PASS' if passed else 'FAIL'} "
          f"(local {total_local:.3g}, cross-block {total_foreign:.3g}, "
          f"frozen-vocab {total_frozen:.3g})")
    if not passed:
        raise SystemExit("strict block-local gradient certification failed")


if __name__ == "__main__":
    main()
