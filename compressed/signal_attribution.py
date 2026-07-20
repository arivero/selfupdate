"""Certify and quantify strictly block-local training gradients.

For each sampled item and layer this reproduces the configured hidden-state
loss, backpropagates it in isolation, and records the intended block norm,
largest foreign-block norm, and frozen-vocabulary leakage.  The branch has no
behavioral readout or final-logit training component.
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import sys
from pathlib import Path


from selfupdate.utils.env import cap_cpu_threads

cap_cpu_threads()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.teacher.cache import TeacherCache, resolve_cache_dir
from selfupdate.train.blocks import BlockStack
from selfupdate.train.locality import certify_locality_resident


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)
    cfg.model.device = args.device

    model, tok = load_student(cfg, args.checkpoint, args.device)
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    cache_root, cache_hash = resolve_cache_dir(cfg)
    cache = TeacherCache(cache_root, expect_hash=cache_hash)
    out_dir = Path(args.checkpoint).parent / "eval"
    out = certify_locality_resident(
        cfg, stack, tok, cache, out_dir.parent, items=args.items)
    print(f"{cfg.run_name}: locality=PASS "
          f"(local {out['local_grad_norm']:.3g}, "
          f"cross-block {out['cross_block_leak_grad_norm']:.3g}, "
          f"frozen-vocab {out['frozen_vocab_grad_norm']:.3g})")


if __name__ == "__main__":
    main()
