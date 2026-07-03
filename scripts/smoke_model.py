"""One-example smoke test for online-teacher LoRA KD.

This is intentionally a fast preflight for expensive queue entries. It checks
that a model can be loaded through the current bf16 HF path, its tokenizer can
adapt the Machado RAG records, LoRA can attach to the expected projection
modules, and one teacher/student KD step fits under the requested VRAM cap.
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
from selfupdate.train.lora import attach_lora
from selfupdate.train.losses import answer_ce, kd_topk_kl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--max-vram-gb", type=float, default=72.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.experiment)
    if cfg.train.method != "kd" or not cfg.train.online_teacher or not cfg.train.lora.enabled:
        sys.exit("smoke_model expects online-teacher LoRA KD")

    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    peft_model = attach_lora(model, cfg.train.lora)
    base = peft_model.get_base_model()
    base.model.embed_tokens.requires_grad_(False)
    base.model.norm.requires_grad_(False)
    base.lm_head.requires_grad_(False)
    base.train()

    ds = DistillDataset(
        cfg.data.examples_path, None, tok,
        need_logits=False,
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
        with_teacher_ids=True,
    )
    it = ds[0]
    device = cfg.model.device
    with torch.no_grad(), peft_model.disable_adapter(), torch.autocast(device, dtype=torch.bfloat16):
        t_h = base.model(input_ids=it.teacher_ids.to(device)[None], use_cache=False).last_hidden_state[0]
        t_logits = base.lm_head(t_h[it.t0: it.t0 + it.A - 1]).float()
    logz = torch.logsumexp(t_logits, -1)
    topk_v, topk_i = t_logits.topk(cfg.cache.topk, -1)

    ids = it.student_ids.to(device)
    pos = it.position_ids.to(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        h = base.model(input_ids=ids[None], position_ids=pos[None], use_cache=False).last_hidden_state[0]
        logits = base.lm_head(h[it.s0: it.s0 + it.A - 1])
        loss = kd_topk_kl(logits, topk_v, topk_i, logz, T=cfg.train.kd_temperature)
        if cfg.train.answer_ce_weight > 0:
            gold = ids[it.ans0: it.s0 + it.A]
            ce_logits = logits[it.ans0 - 1 - it.s0:]
            loss = loss + cfg.train.answer_ce_weight * answer_ce(ce_logits, gold)
    loss.backward()
    vram_gb = torch.cuda.max_memory_allocated() / 2**30
    result = {
        "model": cfg.model.name,
        "run_name": cfg.run_name,
        "loss": float(loss.detach().cpu()),
        "vram_gb": round(vram_gb, 2),
        "ok": vram_gb <= args.max_vram_gb,
    }
    if not result["ok"]:
        print(json.dumps(result, indent=1))
        sys.exit(f"peak VRAM {vram_gb:.2f} GB exceeds cap {args.max_vram_gb:.2f} GB")
    print(json.dumps(result, indent=1))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
