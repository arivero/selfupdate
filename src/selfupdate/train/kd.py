"""Regime 1 — classical KD: KL on logits, backprop through the whole student.

Memory notes for 12 GB: fp32 weights + bf16 autocast + gradient checkpointing,
and the lm_head runs only on the aligned-span slice of the last hidden state
(151k vocab logits over a full 700-token sequence would not fit).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig
from ..data.dataset import DistillDataset, collate_items
from ..eval.recite import recite_eval
from ..teacher.cache import TeacherCache, resolve_cache_dir
from ..utils.runlog import setup_run_dir
from ..utils.seeding import seed_everything
from .losses import answer_ce, kd_topk_kl


def train_kd(cfg: ExperimentConfig) -> Path:
    run_dir, log = setup_run_dir(cfg)
    seed_everything(cfg.train.seed)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    # LoRA: the base is frozen, so fp32 master precision buys nothing — bf16
    # halves its footprint (adapter params stay fp32 via peft).
    base_dtype = torch.bfloat16 if cfg.train.lora.enabled else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=base_dtype)
    model.to(cfg.model.device)
    peft_model = None
    if cfg.train.lora.enabled:
        from .lora import attach_lora

        peft_model = attach_lora(model, cfg.train.lora)
        model = peft_model.get_base_model()  # adapters live inside the modules
    if cfg.train.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # Train transformer blocks only: embedding/lm_head/final-norm deltas would
    # confound the layer-localization question, and freezing them keeps
    # full-FT KD comparable with the layerwise regime (blocks only by
    # construction). It also makes fp32 AdamW fit in 12 GB (0.31B frozen).
    model.model.embed_tokens.requires_grad_(False)
    model.model.norm.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.train()

    online = cfg.train.online_teacher
    if online and peft_model is None:
        raise ValueError("train.online_teacher requires train.lora.enabled "
                         "(the resident base weights ARE the teacher)")
    cache = None
    if not online:
        cache_root, chash = resolve_cache_dir(cfg)
        cache = TeacherCache(cache_root, expect_hash=chash)
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=[], need_logits=not online,
        rebase_gap=(cfg.mask.compaction == "stub_gap"),
        with_teacher_ids=online,
    )
    records = ds.records  # same parsed jsonl the training pairs came from
    loader = DataLoader(
        ds, batch_size=cfg.train.micro_batch, shuffle=True,
        collate_fn=collate_items, num_workers=0,
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.train.lr)
    log.log(kind="setup", trainable_params=sum(p.numel() for p in trainable),
            total_params=sum(p.numel() for p in model.parameters()))
    device = cfg.model.device
    step = accum = 0
    stop = False
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        for items in loader:
            if stop:
                break
            for it in items:
                ids = it.student_ids.to(device)
                pos = it.position_ids.to(device)
                if online:
                    # teacher = the same resident model with adapters off
                    with torch.no_grad(), peft_model.disable_adapter(), \
                            torch.autocast(device, dtype=torch.bfloat16):
                        t_h = model.model(
                            input_ids=it.teacher_ids.to(device)[None], use_cache=False
                        ).last_hidden_state[0]
                        t_logits = model.lm_head(t_h[it.t0: it.t0 + it.A - 1]).float()
                    logz = torch.logsumexp(t_logits, -1)
                    topk_v, topk_i = t_logits.topk(cfg.cache.topk, -1)
                else:
                    topk_v = it.topk_v[:-1].to(device)
                    topk_i = it.topk_i[:-1].to(device)
                    logz = it.logz[:-1].to(device)
                with torch.autocast(device, dtype=torch.bfloat16):
                    h = model.model(
                        input_ids=ids[None], position_ids=pos[None], use_cache=False
                    ).last_hidden_state[0]
                    span = h[it.s0: it.s0 + it.A - 1]
                    logits = model.lm_head(span)
                    loss = kd_topk_kl(
                        logits, topk_v, topk_i, logz, T=cfg.train.kd_temperature
                    )
                    if cfg.train.answer_ce_weight > 0:
                        # answer tokens only — the shared_mid template tokens
                        # are constant across examples and would dilute the CE
                        gold = ids[it.ans0: it.s0 + it.A]
                        ce_logits = logits[it.ans0 - 1 - it.s0:]
                        loss = loss + cfg.train.answer_ce_weight * answer_ce(ce_logits, gold)
                (loss / cfg.train.grad_accum).backward()
                accum += 1
                log.log(kind="train", epoch=epoch, step=step, loss=loss.item())
                if accum % cfg.train.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if cfg.train.max_steps and step >= cfg.train.max_steps:
                        stop = True
                        break

        if (epoch + 1) % cfg.eval.every_epochs == 0 or epoch == cfg.train.epochs - 1 or stop:
            r = recite_eval(model, tok, records, limit=8,
                             rebase_gap=(cfg.mask.compaction == "stub_gap"))
            log.log(kind="eval", epoch=epoch, cer=r["cer"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"epoch {epoch}: eval CER {r['cer']:.3f} line-exact {r['line_exact']:.3f}")
        if stop:
            break

    if peft_model is not None:
        peft_model.save_pretrained(run_dir / "checkpoint")
    else:
        model.to(torch.bfloat16)
        model.save_pretrained(run_dir / "checkpoint")
    tok.save_pretrained(run_dir / "checkpoint")
    log.log(kind="done", vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            minutes=round((time.time() - t0) / 60, 1))
    log.close()
    return run_dir
