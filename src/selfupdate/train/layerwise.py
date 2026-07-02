"""Regime 2 — layer-wise hidden-state matching with local backprop.

Because teacher and student share architecture AND initial weights, block L of
the student can be trained directly against the cached teacher ``h{L}`` at
aligned positions. Activations are detached both entering and leaving each
block, so every ``.backward()`` is local to one block — peak activation memory
is a single block's graph.

Schedules (registry; new variants = one new class):

- ``summed``     every block gets its local loss on every item, all blocks
                 update each optimizer step.
- ``sequential`` block L trains to plateau while blocks < L stay frozen with
                 their outputs precomputed into an activation cache; blocks
                 <= L never run again in later stages. This is the contract
                 that streams one 120B block at a time.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig
from ..data.dataset import DistillDataset, collate_items, load_jsonl
from ..eval.recite import recite_eval
from ..teacher.cache import TeacherCache, resolve_cache_dir
from ..utils.runlog import RunLog
from ..utils.seeding import seed_everything
from .blocks import BlockStack
from .losses import hidden_match


def local_block_step(stack, L, h_in, pos_emb, target, s0, A, kind, autocast=True):
    """One local forward+backward for block L. ``h_in`` must be detached, so
    the recorded graph — and therefore the backward — is confined to block L:
    no gradient from this loss can reach any other block, the lm_head, or the
    logits. Returns (loss value, detached block output). Autocast wraps only
    the forward+loss; backward runs outside it."""
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(L, h_in, pos_emb)
        loss = hidden_match(stack.loss_view(L, h_out)[0, s0: s0 + A], target, kind)
    loss.backward()
    return loss.item(), h_out.detach()


def _run_dir_setup(cfg) -> tuple[Path, RunLog]:
    run_dir = Path("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), allow_unicode=True)
    )
    return run_dir, RunLog(run_dir)


def train_layerwise(cfg: ExperimentConfig) -> Path:
    run_dir, log = _run_dir_setup(cfg)
    seed_everything(cfg.train.seed)
    device = cfg.model.device

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model.to(device)
    peft_model = None
    if cfg.train.lora.enabled:
        from .lora import attach_lora

        peft_model = attach_lora(model, cfg.train.lora)
        model = peft_model.get_base_model()
    model.train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()

    online = cfg.train.online_teacher
    if online and peft_model is None:
        raise ValueError("train.online_teacher requires train.lora.enabled")
    cache = None
    if not online:
        cache_root, chash = resolve_cache_dir(cfg)
        cache = TeacherCache(cache_root, expect_hash=chash)
    records = load_jsonl(cfg.data.examples_path)

    if cfg.train.schedule == "summed":
        _train_summed(cfg, stack, cache, tok, records, log, peft_model)
    elif cfg.train.schedule == "sequential":
        if online:
            raise NotImplementedError(
                "online teacher for the sequential schedule is a planned "
                "extension (lockstep teacher activation cache); use summed or "
                "a prebuilt cache"
            )
        _train_sequential(cfg, stack, cache, tok, records, log)
    else:
        raise ValueError(f"unknown layerwise schedule {cfg.train.schedule!r}")

    if peft_model is not None:
        peft_model.save_pretrained(run_dir / "checkpoint")
    else:
        model.to(torch.bfloat16)
        model.save_pretrained(run_dir / "checkpoint")
    tok.save_pretrained(run_dir / "checkpoint")
    log.log(kind="done", vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    log.close()
    return run_dir


def _make_dataset(cfg, cache, tok, layers, with_teacher_ids=False):
    return DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=layers, need_logits=False,
        rebase_gap=(cfg.mask.compaction == "stub_gap"),
        with_teacher_ids=with_teacher_ids,
    )


def _loader(cfg, ds):
    return DataLoader(
        ds, batch_size=cfg.train.micro_batch, shuffle=True,
        collate_fn=collate_items, num_workers=0,
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )


def _online_targets(stack, peft_model, it, device):
    """Teacher targets computed from the resident base weights (adapters off):
    advance the teacher input through all blocks once, collect the aligned
    slice per layer. Returns {L: [A, H]} like the disk cache would."""
    t_ids = it.teacher_ids.to(device)[None]
    t_pos = torch.arange(t_ids.shape[1], device=device)[None]
    targets = {}
    with torch.no_grad(), peft_model.disable_adapter(), \
            torch.autocast(device, dtype=torch.bfloat16):
        h = stack.embed(t_ids)
        pos_emb = stack.rope(h, t_pos)
        for L in range(1, stack.n_layers + 1):
            h = stack.run_block(L, h, pos_emb)
            targets[L] = stack.loss_view(L, h)[0, it.t0: it.t0 + it.A].detach()
    return targets


def _train_summed(cfg, stack, cache, tok, records, log, peft_model=None):
    device = cfg.model.device
    n = stack.n_layers
    online = cfg.train.online_teacher
    ds = _make_dataset(cfg, cache, tok,
                       [] if online else list(range(1, n + 1)),
                       with_teacher_ids=online)
    loader = _loader(cfg, ds)
    opts = {
        L: torch.optim.AdamW(
            [p for p in stack.block_params(L) if p.requires_grad], lr=cfg.train.lr
        )
        for L in range(1, n + 1)
    }

    step = accum = 0
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        for items in loader:
            for it in items:
                ids = it.student_ids.to(device)[None]
                pos = it.position_ids.to(device)[None]
                targets = _online_targets(stack, peft_model, it, device) if online else None
                h = stack.embed(ids)
                pos_emb = stack.rope(h, pos)
                layer_losses = []
                for L in range(1, n + 1):
                    target = targets[L] if online else it.hidden[L].to(device)
                    loss_val, h = local_block_step(
                        stack, L, h.detach(), pos_emb, target,
                        it.s0, it.A, cfg.train.hidden_loss,
                    )
                    layer_losses.append(loss_val)
                accum += 1
                log.log(kind="train", epoch=epoch, step=step,
                        loss=sum(layer_losses) / n, per_layer=layer_losses)
                if accum % cfg.train.grad_accum == 0:
                    for L, opt in opts.items():
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    step += 1
        if (epoch + 1) % cfg.eval.every_epochs == 0 or epoch == cfg.train.epochs - 1:
            r = recite_eval(stack.model, tok, records, limit=8)
            log.log(kind="eval", epoch=epoch, cer=r["cer"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"epoch {epoch}: eval CER {r['cer']:.3f} line-exact {r['line_exact']:.3f}")


class StudentActCache:
    """Full-sequence layer-L outputs of the frozen student prefix, kept on CPU
    (fp16). Must be full-sequence: attention in block L+1 mixes all positions,
    not just the aligned span."""

    def __init__(self):
        self._data: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def advance(self, stack, L, ds, device):
        """Advance the cache from h_{L-1} to h_L by running block L only —
        the one-block-at-a-time streaming contract (block 1 starts from the
        embeddings). fp16 re-quantization per stage adds bounded per-stage
        rounding, comparable to the bf16 autocast noise already present."""
        for idx in range(len(ds)):
            it = ds[idx]
            pos = it.position_ids.to(device)[None]
            if L == 1:
                h = stack.embed(it.student_ids.to(device)[None])
            else:
                h = self._data[it.example_id].to(device, torch.float32)[None]
            with torch.autocast(device, dtype=torch.bfloat16):
                pos_emb = stack.rope(h, pos)
                h = stack.run_block(L, h, pos_emb)
            self._data[it.example_id] = h[0].to(torch.float16).cpu()

    def get(self, example_id: str) -> torch.Tensor:
        return self._data[example_id]


def _train_sequential(cfg, stack, cache, tok, records, log):
    device = cfg.model.device
    n = stack.n_layers
    act_cache = StudentActCache()
    t0 = time.time()

    for L in range(1, n + 1):
        ds = _make_dataset(cfg, cache, tok, [L])
        loader = _loader(cfg, ds)
        opt = torch.optim.AdamW(
            [p for p in stack.block_params(L) if p.requires_grad], lr=cfg.train.lr
        )
        best = float("inf")
        stall = 0
        steps = accum = 0
        done = False
        epoch = 0
        while not done:
            epoch_losses = []
            for items in loader:
                if done:
                    break
                for it in items:
                    pos = it.position_ids.to(device)[None]
                    if L == 1:
                        h_in = stack.embed(it.student_ids.to(device)[None])
                    else:
                        h_in = act_cache.get(it.example_id).to(device, torch.float32)[None]
                    pos_emb = stack.rope(h_in, pos)
                    target = it.hidden[L].to(device)
                    loss_val, _ = local_block_step(
                        stack, L, h_in.detach(), pos_emb, target,
                        it.s0, it.A, cfg.train.hidden_loss,
                    )
                    epoch_losses.append(loss_val)
                    accum += 1
                    if accum % cfg.train.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        steps += 1
                        if steps >= cfg.train.stage_max_steps:
                            done = True
                            break
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            log.log(kind="stage", layer=L, epoch=epoch, loss=mean_loss, steps=steps)
            if mean_loss < best * 0.99:
                best, stall = mean_loss, 0
            else:
                stall += 1
                if stall >= cfg.train.plateau_patience:
                    done = True
            epoch += 1
        print(f"layer {L}: {steps} steps, final loss {mean_loss:.5f}")

        if L < n:
            act_cache.advance(stack, L, ds, device)
        if L % 7 == 0 or L == n:
            r = recite_eval(stack.model, tok, records, limit=8)
            log.log(kind="eval", layer=L, cer=r["cer"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"after layer {L}: eval CER {r['cer']:.3f}")
