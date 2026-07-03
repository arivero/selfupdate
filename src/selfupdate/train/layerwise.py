"""Regime 2 — layer-wise hidden-state matching with local backprop.

Because teacher and student share architecture AND initial weights, block L of
the student can be trained directly against the cached teacher ``h{L}`` at
aligned positions. Activations are detached both entering and leaving each
block, so every ``.backward()`` is local to one block — peak activation memory
is a single block's graph.

Schedules (registry; new variants = one new class):

- ``summed``     student-stream inputs: block L consumes the student's own
                 h_{L-1} (detached); every block gets its local loss on every
                 item. Inputs drift as shallow blocks train.
- ``sequential`` block L trains to plateau while blocks < L stay frozen with
                 their outputs precomputed into an activation cache; blocks
                 <= L never run again in later stages. This is the contract
                 that streams one 120B block at a time.
- ``teacher_censored`` teacher-stream inputs: block L consumes the TEACHER's
                 h_{L-1} with the privileged rows deleted (censored own
                 attention, teacher position ids kept so the RoPE gap is
                 preserved). Teacher h_{L-1} at answer positions already
                 carries the context influence of layers 1..L-1, so each block
                 learns only its own layer's increment of the context effect.
                 Inputs are stationary and every layer is independent —
                 embarrassingly parallel across GPUs. Requires the online
                 teacher (LoRA) and compaction=remove.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig
from ..data.dataset import DistillDataset, collate_items
from ..eval.general import general_ce
from ..eval.recite import recite_eval
from ..teacher.cache import TeacherCache, resolve_cache_dir
from ..utils.runlog import setup_run_dir
from ..utils.seeding import seed_everything
from .blocks import BlockStack
from .losses import answer_ce, hidden_match, lens_kl


def local_block_step(stack, L, h_in, pos_emb, target, s0, A, kind, autocast=True,
                     lens_ce_w=0.0, gold=None, ans_off=None, lens_kl_w=0.0):
    """One local forward+backward for block L. ``h_in`` must be detached, so
    the recorded graph — and therefore the backward — is confined to block L:
    no gradient from this loss can reach any other block, the lm_head, or the
    logits. Returns (loss value, detached block output). Autocast wraps only
    the forward+loss; backward runs outside it.

    ``lens_ce_w > 0`` adds a per-block behavioral auxiliary: block L's output
    is decoded through the frozen final norm + lm_head (the logit lens) and
    CE'd against the gold answer — Belilovsky-style local auxiliary heads,
    for free. The head is frozen and ``h_in`` detached, so locality is
    untouched: only block L's params see this gradient."""
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(L, h_in, pos_emb)
        loss = hidden_match(stack.loss_view(L, h_out)[0, s0: s0 + A], target, kind)
        if lens_ce_w > 0 or lens_kl_w > 0:
            s_lens = stack.lm_head(
                stack.final_norm(h_out)[0, s0 + ans_off - 1: s0 + A - 1])
            if lens_ce_w > 0:
                loss = loss + lens_ce_w * answer_ce(s_lens, gold)
            if lens_kl_w > 0:
                # teacher's layer-L lens distribution as target (target is
                # the aligned-span teacher hidden, raw for L<n by the cache
                # convention — norm it the same way as the student side)
                with torch.no_grad():
                    t_lens = stack.lm_head(
                        stack.final_norm(target[ans_off - 1: A - 1]))
                loss = loss + lens_kl_w * lens_kl(s_lens, t_lens)
    loss.backward()
    return loss.item(), h_out.detach()


def last_block_step(stack, h_in, pos_emb, target, s0, A, ans_off, gold, kind,
                    ce_w, autocast=True):
    """Block n's local step with the optional gold-CE hybrid: logits go
    through the frozen final norm + lm_head, but the graph is rooted at the
    detached ``h_in``, so the backward still touches only block n's params."""
    n = stack.n_layers
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(n, h_in, pos_emb)
        normed = stack.final_norm(h_out)
        loss = hidden_match(normed[0, s0: s0 + A], target, kind)
        if ce_w > 0:
            logits = stack.lm_head(normed[0, s0 + ans_off - 1: s0 + A - 1])
            loss = loss + ce_w * answer_ce(logits, gold)
    loss.backward()
    return loss.item(), h_out.detach()


def tail_step(stack, L0, h_in, pos_emb, targets, s0, A, ans_off, gold, kind,
              ce_w, autocast=True):
    """Joint step for the tail window [L0..n]: blocks are CONNECTED, so the
    answer-CE at the top can assign credit across the final blocks — the
    logit-lens finding says block-local matching stores recall fine up to
    the tail, and the behavioral deficit lives in the last-mile readout.
    Per-block hidden losses are kept (storage signal). The window is rooted
    at a detached ``h_in``: no gradient reaches blocks < L0, and the frozen
    norm/head receive none. Peak graph = ``n - L0 + 1`` blocks."""
    n = stack.n_layers
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h = h_in
        losses = []
        for L in range(L0, n + 1):
            h = stack.run_block(L, h, pos_emb)
            losses.append(hidden_match(
                stack.loss_view(L, h)[0, s0: s0 + A], targets[L], kind))
        total = sum(losses)
        if ce_w > 0:
            logits = stack.lm_head(
                stack.final_norm(h)[0, s0 + ans_off - 1: s0 + A - 1])
            total = total + ce_w * answer_ce(logits, gold)
    total.backward()
    return [l.item() for l in losses], h.detach()


def train_layerwise(cfg: ExperimentConfig) -> Path:
    run_dir, log = setup_run_dir(cfg)
    seed_everything(cfg.train.seed)
    device = cfg.model.device

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    # bf16 base for LoRA (frozen weights) AND for the sequential schedule:
    # only the single active block needs fp32 master weights (cast per stage
    # in _train_sequential); summed full-FT trains all blocks every step and
    # keeps fp32 masters throughout.
    full_ft_all_blocks = not cfg.train.lora.enabled and cfg.train.schedule != "sequential"
    base_dtype = torch.float32 if full_ft_all_blocks else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=base_dtype)
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

    if cfg.train.schedule == "summed":
        _train_summed(cfg, stack, cache, tok, log, peft_model)
    elif cfg.train.schedule == "teacher_censored":
        if not online:
            raise NotImplementedError(
                "teacher_censored needs full-sequence teacher states; only the "
                "online teacher (train.lora.enabled + train.online_teacher) "
                "provides them without a full-sequence cache"
            )
        if cfg.mask.compaction != "remove":
            raise ValueError("teacher_censored assumes compaction=remove "
                             "(stub rows have no teacher counterpart)")
        _train_teacher_censored(cfg, stack, tok, log, peft_model)
    elif cfg.train.schedule == "sequential":
        if online:
            raise NotImplementedError(
                "online teacher for the sequential schedule is a planned "
                "extension (lockstep teacher activation cache); use summed or "
                "a prebuilt cache"
            )
        _train_sequential(cfg, stack, cache, tok, log)
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
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
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


def _train_teacher_censored(cfg, stack, tok, log, peft_model):
    """Schedule (b): per-block fitting on stationary teacher-stream inputs.

    One adapters-off pass per item yields the full-sequence teacher states
    t_h[0..n]. Block L (adapters on) consumes the censored rows of t_h[L-1]
    (prefix + aligned span, privileged rows deleted, teacher position ids
    kept) and matches the teacher's aligned-span t_h[L]. Blocks never see
    each other's outputs: layer independence holds by construction, so this
    is the schedule that parallelizes across GPUs at scale."""
    device = cfg.model.device
    n = stack.n_layers
    ds = _make_dataset(cfg, None, tok, [], with_teacher_ids=True)
    records = ds.records
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
                t_ids = it.teacher_ids.to(device)[None]
                n_t = t_ids.shape[1]
                t_pos = torch.arange(n_t, device=device)[None]
                # frozen teacher states, all layers, full teacher sequence
                with torch.no_grad(), peft_model.disable_adapter(), \
                        torch.autocast(device, dtype=torch.bfloat16):
                    h = stack.embed(t_ids)
                    pos_emb_full = stack.rope(h, t_pos)
                    t_states = [h]
                    for L in range(1, n + 1):
                        h = stack.run_block(L, h, pos_emb_full)
                        t_states.append(h)

                # censored view: prefix rows + aligned rows, teacher positions
                tA0 = it.t0
                rows = torch.cat([
                    torch.arange(it.s0, device=device),          # shared prefix
                    torch.arange(tA0, tA0 + it.A, device=device),  # mid+answer
                ])
                pos_c = rows[None]  # teacher absolute positions == row indices
                pos_emb_c = stack.rope(t_states[0][:, :1], pos_c)

                layer_losses = []
                for L in range(1, n + 1):
                    inp = t_states[L - 1][:, rows].detach()
                    with torch.no_grad():
                        target = t_states[L][0, tA0: tA0 + it.A]
                        if L == n:
                            target = stack.final_norm(target)
                    if L == n:
                        gold = it.student_ids.to(device)[it.ans0: it.s0 + it.A]
                        loss_val, _ = last_block_step(
                            stack, inp, pos_emb_c, target, it.s0, it.A,
                            it.ans0 - it.s0, gold, cfg.train.hidden_loss,
                            cfg.train.last_block_ce_weight,
                        )
                    else:
                        loss_val, _ = local_block_step(
                            stack, L, inp, pos_emb_c, target,
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
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"epoch {epoch}: eval CER {r['cer']:.3f} line-exact {r['line_exact']:.3f}")


def _train_summed(cfg, stack, cache, tok, log, peft_model=None):
    device = cfg.model.device
    n = stack.n_layers
    online = cfg.train.online_teacher
    ds = _make_dataset(cfg, cache, tok,
                       [] if online else list(range(1, n + 1)),
                       with_teacher_ids=online)
    records = ds.records
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
                tail0 = n - cfg.train.tail_ce_blocks + 1 if cfg.train.tail_ce_blocks > 0 else n + 1
                for L in range(1, n + 1):
                    target = targets[L] if online else it.hidden[L].to(device)
                    gold = ids[0, it.ans0: it.s0 + it.A]
                    if L == tail0:
                        tail_targets = {
                            LL: (targets[LL] if online else it.hidden[LL].to(device))
                            for LL in range(tail0, n + 1)
                        }
                        tail_losses, h = tail_step(
                            stack, tail0, h.detach(), pos_emb, tail_targets,
                            it.s0, it.A, it.ans0 - it.s0, gold,
                            cfg.train.hidden_loss, cfg.train.tail_ce_weight,
                        )
                        layer_losses.extend(tail_losses)
                        break
                    if L == n:
                        loss_val, h = last_block_step(
                            stack, h.detach(), pos_emb, target, it.s0, it.A,
                            it.ans0 - it.s0, gold, cfg.train.hidden_loss,
                            max(cfg.train.last_block_ce_weight,
                                cfg.train.lens_ce_weight
                                if L >= cfg.train.lens_ce_from else 0.0),
                        )
                    else:
                        lens_w = (cfg.train.lens_ce_weight
                                  if L >= cfg.train.lens_ce_from else 0.0)
                        kl_w = (cfg.train.lens_kl_weight
                                if L >= cfg.train.lens_kl_from else 0.0)
                        loss_val, h = local_block_step(
                            stack, L, h.detach(), pos_emb, target,
                            it.s0, it.A, cfg.train.hidden_loss,
                            lens_ce_w=lens_w, gold=gold,
                            ans_off=it.ans0 - it.s0, lens_kl_w=kl_w,
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
            r = recite_eval(stack.model, tok, records, limit=8,
                            rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")))
            log.log(kind="eval", epoch=epoch, cer=r["cer"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
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
        rounding, comparable to the bf16 autocast noise already present.

        Runs in eval mode: stochastic modules (LoRA dropout) must not bake a
        frozen noise sample into activations that all later stages train on.
        Iterates ds.pairs directly — the teacher targets ds[idx] would read
        from disk are not needed here."""
        was_training = stack.model.training
        stack.model.eval()
        for pair in ds.pairs:
            pos = torch.tensor(
                pair.student_position_ids(ds.rebase_gap), device=device
            )[None]
            if L == 1:
                ids = torch.tensor(pair.student_ids, device=device)[None]
                h = stack.embed(ids)
            else:
                h = self._data[pair.example_id].to(device, torch.float32)[None]
            with torch.autocast(device, dtype=torch.bfloat16):
                pos_emb = stack.rope(h, pos)
                h = stack.run_block(L, h, pos_emb)
            self._data[pair.example_id] = h[0].to(torch.float16).cpu()
        if was_training:
            stack.model.train()

    def get(self, example_id: str) -> torch.Tensor:
        return self._data[example_id]


def _train_sequential(cfg, stack, cache, tok, log):
    device = cfg.model.device
    n = stack.n_layers
    act_cache = StudentActCache()
    t0 = time.time()

    ds = _make_dataset(cfg, cache, tok, [1])  # pairs built once; layer swapped per stage
    records = ds.records
    full_ft = not cfg.train.lora.enabled
    for L in range(1, n + 1):
        ds.need_layers = [L]
        loader = _loader(cfg, ds)
        if full_ft:
            stack.blocks[L - 1].float()  # fp32 master for the active block only
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
                    if L == n:
                        gold = it.student_ids.to(device)[it.ans0: it.s0 + it.A]
                        loss_val, _ = last_block_step(
                            stack, h_in.detach(), pos_emb, target, it.s0, it.A,
                            it.ans0 - it.s0, gold, cfg.train.hidden_loss,
                            cfg.train.last_block_ce_weight,
                        )
                    else:
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
        if full_ft:
            stack.blocks[L - 1].to(torch.bfloat16)  # done training: back to bf16

        if L < n:
            act_cache.advance(stack, L, ds, device)
        if L % 7 == 0 or L == n:
            r = recite_eval(stack.model, tok, records, limit=8,
                            rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")))
            log.log(kind="eval", layer=L, cer=r["cer"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"after layer {L}: eval CER {r['cer']:.3f}")
