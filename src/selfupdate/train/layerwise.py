"""Regime 2 — layer-wise hidden-state matching with local backprop.

Because teacher and student share architecture AND initial weights, block L of
the student can be trained directly against the cached teacher ``h{L}`` at
aligned positions. Activations are detached both entering and leaving each
block, so every ``.backward()`` is local to one block — peak activation memory
is a single block's graph.

This module holds the SCHEDULES: what order blocks train in and where their
targets come from. The layers around it each own one concern —

- ``steps.py``          block/window forward+backward primitives (detach
                        discipline lives there);
- ``runtime.py``        model/teacher placement, optimizer policy, tripwires;
- ``teacher_source.py`` per-step frozen-teacher states;
- ``validate.py``       dispatch-time knob-flow validation;
- ``telemetry.py``      loss aggregation and epoch probes;
- ``anchor.py``         the anti-intrusion anchor regularizer.

Schedules (one function per variant):

- ``summed``     student-stream inputs: block L consumes the student's own
                 h_{L-1} (detached); every block gets its local loss on every
                 item. Inputs drift as shallow blocks train.
- ``sequential`` block L trains to plateau while blocks < L stay frozen with
                 their outputs precomputed into an activation cache; blocks
                 <= L never run again in later stages. This is the contract
                 that streams one 120B block at a time.
- Connected WINDOWS (conn_window / readout_window_blocks) are gradient-isolation
  units, NOT memory management: backward exists only inside [L0..L1] and
  stops at the detached input of L0 — see docs/windows.md for the precise
  2x2 semantics (loss placement x window-input stream) before editing.
- ``teacher_censored`` teacher-stream inputs: block L consumes the TEACHER's
                 h_{L-1} with the privileged rows deleted (censored own
                 attention, teacher position ids kept so the RoPE gap is
                 preserved). Teacher h_{L-1} at answer positions already
                 carries the context influence of layers 1..L-1, so each block
                 learns only its own layer's increment of the context effect.
                 Inputs are stationary and every layer is independent —
                 embarrassingly parallel across GPUs. Requires the online
                 teacher (LoRA) and compaction=remove.
- ``mixed``      scheduled sampling between the two streams above.
"""

from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.dataset import (
    Batch,
    DistillDataset,
    LengthBucketBatchSampler,
    collate_items,
    collate_padded_items,
)
from ..eval.tasks import tasks_eval
from ..utils.runlog import setup_run_dir
from ..utils.seeding import seed_everything
from .losses import HiddenLoss
from .moe import MoEController, dequantize_overrides
from .anchor import (  # noqa: F401  (AnchorBank re-exported for scripts)
    AnchorBank,
    anchor_step,
    anchor_trajectory_step,
    make_anchor as _make_anchor,
)
from .runtime import (  # noqa: F401  (underscore names re-exported for scripts)
    OptimizerPlan,
    TrainingRuntime,
    _move_opt_state,
    load_causal_lm as _load_causal_lm,
    pp_device_map as _pp_device_map,
    uses_pipeline_map as _uses_pipeline_map,
    vocab_signature as _vocab_signature,
)
from .steps import (  # noqa: F401  (step primitives re-exported for scripts)
    _capture_block_components,
    _gather_batch_rows,
    _layer_loss_per_example,
    _sliding_windows_dedup,
    _span_batch,
    last_block_step,
    last_block_step_batch,
    local_block_step,
    local_block_step_batch,
    window_step,
    window_step_batch,
)
from .teacher_source import OnlineTeacherSource, _online_targets  # noqa: F401
from .telemetry import (
    _epoch_end_telemetry,
    _epoch_zero_telemetry,
    _flush_train_log,
    _loss_float,
)
from .validate import (  # noqa: F401  (RUN_CLASSES re-exported for scripts)
    NON_METHOD_CLASSES,
    RUN_CLASSES,
    validate_knob_schedule as _validate_knob_schedule,
)


def train_layerwise(cfg: ExperimentConfig) -> Path:
    _validate_knob_schedule(cfg)
    run_dir, log = setup_run_dir(cfg)
    seed_everything(cfg.train.seed)
    # dequantize_overrides checks the RELEASED identity (cfg.model.name), not
    # student_src: a warm-start checkpoint dir carries no base quantization_config.
    moe_load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    rt = TrainingRuntime(cfg).load(moe_load_kw)
    tok, stack = rt.tokenizer, rt.stack
    # Depth-dependent window checks live here, not in the config validator:
    # n_layers is only known once the model is loaded. Oversized windows
    # previously KeyError'd deep in the walk (readout0 < 1 indexes a
    # nonexistent layer) instead of naming the misconfiguration.
    for knob in ("conn_window", "readout_window_blocks"):
        val = getattr(cfg.train, knob)
        if val > stack.n_layers:
            raise ValueError(
                f"train.{knob}={val} exceeds the model's {stack.n_layers} "
                f"blocks ({cfg.model.name})")
    teacher = rt.load_teacher(moe_load_kw)
    online = teacher is not None
    cache = None if online else rt.load_cache()

    moe = None
    if cfg.train.moe_mode != "dense_or_black_box":
        moe = MoEController(stack, cfg.train.moe_mode,
                            cfg.train.moe_router_weight)

    if cfg.train.schedule == "summed":
        _train_summed(cfg, stack, cache, tok, log, teacher, moe)
    elif cfg.train.schedule == "teacher_censored":
        if teacher is None:
            raise ValueError(
                "teacher_censored needs full-sequence teacher states: enable "
                "train.online_teacher (LoRA) or train.frozen_teacher_copy "
                "(full-FT); the disk cache stores aligned slices only"
            )
        if cfg.mask.compaction != "remove":
            raise ValueError("teacher_censored assumes compaction=remove "
                             "(stub rows have no teacher counterpart)")
        _train_teacher_censored(cfg, stack, tok, log, teacher)
    elif cfg.train.schedule == "mixed":
        if teacher is None:
            raise ValueError(
                "mixed needs full-sequence teacher states: enable "
                "train.online_teacher (LoRA) or train.frozen_teacher_copy"
            )
        if cfg.mask.compaction != "remove":
            raise ValueError("mixed assumes compaction=remove "
                             "(teacher branch deletes privileged rows)")
        _train_mixed(cfg, stack, tok, log, teacher)
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

    rt.save_checkpoint(run_dir)
    log.log(kind="done", **rt.memory_summary())
    log.close()
    return run_dir


# -- shared plumbing ---------------------------------------------------------


def _make_dataset(cfg, cache, tok, layers, with_teacher_ids=False):
    return DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=layers,
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
        with_teacher_ids=with_teacher_ids,
    )


def _loader(cfg, ds):
    if cfg.train.batching == "bucketed":
        lengths = [len(pair.student_ids) for pair in ds.pairs]
        sampler = LengthBucketBatchSampler(
            lengths,
            batch_size=cfg.train.micro_batch,
            bucket_width=cfg.train.length_bucket_width,
            seed=cfg.train.seed,
        )
        return DataLoader(
            ds, batch_sampler=sampler, collate_fn=collate_padded_items,
            num_workers=0,
        )
    if cfg.train.batching == "padded":
        return DataLoader(
            ds, batch_size=cfg.train.micro_batch, shuffle=True,
            collate_fn=collate_padded_items, num_workers=0,
            generator=torch.Generator().manual_seed(cfg.train.seed),
        )
    return DataLoader(
        ds, batch_size=cfg.train.micro_batch, shuffle=True,
        collate_fn=collate_items, num_workers=0,
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )


def _block_adamws(stack, cfg) -> dict[int, torch.optim.Optimizer]:
    """Historical per-block AdamW set used by the teacher-stream schedules.
    (The summed schedule goes through ``OptimizerPlan`` instead; per-block
    instances here keep paging/step granularity at one block.)"""
    return {
        L: torch.optim.AdamW(
            [p for p in stack.block_params(L) if p.requires_grad], lr=cfg.train.lr
        )
        for L in range(1, stack.n_layers + 1)
    }


def _step_block_adamws(stack, opts: dict[int, torch.optim.Optimizer]) -> None:
    """Per-block clip + step + zero — the same clipping granularity as
    ``OptimizerPlan.step`` (clipping is part of the experiment)."""
    for L, opt in opts.items():
        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)


# -- teacher_censored schedule ------------------------------------------------


def _train_teacher_censored(cfg, stack, tok, log, teacher):
    """Schedule (b): per-block fitting on stationary teacher-stream inputs.

    One adapters-off pass per item yields the full-sequence teacher states
    t_h[0..n]. Block L (adapters on) consumes the censored rows of t_h[L-1]
    (prefix + aligned span, privileged rows deleted, teacher position ids
    kept) and matches the teacher's aligned-span t_h[L]. Blocks never see
    each other's outputs: layer independence holds by construction, so this
    is the schedule that parallelizes across GPUs at scale."""
    device = cfg.model.device
    n = stack.n_layers
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    ds = _make_dataset(cfg, None, tok, [], with_teacher_ids=True)
    loader = _loader(cfg, ds)
    opts = _block_adamws(stack, cfg)

    step = accum = 0
    pending_losses: list[list[torch.Tensor]] = []
    t0 = time.time()
    standard_baseline = _epoch_zero_telemetry(cfg, stack, tok, log, t0)
    for epoch in range(cfg.train.epochs):
        for items in loader:
            for it in items:
                # frozen teacher states, all layers, full teacher sequence
                t_states = teacher.full_states(it, device)
                layer_losses = _censored_item(cfg, stack, loss_fn, it,
                                              t_states, device)
                accum += 1
                pending_losses.append(layer_losses)
                if accum % cfg.train.grad_accum == 0:
                    _flush_train_log(log, epoch=epoch, step=step,
                                     accum=accum, pending=pending_losses,
                                     n_layers=n)
                    _step_block_adamws(stack, opts)
                    step += 1
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n, partial=True)
        standard_baseline = _epoch_end_telemetry(
            cfg, stack, tok, log, epoch=epoch, baseline=standard_baseline,
            started_at=t0)


def censored_rows(s0: int, t0: int, A: int, t_priv, device) -> torch.Tensor:
    """Teacher-row indices of the STUDENT's view: everything before the
    aligned span except the privileged runs, then the aligned span itself.

    ``t_priv`` None/empty = the classic single block at [s0, t0) (rag /
    whole-think modes). A list of (start, stop) ranges = interleaved
    (thinking_selective): kept think runs survive between censored ones.
    The invariant ``len(rows) == s0 + A`` ties teacher-row selection to the
    student sequence length — any drift is an alignment bug, not noise."""
    if not t_priv:
        keep = [torch.arange(s0, device=device)]
    else:
        keep = []
        cur = 0
        for a, b in t_priv:
            if a > cur:
                keep.append(torch.arange(cur, a, device=device))
            cur = b
        if cur < t0:
            keep.append(torch.arange(cur, t0, device=device))
    keep.append(torch.arange(t0, t0 + A, device=device))
    rows = torch.cat(keep)
    assert len(rows) == s0 + A, (len(rows), s0, A, t_priv)
    return rows


def _censored_item(cfg, stack, loss_fn, it, t_states, device):
    """One item's per-block fitting on censored teacher-stream inputs
    (prefix rows + aligned rows, teacher position ids, privileged rows
    deleted). RESTORED to its original purpose 2026-07-05: stationary
    inputs, every layer independent, NO connected window and NO readout
    term. Teacher-stream k-windows are a distinct future mode
    (docs/windows.md)."""
    n = stack.n_layers
    # Unit-level callers historically pass a kind string, whereas the trainer
    # constructs one configured object.  Normalize once here so delta-kind
    # routing has the same information in both paths.
    if isinstance(loss_fn, str):
        loss_fn = HiddenLoss(loss_fn, stack.final_norm, stack.lm_head)
    tA0 = it.t0
    rows = censored_rows(it.s0, tA0, it.A, getattr(it, "t_priv", None), device)
    pos_c = rows[None]  # teacher absolute positions == row indices
    pos_emb_c = stack.rope(t_states[0][:, :1], pos_c)
    def _target(L):
        t = t_states[L][0, tA0: tA0 + it.A]
        return (stack.final_norm(t) if L == n else t).detach()

    layer_losses = []
    for L in range(1, n + 1):
        inp = t_states[L - 1][:, rows].detach()
        previous_target = (
            t_states[L - 1][0, tA0: tA0 + it.A].detach()
            if loss_fn.is_delta and 1 < L < n else None
        )
        if L == n:
            loss_val, _ = last_block_step(
                stack, inp, pos_emb_c, _target(L), it.s0, it.A,
                loss_fn,
            )
        else:
            loss_val, _ = local_block_step(
                stack, L, inp, pos_emb_c, _target(L), it.s0, it.A, loss_fn,
                previous_target=previous_target,
            )
        layer_losses.append(loss_val)
    return layer_losses


def _moe_row_maps(x, device):
    """Flat student-row -> flat teacher-row maps for MoEController.set_maps:
    row b*S+j of the student batch takes its routing target from teacher row
    map[b*S+j] (censored-row alignment — the same map the schedules use for
    hidden targets). mask marks real rows; padding rows carry clamped junk."""
    if isinstance(x, Batch):
        B, S = x.student_ids.shape
        T = x.teacher_ids.shape[1]
        rmap = torch.zeros(B * S, dtype=torch.long)
        mask = torch.zeros(B * S, dtype=torch.bool)
        for i in range(B):
            t_priv = x.t_priv[i] if x.t_priv is not None else None
            rows = censored_rows(int(x.s0[i]), int(x.t0[i]), int(x.A[i]),
                                 t_priv, "cpu")
            rmap[i * S: i * S + len(rows)] = rows + i * T
            mask[i * S: i * S + len(rows)] = True
        return rmap.to(device), mask.to(device)
    rows = censored_rows(x.s0, x.t0, x.A, getattr(x, "t_priv", None), device)
    return rows, torch.ones(len(rows), dtype=torch.bool, device=device)


# -- summed schedule ----------------------------------------------------------


def _summed_batch(cfg, stack, loss_fn, batch: Batch, targets, device):
    """Batched summed-schedule pass over padded examples.

    ``targets`` is {L: [B, Amax, H]}. Returned losses are a list of [B]
    tensors, ordered like the historical per-item ``layer_losses`` list.
    """
    n = stack.n_layers
    ids = batch.student_ids.to(device)
    pos = batch.position_ids.to(device)
    h = stack.embed(ids)
    pos_emb = stack.rope(h, pos)
    readout0 = n - cfg.train.readout_window_blocks + 1 if cfg.train.readout_window_blocks > 0 else n + 1
    W = max(cfg.train.conn_window, 1)
    layer_losses = []
    L = 1
    while L <= n:
        if L == readout0:
            readout_targets = {LL: targets[LL] for LL in range(readout0, n + 1)}
            readout_losses, h = window_step_batch(
                stack, readout0, h.detach(), pos_emb, readout_targets,
                batch, loss_fn, cfg.train.readout_weight,
                hidden_w=cfg.train.window_hidden_weight,
                readout_source=cfg.train.readout_source,
                all_targets=targets,
            )
            layer_losses.extend(readout_losses)
            break
        if W > 1 and cfg.train.conn_stride == 1:
            last_body = min(readout0 - 1, n)
            with torch.no_grad():
                h_traj = {L - 1: h.detach()}
                t = h
                for LL in range(L, last_body + 1):
                    t = stack.run_block(LL, t, pos_emb)
                    h_traj[LL] = t.detach()
            if cfg.train.window_dedup:
                def _endpoint_loss(L1, x, y):
                    per_ex = _layer_loss_per_example(
                        loss_fn, stack, L1, y, x, targets[L1],
                        targets.get(L1 - 1), batch,
                    )
                    return per_ex.sum(), per_ex.detach()
                layer_losses.extend(_sliding_windows_dedup(
                    stack, L, last_body, W, h_traj, pos_emb, _endpoint_loss))
            else:
                for L1 in range(L, last_body + 1):
                    L0 = max(1, L1 - W + 1)
                    win_losses, _ = window_step_batch(
                        stack, L0, h_traj[L0 - 1], pos_emb, {L1: targets[L1]},
                        batch, loss_fn, readout_w=0.0, L1=L1,
                        all_targets=targets,
                    )
                    layer_losses.extend(win_losses)
                    # trajectory lifetime: roots below the NEXT window's root
                    # are done — keep residency at W states, not the full
                    # depth (h_traj[last_body] survives as the walk's output)
                    next_root = max(L - 1, L1 + 1 - W)
                    for j in [j for j in h_traj if j < next_root and j != last_body]:
                        del h_traj[j]
            h = h_traj[last_body]
            L = last_body + 1
            continue
        if W > 1:
            L1 = min(L + W - 1, readout0 - 1, n)
            win_targets = {LL: targets[LL] for LL in range(L, L1 + 1)}
            win_losses, h = window_step_batch(
                stack, L, h.detach(), pos_emb, win_targets,
                batch, loss_fn, readout_w=0.0, L1=L1,
                all_targets=targets,
            )
            layer_losses.extend(win_losses)
            L = L1 + 1
            continue
        target = ((targets[L], targets[("attn", L)], targets[("mlp", L)])
                  if loss_fn.kind == "component_nmse" else targets[L])
        if L == n:
            loss_vals, h = last_block_step_batch(
                stack, h.detach(), pos_emb, target, batch, loss_fn,
            )
        else:
            loss_vals, h = local_block_step_batch(
                stack, L, h.detach(), pos_emb, target,
                batch, loss_fn, previous_target=targets.get(L - 1),
            )
        layer_losses.append(loss_vals)
        L += 1
    return layer_losses


def _extend_pending_from_batch(pending: list[list[torch.Tensor]],
                               layer_losses: list[torch.Tensor]) -> None:
    if not layer_losses:
        return
    B = layer_losses[0].shape[0]
    for i in range(B):
        pending.append([losses[i] for losses in layer_losses])


def _train_summed(cfg, stack, cache, tok, log, teacher=None, moe=None):
    device = cfg.model.device
    n = stack.n_layers
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    anchor = _make_anchor(cfg, tok, teacher)
    online = teacher is not None
    ds = _make_dataset(cfg, cache, tok,
                       [] if online else list(range(1, n + 1)),
                       with_teacher_ids=online)
    loader = _loader(cfg, ds)
    plan = OptimizerPlan.build(stack, cfg)

    step = accum = 0
    next_step = cfg.train.grad_accum
    pending_losses: list[list[torch.Tensor]] = []
    t0 = time.time()
    done = False
    standard_baseline = _epoch_zero_telemetry(cfg, stack, tok, log, t0)
    for epoch in range(cfg.train.epochs):
        if done:
            break
        for items in loader:
            if done:
                break
            # item batching flows through the same batched stages as B=1
            # padded batches: bit-identical to the historical item loop (no
            # pad rows, gather == slice, same kernel shapes), one item per
            # grad-accum increment exactly as before
            batches = ([items] if isinstance(items, Batch)
                       else [collate_padded_items([it]) for it in items])
            for batch in batches:
                if done:
                    break
                # teacher stage: aligned targets from the online teacher or
                # the collated disk-cache slices
                with (moe.teacher_phase() if moe else contextlib.nullcontext()):
                    targets = (teacher.aligned_targets_batch(
                                   batch, device,
                                   capture_components=(cfg.train.hidden_loss == "component_nmse"))
                               if online else batch.hidden)
                if cfg.train.scramble_targets:
                    # audit control: layer-permuted targets (see config)
                    perm = list(range(1, n + 1))
                    random.Random(cfg.train.seed).shuffle(perm)
                    targets = {L: targets[perm[L - 1]] for L in range(1, n + 1)}
                if moe is not None:
                    moe.set_maps(*_moe_row_maps(batch, device))
                # trajectory + loss stages: the summed walk (backward happens
                # inside _summed_batch, block-local by detach discipline)
                with (moe.student_phase() if moe else contextlib.nullcontext()):
                    layer_losses = _summed_batch(cfg, stack, loss_fn, batch,
                                                 targets, device)
                accum += len(batch.example_ids)
                _extend_pending_from_batch(pending_losses, layer_losses)
                # update stage: per-block clip + optimizer policy step at
                # grad-accum boundaries
                if accum >= next_step:
                    _flush_train_log(log, epoch=epoch, step=step,
                                     accum=accum, pending=pending_losses,
                                     n_layers=n,
                                     batch_size=len(batch.example_ids),
                                     batching=cfg.train.batching,
                                     **({"router_overlap": moe.overlap_flush()}
                                        if moe else {}))
                    if anchor is not None:
                        a_ids, a_logits, a_states = anchor[0].next()
                        if cfg.train.anchor_kl_weight > 0:
                            anchor_step(stack, n - cfg.train.readout_window_blocks + 1,
                                        a_ids, cfg.train.anchor_kl_weight,
                                        base_logits=a_logits)
                        if cfg.train.anchor_hidden_weight > 0:
                            anchor_trajectory_step(stack, a_ids, a_states,
                                                   cfg.train.anchor_hidden_weight)
                    plan.step()
                    step += 1
                    next_step += cfg.train.grad_accum
                    if cfg.train.max_steps and step >= cfg.train.max_steps:
                        done = True
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n, partial=True,
                         **({"router_overlap": moe.overlap_flush()}
                            if moe else {}))
        standard_baseline = _epoch_end_telemetry(
            cfg, stack, tok, log, epoch=epoch, baseline=standard_baseline,
            started_at=t0)


# -- mixed schedule -----------------------------------------------------------


def mix_teacher_p(cfg, epoch: int) -> float:
    """Linear anneal of the teacher-branch probability from
    ``mix_teacher_start`` (epoch 0) to ``mix_teacher_end`` (last epoch)."""
    s, e = cfg.train.mix_teacher_start, cfg.train.mix_teacher_end
    if cfg.train.epochs <= 1:
        return e
    return s + (e - s) * epoch / (cfg.train.epochs - 1)


def _train_mixed(cfg, stack, tok, log, teacher):
    """Scheduled-sampling routing: per item, a Bernoulli draw picks between
    the teacher-stream censored branch (stationary inputs, early training)
    and the student-stream summed branch (the deployment-matched input
    distribution, late training). One teacher forward per item feeds both
    branches. The branch generator is separate from the loader's shuffle
    generator so sibling arms at the same seed see identical item order."""
    device = cfg.model.device
    n = stack.n_layers
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    ds = _make_dataset(cfg, None, tok, [], with_teacher_ids=True)
    loader = _loader(cfg, ds)
    opts = _block_adamws(stack, cfg)
    branch_gen = torch.Generator().manual_seed(cfg.train.seed + 1)

    step = accum = 0
    pending_losses: list[list[torch.Tensor]] = []
    branch_counts = {"teacher": 0, "student": 0}
    t0 = time.time()
    standard_baseline = _epoch_zero_telemetry(cfg, stack, tok, log, t0)
    for epoch in range(cfg.train.epochs):
        p = mix_teacher_p(cfg, epoch)
        for items in loader:
            for it in items:
                t_states = teacher.full_states(it, device)
                use_teacher = torch.rand((), generator=branch_gen).item() < p
                if use_teacher:
                    layer_losses = _censored_item(cfg, stack, loss_fn, it,
                                                  t_states, device)
                else:
                    # student branch through the unified batched walk (B=1)
                    targets = {
                        L: (stack.final_norm(t_states[L][0, it.t0: it.t0 + it.A])
                            if L == n else t_states[L][0, it.t0: it.t0 + it.A]
                            ).detach()[None]
                        for L in range(1, n + 1)
                    }
                    batch_losses = _summed_batch(
                        cfg, stack, loss_fn, collate_padded_items([it]),
                        targets, device)
                    layer_losses = [loss[0] for loss in batch_losses]
                accum += 1
                branch = "teacher" if use_teacher else "student"
                branch_counts[branch] += 1
                pending_losses.append(layer_losses)
                if accum % cfg.train.grad_accum == 0:
                    _flush_train_log(log, epoch=epoch, step=step,
                                     accum=accum, pending=pending_losses,
                                     n_layers=n, p_teacher=round(p, 4),
                                     teacher_items=branch_counts["teacher"],
                                     student_items=branch_counts["student"])
                    branch_counts = {"teacher": 0, "student": 0}
                    _step_block_adamws(stack, opts)
                    step += 1
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n,
                         p_teacher=round(p, 4),
                         teacher_items=branch_counts["teacher"],
                         student_items=branch_counts["student"],
                         partial=True)
        branch_counts = {"teacher": 0, "student": 0}
        standard_baseline = _epoch_end_telemetry(
            cfg, stack, tok, log, epoch=epoch, baseline=standard_baseline,
            started_at=t0)


# -- sequential schedule ------------------------------------------------------


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
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    act_cache = StudentActCache()
    t0 = time.time()

    ds = _make_dataset(cfg, cache, tok, [1])  # pairs built once; layer swapped per stage
    full_ft = not cfg.train.lora.enabled
    for L in range(1, n + 1):
        ds.need_layers = ([L - 1, L] if loss_fn.is_delta and 1 < L < n else [L])
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
                    previous_target = (
                        it.hidden[L - 1].to(device)
                        if loss_fn.is_delta and 1 < L < n else None
                    )
                    if L == n:
                        loss_val, _ = last_block_step(
                            stack, h_in.detach(), pos_emb, target, it.s0, it.A,
                            loss_fn,
                        )
                    else:
                        loss_val, _ = local_block_step(
                            stack, L, h_in.detach(), pos_emb, target,
                            it.s0, it.A, loss_fn,
                            previous_target=previous_target,
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
            mean_loss = (sum(_loss_float(loss) for loss in epoch_losses)
                         / max(1, len(epoch_losses)))
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
            r = tasks_eval(stack.model, tok, cfg.data.poem_path,
                           n_per_task=8,
                           generation_batch=cfg.eval.generation_batch)
            t = r["tasks"]
            log.log(kind="eval", layer=L,
                    next_acc=t["next"]["word_acc"],
                    prev_acc=t["prev"]["word_acc"],
                    cloze_acc=t["cloze"]["word_acc"],
                    overall_word_acc=r["overall_word_acc"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"after layer {L}: overall word-acc {r['overall_word_acc']:.2f}")
