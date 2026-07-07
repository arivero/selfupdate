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
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig
from ..data.dataset import (
    Batch,
    DistillDataset,
    LengthBucketBatchSampler,
    collate_items,
    collate_padded_items,
)
from ..eval.general import general_ce
from ..eval.recite import recite_eval
from ..teacher.cache import TeacherCache, resolve_cache_dir
from ..utils.runlog import setup_run_dir
from ..utils.seeding import seed_everything
from .blocks import BlockStack
from .losses import HiddenLoss
from .moe import MoEController, dequantize_overrides, pending_router_loss

RUN_CLASSES = {
    "method", "teacher_reference", "ablation", "control",
    "legacy_archive", "confounded", "open",
}
NON_METHOD_CLASSES = RUN_CLASSES - {"method"}


def _vocab_signature(stack) -> tuple:
    """Cheap exact fingerprint of the frozen vocabulary tensors (embedding,
    final norm, head). Computed at trainer start and re-checked before
    save: NO learning of any kind may modify these — they are the fixed
    basis of every lens and every cached teacher target."""
    sig = []
    seen: set[int] = set()
    for m in (stack.embed_tokens, stack.final_norm, stack.lm_head):
        for p in m.parameters():
            # tied-embedding models (Qwen3 <=1.7B): embed IS lm_head — one
            # pass over the shared tensor, not two (only compared within-run)
            if id(p) in seen:
                continue
            seen.add(id(p))
            # chunked fp64 sums: a full p.double() copy of a 200k-vocab
            # embedding is ~4 GB — enough to OOM a 20B-resident card
            s = a = 0.0
            for chunk in p.detach().reshape(-1).split(1 << 22):
                c = chunk.double()
                s += c.sum().item()
                a += c.abs().sum().item()
            sig.append((s, a))
    return tuple(sig)


def local_block_step(stack, L, h_in, pos_emb, target, s0, A, kind, autocast=True):
    """One local forward+backward for block L. ``h_in`` must be detached, so
    the recorded graph — and therefore the backward — is confined to block L:
    no gradient from this loss can reach any other block, the lm_head, or the
    logits. Returns (loss value, detached block output). Autocast wraps only
    the forward+loss; backward runs outside it.

    ``kind`` is a HiddenLoss or a kind string (coerced; vocab-metric kinds
    need the constructed HiddenLoss carrying the frozen norm/head)."""
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(L, h_in, pos_emb)
        loss = loss_fn(stack.loss_view(L, h_out)[0, s0: s0 + A], target,
                       normed=(L == stack.n_layers))
    extra = pending_router_loss()
    (loss if extra is None else loss + extra).backward()
    return loss.detach(), h_out.detach()


def last_block_step(stack, h_in, pos_emb, target, s0, A, kind, autocast=True):
    """Block n's local teacher-state step."""
    n = stack.n_layers
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(n, h_in, pos_emb)
        normed = stack.final_norm(h_out)
        loss = loss_fn(normed[0, s0: s0 + A], target, normed=True)
    extra = pending_router_loss()
    (loss if extra is None else loss + extra).backward()
    return loss.detach(), h_out.detach()


def _gather_batch_rows(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    idx = index.to(h.device).unsqueeze(-1).expand(-1, -1, h.shape[-1])
    return h.gather(1, idx)


def _hidden_loss_per_example(loss_fn, student_h: torch.Tensor, teacher_h: torch.Tensor,
                             lens: list[int], *, normed: bool,
                             layer: int | None = None) -> torch.Tensor:
    """Right-padded batches keep every valid row in a PREFIX (collate
    invariant), so slicing by the CPU-side length replaces bool-mask
    indexing — whose implicit nonzero() is a host-device sync per example
    per layer, the same stall class as .item() in the block walk."""
    teacher_h = teacher_h.to(student_h.device)
    losses = []
    for i, k in enumerate(lens):
        losses.append(loss_fn(student_h[i, :k], teacher_h[i, :k],
                              normed=normed, layer=layer))
    return torch.stack(losses)


def _teacher_kl_per_example(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            lens: list[int]) -> torch.Tensor:
    teacher_logits = teacher_logits.to(student_logits.device)
    losses = []
    for i, k in enumerate(lens):
        losses.append(F.kl_div(
            F.log_softmax(student_logits[i, :k].float(), dim=-1),
            F.log_softmax(teacher_logits[i, :k].float(), dim=-1),
            log_target=True,
            reduction="batchmean",
        ))
    return torch.stack(losses)


def local_block_step_batch(stack, L, h_in, pos_emb, target, batch: Batch, kind,
                           autocast=True):
    """Batched counterpart of :func:`local_block_step`.

    The total backward scalar is the sum of per-example losses, matching the
    historical item loop's gradient scale while sharing the block forward.
    """
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(L, h_in, pos_emb)
        view = stack.loss_view(L, h_out)
        aligned = _gather_batch_rows(view, batch.aligned_index)
        losses = _hidden_loss_per_example(
            loss_fn, aligned, target, batch.A.tolist(),
            normed=(L == stack.n_layers), layer=L
        )
        total = losses.sum()
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return losses.detach(), h_out.detach()


def last_block_step_batch(stack, h_in, pos_emb, target, batch: Batch, kind,
                          autocast=True):
    n = stack.n_layers
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h_out = stack.run_block(n, h_in, pos_emb)
        normed = stack.final_norm(h_out)
        aligned = _gather_batch_rows(normed, batch.aligned_index)
        losses = _hidden_loss_per_example(
            loss_fn, aligned, target, batch.A.tolist(), normed=True,
            layer=n
        )
        total = losses.sum()
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return losses.detach(), h_out.detach()


def window_step(stack, L0, h_in, pos_emb, targets, s0, A, ans_off, kind,
                readout_w, hidden_w=1.0, L1=None, readout_source="teacher_kl",
                autocast=True):
    """Joint step for a CONNECTED window [L0..L1] (default L1 = n, the
    top readout window): gradient flows within the window so a loss anywhere
    in it can assign credit up to ``L1 - L0 + 1`` blocks deep. Per-block
    hidden losses are scaled by ``hidden_w``. A readout applies only when the
    window ends at the top (L1 == n), where logits exist. The window is rooted
    at a detached ``h_in``: no gradient reaches blocks < L0, and the frozen
    norm/head receive none. Peak graph = window width."""
    n = stack.n_layers
    L1 = n if L1 is None else L1
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h = h_in
        losses = []
        for L in range(L0, L1 + 1):
            h = stack.run_block(L, h, pos_emb)
            if L in targets:  # sparse targets: endpoint-sliding windows
                losses.append(loss_fn(
                    stack.loss_view(L, h)[0, s0: s0 + A], targets[L],
                    normed=(L == n)))
        total = hidden_w * sum(losses)
        if readout_w > 0 and L1 == n:
            logits = stack.lm_head(
                stack.final_norm(h)[0, s0 + ans_off - 1: s0 + A - 1])
            if readout_source == "teacher_kl":
                # 100% teacher-sourced readout: targets[n] is the teacher's
                # post-norm state at the aligned span — its logits through
                # the frozen head ARE the context-conditioned distribution
                with torch.no_grad():
                    t_logits = stack.lm_head(
                        targets[n][ans_off - 1: A - 1].to(logits.dtype))
                total = total + readout_w * F.kl_div(
                    F.log_softmax(logits.float(), dim=-1),
                    F.log_softmax(t_logits.float(), dim=-1),
                    log_target=True, reduction="batchmean")
            else:
                raise ValueError(
                    f"unknown readout_source {readout_source!r}; only teacher_kl is allowed"
                )
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return [l.detach() for l in losses], h.detach()


def window_step_batch(stack, L0, h_in, pos_emb, targets, batch: Batch, kind,
                      readout_w, hidden_w=1.0, L1=None,
                      readout_source="teacher_kl", autocast=True):
    """Batched connected-window step.

    Returns a list of per-example loss vectors in the same layer order as
    :func:`window_step`.
    """
    n = stack.n_layers
    L1 = n if L1 is None else L1
    loss_fn = HiddenLoss(kind) if isinstance(kind, str) else kind
    with torch.autocast(h_in.device.type, dtype=torch.bfloat16, enabled=autocast):
        h = h_in
        losses = []
        for L in range(L0, L1 + 1):
            h = stack.run_block(L, h, pos_emb)
            if L in targets:
                aligned = _gather_batch_rows(stack.loss_view(L, h), batch.aligned_index)
                losses.append(_hidden_loss_per_example(
                    loss_fn, aligned, targets[L], batch.A.tolist(),
                    normed=(L == n), layer=L,
                ))
        if losses:
            total = hidden_w * sum(loss.sum() for loss in losses)
        else:
            total = h.sum() * 0.0
        if readout_w > 0 and L1 == n:
            logits = stack.lm_head(
                stack.final_norm(_gather_batch_rows(h, batch.readout_index))
            )
            if readout_source == "teacher_kl":
                teacher_rows = (batch.readout_index.to(targets[n].device)
                                - batch.s0.to(targets[n].device).unsqueeze(1))
                teacher_rows = teacher_rows.clamp_min(0)
                teacher_h = _gather_batch_rows(targets[n], teacher_rows)
                with torch.no_grad():
                    t_logits = stack.lm_head(
                        teacher_h.to(device=logits.device, dtype=logits.dtype)
                    )
                total = total + readout_w * _teacher_kl_per_example(
                    logits, t_logits, (batch.s0 + batch.A - batch.ans0).tolist(),
                ).sum()
            else:
                raise ValueError(
                    f"unknown readout_source {readout_source!r}; only teacher_kl is allowed"
                )
    extra = pending_router_loss()
    (total if extra is None else total + extra).backward()
    return [l.detach() for l in losses], h.detach()


def _sliding_windows_dedup(stack, L_start, last_body, W, h_traj, pos_emb,
                           compute_loss, autocast=True):
    """Forward-deduplicated FAITHFUL sliding windows (train.window_dedup).

    Same semantics as the per-endpoint ``window_step`` replay: every body
    layer L1 is the ENDPOINT of a window [L1-W+1 .. L1] rooted at the
    detached trajectory state h_traj[L0-1], and its backward updates ALL
    covered blocks — uniform k-deep credit (docs/windows.md). The backward
    count per block is untouched (that IS the credit assignment); what goes
    away is the forward duplication: instead of re-forwarding W blocks per
    endpoint, each block is grad-forwarded ONCE from its detached trajectory
    root, and every window chains its backward through the stored per-block
    graphs via ``torch.autograd.grad`` grad_outputs injection. This is valid
    because all windows follow the same trajectory: the value of window
    [L0..L1]'s intermediate state at depth b equals h_traj[b], the root of
    block b+1's stored graph (exactly in fp32; up to autocast replay rounding
    in bf16). Gradient isolation is preserved — each chain stops at the
    detached root x_{L0-1}, and no graph connects two blocks.

    Peak graph memory stays at W blocks: a block's graph is freed right
    after its last covering window (endpoint min(last_body, b+W-1)).

    ``compute_loss(L1, y)`` returns (backward scalar, detached report value)
    for the endpoint loss on block output ``y``; it runs under the same
    autocast as the forward. Returns the report values in endpoint order.
    """
    dev_type = h_traj[L_start - 1].device.type
    xs, ys, params = {}, {}, {}
    reports = []
    for L1 in range(L_start, last_body + 1):
        # .detach() gives x its own autograd identity; h_traj stays a plain
        # detached trajectory shared with the caller
        x = h_traj[L1 - 1].detach().requires_grad_(True)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=autocast):
            y = stack.run_block(L1, x, pos_emb)
            loss, report = compute_loss(L1, y)
        xs[L1], ys[L1] = x, y
        params[L1] = [p for p in stack.block_params(L1) if p.requires_grad]
        L0 = max(L_start, L1 - W + 1)
        g = None
        for b in range(L1, L0 - 1, -1):
            last_use = L1 == min(last_body, b + W - 1)
            inputs = [xs[b]] + params[b]
            if b == L1:
                grads = torch.autograd.grad(
                    loss, inputs, retain_graph=not last_use, allow_unused=True)
            else:
                g = g.to(device=ys[b].device, dtype=ys[b].dtype)
                grads = torch.autograd.grad(
                    ys[b], inputs, grad_outputs=g,
                    retain_graph=not last_use, allow_unused=True)
            g = grads[0]
            for p, gp in zip(params[b], grads[1:]):
                if gp is None:
                    continue
                p.grad = gp.detach() if p.grad is None else p.grad.add_(gp)
            if last_use:
                del xs[b], ys[b], params[b]
        reports.append(report)
    return reports


def _load_causal_lm(src, **kw):
    """Load a decoder LM regardless of head registration. Multimodal
    releases like Mistral-Medium-3.5 (mistral3) register ONLY as
    image-text-to-text and are absent from the causal-LM auto-map, so
    ``AutoModelForCausalLM`` raises; the ITT wrapper exposes the same
    ``.model.language_model`` decoder stack BlockStack navigates. gemma4 and
    qwen3_5_moe ARE in the causal map and take the first path unchanged."""
    try:
        return AutoModelForCausalLM.from_pretrained(src, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(src, **kw)


def _pp_device_map(cfg) -> dict:
    """Pipeline map: embedding on cuda:0; decoder blocks partitioned by
    ``pipeline_split`` (2 GPUs) or ``pipeline_splits`` (N GPUs). The final
    norm/head live on the last card for untied models, so the top readout
    window stays colocated with logits."""
    from transformers import AutoConfig

    mc = AutoConfig.from_pretrained(cfg.model.name)
    text_cfg = getattr(mc, "text_config", mc)
    n = text_cfg.num_hidden_layers
    splits = list(cfg.model.pipeline_splits or [])
    if splits:
        if torch.cuda.device_count() < len(splits) + 1:
            raise ValueError(
                f"pipeline_splits {splits} needs {len(splits) + 1} visible GPUs"
            )
        if splits != sorted(splits) or splits[0] <= 0 or splits[-1] >= n:
            raise ValueError(f"pipeline_splits {splits} outside 1..{n - 1}")
    else:
        if torch.cuda.device_count() < 2:
            raise ValueError("pipeline_split needs 2 visible GPUs (queue n_gpus=2)")
        split = cfg.model.pipeline_split
        if not 0 < split < n:
            raise ValueError(f"pipeline_split {split} outside 1..{n - 1}")
        splits = [split]
    # tied embeddings (Qwen3 <=1.7B): embed IS lm_head — one tensor cannot
    # live on two cards, so the whole vocabulary stack stays on cuda:0 and
    # readout-window loss calls hop back (an [A,H] transfer per call). Untied
    # models put norm+head on cuda:1 with the readout window.
    tied = getattr(mc, "tie_word_embeddings",
                   getattr(text_cfg, "tie_word_embeddings", False))
    last_dev = len(splits)
    vocab_dev = 0 if tied else last_dev
    prefix = "model.language_model" if getattr(mc, "model_type", "") == "gemma4" else "model"
    dm = {f"{prefix}.embed_tokens": 0, f"{prefix}.rotary_emb": 0,
          f"{prefix}.norm": vocab_dev, "lm_head": vocab_dev}
    if prefix != "model":
        dm["model.vision_tower"] = 0
        dm["model.embed_vision"] = 0
    for i in range(n):
        dev = 0
        while dev < len(splits) and i >= splits[dev]:
            dev += 1
        dm[f"{prefix}.layers.{i}"] = dev
    return dm


def _uses_pipeline_map(cfg) -> bool:
    return cfg.model.pipeline_split > 0 or bool(cfg.model.pipeline_splits)


def _validate_knob_schedule(cfg) -> None:
    """Knob-flow law (2026-07-05): a knob that a schedule does not implement
    must RAISE, never silently ignore — spec/code divergence is the bug
    class that produced the unwired-ce_kind incident. Keep this in sync
    with the knob-flow audit table in tests/test_training_target_law.py."""
    sched = cfg.train.schedule
    run_class = cfg.train.run_class
    bad = []
    if run_class not in RUN_CLASSES:
        raise ValueError(f"unknown train.run_class {run_class!r}")
    if run_class == "teacher_reference":
        raise ValueError(
            "train.run_class='teacher_reference' is eval-only: epoch-zero "
            "recall belongs in evaluation artifacts, never in training")
    if cfg.train.batching not in ("item", "padded", "bucketed"):
        raise ValueError(f"unknown train.batching {cfg.train.batching!r}")
    if cfg.train.moe_mode not in (
        "dense_or_black_box", "teacher_forced", "router_aligned",
    ):
        raise ValueError(f"unknown train.moe_mode {cfg.train.moe_mode!r}")
    if cfg.train.moe_mode != "dense_or_black_box":
        if sched != "summed":
            bad.append("moe_mode (teacher_forced/router_aligned are "
                       "implemented for the summed schedule only)")
        if not cfg.train.online_teacher:
            bad.append("moe_mode needs train.online_teacher (routing targets "
                       "are per-step, captured adapters-off on the same "
                       "wrapped blocks — disk cache stores no routing)")
        if (cfg.train.moe_mode == "router_aligned"
                and cfg.train.moe_router_weight <= 0):
            bad.append("router_aligned requires explicit moe_router_weight > 0"
                       " (no silent default)")
    elif cfg.train.moe_router_weight != 0.0:
        bad.append("moe_router_weight without moe_mode=router_aligned")
    if cfg.train.batching != "item":
        if sched != "summed":
            bad.append("batching (currently implemented for summed schedule only)")
        if cfg.train.grad_accum % cfg.train.micro_batch != 0:
            bad.append("grad_accum must be a multiple of micro_batch for batched training")
    is_method = run_class == "method"
    if cfg.train.conn_window > 1 and sched not in ("summed", "mixed"):
        bad.append("conn_window")
    if cfg.train.scramble_targets and sched != "summed":
        bad.append("scramble_targets")
    if cfg.train.window_dedup and (cfg.train.conn_window <= 1
                                   or cfg.train.conn_stride != 1):
        bad.append("window_dedup (needs faithful sliding windows: "
                   "conn_window > 1, conn_stride == 1)")
    if cfg.train.offload_adam and sched != "summed":
        bad.append("offload_adam")
    if cfg.train.readout_window_blocks > 0 and cfg.train.readout_source == "UNSET":
        raise ValueError(
            "readout_source must be set EXPLICITLY to teacher_kl when "
            "readout_window_blocks > 0 — defaults are experiment variables")
    if cfg.train.readout_source not in ("UNSET", "teacher_kl"):
        bad.append("readout_source must be teacher_kl; reference-text training is forbidden")
    if cfg.train.readout_window_blocks > 0:
        if sched == "teacher_censored":
            bad.append("readout_window_blocks (teacher_censored is pure by definition)")
        if is_method:
            if cfg.train.conn_window <= 0 or cfg.train.conn_stride != 1:
                bad.append("readout_window_blocks without sanctioned sliding conn_window/conn_stride")
            if cfg.train.readout_window_blocks != cfg.train.conn_window:
                bad.append("readout_window_blocks must equal conn_window for method arms")
    if is_method and cfg.train.window_hidden_weight != 1.0:
        bad.append("window_hidden_weight != 1.0 (ablation/control only)")
    if sched == "tail_only":
        raise ValueError("schedule 'tail_only' was expunged 2026-07-05 "
                         "(damnatio memoriae — owner directive); its CE "
                         "silently targeted the original text")
    if bad:
        raise ValueError(
            f"knob(s) {bad} not implemented for schedule {sched!r} — "
            "refusing to silently ignore")


def train_layerwise(cfg: ExperimentConfig) -> Path:
    _validate_knob_schedule(cfg)
    run_dir, log = setup_run_dir(cfg)
    seed_everything(cfg.train.seed)
    device = cfg.model.device
    if _uses_pipeline_map(cfg) and cfg.model.device_map:
        raise ValueError("model.pipeline_split(s) and model.device_map are mutually exclusive")
    if cfg.model.device_map not in ("", "auto"):
        raise ValueError("model.device_map must be empty or 'auto'")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    # bf16 base for LoRA (frozen weights) AND for the sequential
    # schedules: only actively-training blocks need fp32 master weights
    # (cast per stage / per window); summed full-FT trains all blocks every
    # step and keeps fp32 masters throughout.
    full_ft_all_blocks = (not cfg.train.lora.enabled
                          and cfg.train.schedule != "sequential")
    base_dtype = torch.float32 if full_ft_all_blocks else torch.bfloat16
    # warm-start: student weights from a prior run's checkpoint; the teacher
    # (cache identity / frozen copy / adapters-off) stays cfg.model.name
    student_src = (str(Path("runs") / cfg.train.init_from / "checkpoint")
                   if cfg.train.init_from else cfg.model.name)
    pp_map = _pp_device_map(cfg) if _uses_pipeline_map(cfg) else None
    auto_map = cfg.model.device_map == "auto"
    # dequantize_overrides checks the RELEASED identity (cfg.model.name), not
    # student_src: a warm-start checkpoint dir carries no base quantization_config.
    moe_load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    if pp_map is not None:
        model = _load_causal_lm(student_src, dtype=base_dtype, device_map=pp_map,
                                **moe_load_kw)
    elif auto_map:
        model = _load_causal_lm(student_src, dtype=base_dtype, device_map="auto",
                                **moe_load_kw)
    else:
        model = _load_causal_lm(student_src, dtype=base_dtype, **moe_load_kw)
        model.to(device)
    peft_model = None
    if cfg.train.lora.enabled:
        from .lora import attach_lora

        peft_model = attach_lora(model, cfg.train.lora)
        model = peft_model.get_base_model()
    model.train()
    stack = BlockStack(model)
    stack.freeze_non_blocks()
    vocab_sig0 = _vocab_signature(stack)

    if cfg.train.online_teacher and peft_model is None:
        raise ValueError("train.online_teacher requires train.lora.enabled")
    teacher = None
    if cfg.train.online_teacher:
        teacher = OnlineTeacherSource(stack, peft_model=peft_model)
    elif cfg.train.frozen_teacher_copy:
        # resident frozen bf16 copy: online teacher for full-FT schedules
        if pp_map is not None:
            t_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name, dtype=torch.bfloat16, device_map=pp_map, **moe_load_kw)
            t_model.eval().requires_grad_(False)
        elif auto_map:
            t_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name, dtype=torch.bfloat16, device_map="auto", **moe_load_kw)
            t_model.eval().requires_grad_(False)
        else:
            t_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name, dtype=torch.bfloat16, **moe_load_kw)
            t_model.to(device).eval().requires_grad_(False)
        teacher = OnlineTeacherSource(stack, frozen_stack=BlockStack(t_model))
    online = teacher is not None
    cache = None
    if not online:
        cache_root, chash = resolve_cache_dir(cfg)
        cache = TeacherCache(cache_root, expect_hash=chash)

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

    if _vocab_signature(stack) != vocab_sig0:
        raise RuntimeError(
            "frozen-vocabulary violation: embedding/final-norm/head changed "
            "during training — refusing to save (docs/hidden_loss.md)"
        )
    if peft_model is not None:
        peft_model.save_pretrained(run_dir / "checkpoint")
    else:
        model.to(torch.bfloat16)
        model.save_pretrained(run_dir / "checkpoint")
    tok.save_pretrained(run_dir / "checkpoint")
    n_dev = torch.cuda.device_count()
    log.log(kind="done",
            # summed across visible cards (pipeline-parallel jobs use two)
            vram_gb=round(sum(torch.cuda.max_memory_allocated(d)
                              for d in range(n_dev)) / 2**30, 2),
            # reserved = what the allocator actually holds from the device —
            # the honest footprint for "does it fit on this card" claims
            vram_reserved_gb=round(sum(torch.cuda.max_memory_reserved(d)
                                       for d in range(n_dev)) / 2**30, 2),
            vram_per_device_gb=[round(torch.cuda.max_memory_reserved(d) / 2**30, 2)
                                for d in range(n_dev)])
    log.close()
    return run_dir


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


def _loss_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.detach().float().cpu())
    return float(v)


def _summarize_pending_losses(pending: list[list[torch.Tensor]], n_layers: int) -> tuple[float, list[float]]:
    sums: list[torch.Tensor | None] = [None] * n_layers
    counts = [0] * n_layers
    for losses in pending:
        for i, loss in enumerate(losses[:n_layers]):
            t = loss.detach().float() if torch.is_tensor(loss) else torch.tensor(float(loss))
            sums[i] = t if sums[i] is None else sums[i] + t
            counts[i] += 1
    # single host transfer per flush: gather per-layer means onto one device
    # (async d2d under pipeline parallel) and .cpu() the stack once, instead
    # of one GPU->CPU round-trip per layer
    dev = next((s.device for s in sums if s is not None), torch.device("cpu"))
    means = [
        (sums[i] / counts[i]).to(dev, non_blocking=True) if counts[i]
        else torch.full((), float("nan"), device=dev)
        for i in range(n_layers)
    ]
    per_layer = [float(v) for v in torch.stack(means).cpu()]
    valid = [v for v in per_layer if v == v]
    mean = sum(valid) / len(valid) if valid else float("nan")
    return mean, per_layer


def _flush_train_log(log, *, epoch: int, step: int, accum: int,
                     pending: list[list[torch.Tensor]], n_layers: int, **extra) -> None:
    if not pending:
        return
    loss, per_layer = _summarize_pending_losses(pending, n_layers)
    log.log(kind="train", epoch=epoch, step=step, items_seen=accum,
            accum_items=len(pending), loss=loss, per_layer=per_layer, **extra)
    pending.clear()


class OnlineTeacherSource:
    """Frozen-teacher forwards for schedules that need per-step teacher
    states. Two backends, exactly one active:

    - ``peft_model``: adapters-off pass on the resident base (LoRA runs) —
      the teacher is already resident, zero extra VRAM.
    - ``frozen_stack``: a resident frozen bf16 copy of the base model — the
      full-FT path (``train.frozen_teacher_copy``), ~1.2 GB at 0.6B.

    ``full_states`` returns raw block outputs [h0..hn] over the full teacher
    sequence (final norm applied by the consumer, matching the
    teacher_censored convention). ``aligned_targets`` returns {L: [A, H]}
    with the h_n post-norm convention — exactly what the disk cache stores.
    """

    def __init__(self, student_stack, peft_model=None, frozen_stack=None):
        if (peft_model is None) == (frozen_stack is None):
            raise ValueError("exactly one of peft_model / frozen_stack")
        self.stack = frozen_stack if frozen_stack is not None else student_stack
        self.peft_model = peft_model

    def _ctx(self):
        return (self.peft_model.disable_adapter() if self.peft_model
                else contextlib.nullcontext())

    @torch.no_grad()
    def full_states(self, it, device) -> list[torch.Tensor]:
        t_ids = it.teacher_ids.to(device)[None]
        t_pos = torch.arange(t_ids.shape[1], device=device)[None]
        with self._ctx(), torch.autocast(device, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            states = [h]
            for L in range(1, self.stack.n_layers + 1):
                h = self.stack.run_block(L, h, pos_emb)
                states.append(h)
        return states

    @torch.no_grad()
    def aligned_targets(self, it, device) -> dict[int, torch.Tensor]:
        states = self.full_states(it, device)
        return {
            L: self.stack.loss_view(L, states[L])[0, it.t0: it.t0 + it.A].detach()
            for L in range(1, self.stack.n_layers + 1)
        }

    @torch.no_grad()
    def aligned_targets_batch(self, batch: Batch, device) -> dict[int, torch.Tensor]:
        """Streamed: each layer's aligned rows are gathered as the block
        runs, so a single full-sequence state is resident at a time instead
        of all n+1 — the batched teacher costs one layer of VRAM, not a
        stack. (full_states stays list-shaped for teacher_censored/mixed,
        which genuinely consume every layer's full sequence.)"""
        if batch.teacher_ids is None:
            raise ValueError("online teacher batch needs teacher_ids")
        if batch.t0 is None:
            raise ValueError("online teacher batch needs t0")
        t_ids = batch.teacher_ids.to(device)
        t_pos = torch.arange(t_ids.shape[1], device=device)[None].expand(
            t_ids.shape[0], -1
        )
        B, Amax = batch.hidden_mask.shape
        offsets = torch.arange(Amax, device=device)[None]
        t0 = batch.t0.to(device)[:, None]
        row = torch.arange(B, device=device)[:, None]
        out: dict[int, torch.Tensor] = {}
        with self._ctx(), torch.autocast(device, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            idx = (t0 + offsets).clamp_max(h.shape[1] - 1)
            for L in range(1, self.stack.n_layers + 1):
                h = self.stack.run_block(L, h, pos_emb)
                view = self.stack.loss_view(L, h)
                view_device = view.device
                out[L] = view[row.to(view_device), idx.to(view_device)].detach()
        return out


def _online_targets(stack, peft_model, it, device):
    """Back-compat wrapper (tests import this): adapters-off aligned targets."""
    return OnlineTeacherSource(stack, peft_model=peft_model).aligned_targets(it, device)


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
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head,
                         tuned_lens_path=cfg.train.tuned_lens_path)
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
    pending_losses: list[list[torch.Tensor]] = []
    t0 = time.time()
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
                    for L, opt in opts.items():
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    step += 1
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n, partial=True)
        if (epoch + 1) % cfg.eval.every_epochs == 0 or epoch == cfg.train.epochs - 1:
            r = recite_eval(stack.model, tok, records, limit=8)
            log.log(kind="eval", epoch=epoch, cer=r["cer"], cer_flat=r["cer_flat"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"epoch {epoch}: eval CER {r['cer']:.3f} line-exact {r['line_exact']:.3f}")


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
        if L == n:
            loss_val, _ = last_block_step(
                stack, inp, pos_emb_c, _target(L), it.s0, it.A,
                loss_fn,
            )
        else:
            loss_val, _ = local_block_step(
                stack, L, inp, pos_emb_c, _target(L), it.s0, it.A, loss_fn,
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


def _summed_item(cfg, stack, loss_fn, it, targets, device):
    """One item's pass on the student's own stream: block-local steps (or
    sliding conn_window-connected windows) below, plus an optional teacher-KL
    connected readout window at the top. ``targets`` is {L: [A, H]}
    regardless of source (disk cache or online teacher)."""
    n = stack.n_layers
    ids = it.student_ids.to(device)[None]
    pos = it.position_ids.to(device)[None]
    h = stack.embed(ids)
    pos_emb = stack.rope(h, pos)
    readout0 = n - cfg.train.readout_window_blocks + 1 if cfg.train.readout_window_blocks > 0 else n + 1
    W = max(cfg.train.conn_window, 1)
    layer_losses = []
    L = 1
    while L <= n:
        if L == readout0:
            readout_targets = {LL: targets[LL] for LL in range(readout0, n + 1)}
            readout_losses, h = window_step(
                stack, readout0, h.detach(), pos_emb, readout_targets,
                it.s0, it.A, it.ans0 - it.s0, loss_fn,
                cfg.train.readout_weight,
                hidden_w=cfg.train.window_hidden_weight,
                readout_source=cfg.train.readout_source,
            )
            layer_losses.extend(readout_losses)
            break
        if W > 1 and cfg.train.conn_stride == 1:
            # FAITHFUL sliding windows: one clean no-grad trajectory, then
            # every body layer L1 is matched as the ENDPOINT of a window
            # [L1-W+1 .. L1] whose backward updates ALL covered blocks —
            # uniform k-deep credit for every layer (owner's design).
            last_body = min(readout0 - 1, n)
            with torch.no_grad():
                h_traj = {L - 1: h.detach()}
                t = h
                for LL in range(L, last_body + 1):
                    t = stack.run_block(LL, t, pos_emb)
                    h_traj[LL] = t.detach()
            if cfg.train.window_dedup:
                def _endpoint_loss(L1, y):
                    loss = loss_fn(stack.loss_view(L1, y)[0, it.s0: it.s0 + it.A],
                                   targets[L1], normed=(L1 == n))
                    return loss, loss.detach()
                layer_losses.extend(_sliding_windows_dedup(
                    stack, L, last_body, W, h_traj, pos_emb, _endpoint_loss))
            else:
                for L1 in range(L, last_body + 1):
                    L0 = max(1, L1 - W + 1)
                    win_losses, _ = window_step(
                        stack, L0, h_traj[L0 - 1], pos_emb, {L1: targets[L1]},
                        it.s0, it.A, it.ans0 - it.s0, loss_fn,
                        readout_w=0.0, L1=L1,
                    )
                    layer_losses.extend(win_losses)
            h = h_traj[last_body]
            L = last_body + 1
            continue
        if W > 1:
            # DISJOINT windows (conn_stride 0): detach every W blocks —
            # cheap approximation; credit depth varies inside the window
            L1 = min(L + W - 1, readout0 - 1, n)
            win_targets = {LL: targets[LL] for LL in range(L, L1 + 1)}
            win_losses, h = window_step(
                stack, L, h.detach(), pos_emb, win_targets,
                it.s0, it.A, it.ans0 - it.s0, loss_fn,
                readout_w=0.0, L1=L1,
            )
            layer_losses.extend(win_losses)
            L = L1 + 1
            continue
        if L == n:
            loss_val, h = last_block_step(
                stack, h.detach(), pos_emb, targets[L], it.s0, it.A,
                loss_fn,
            )
        else:
            loss_val, h = local_block_step(
                stack, L, h.detach(), pos_emb, targets[L],
                it.s0, it.A, loss_fn,
            )
        layer_losses.append(loss_val)
        L += 1
    return layer_losses


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
                def _endpoint_loss(L1, y):
                    aligned = _gather_batch_rows(stack.loss_view(L1, y),
                                                 batch.aligned_index)
                    per_ex = _hidden_loss_per_example(
                        loss_fn, aligned, targets[L1], batch.A.tolist(),
                        normed=(L1 == n), layer=L1)
                    return per_ex.sum(), per_ex.detach()
                layer_losses.extend(_sliding_windows_dedup(
                    stack, L, last_body, W, h_traj, pos_emb, _endpoint_loss))
            else:
                for L1 in range(L, last_body + 1):
                    L0 = max(1, L1 - W + 1)
                    win_losses, _ = window_step_batch(
                        stack, L0, h_traj[L0 - 1], pos_emb, {L1: targets[L1]},
                        batch, loss_fn, readout_w=0.0, L1=L1,
                    )
                    layer_losses.extend(win_losses)
            h = h_traj[last_body]
            L = last_body + 1
            continue
        if W > 1:
            L1 = min(L + W - 1, readout0 - 1, n)
            win_targets = {LL: targets[LL] for LL in range(L, L1 + 1)}
            win_losses, h = window_step_batch(
                stack, L, h.detach(), pos_emb, win_targets,
                batch, loss_fn, readout_w=0.0, L1=L1,
            )
            layer_losses.extend(win_losses)
            L = L1 + 1
            continue
        if L == n:
            loss_vals, h = last_block_step_batch(
                stack, h.detach(), pos_emb, targets[L], batch, loss_fn,
            )
        else:
            loss_vals, h = local_block_step_batch(
                stack, L, h.detach(), pos_emb, targets[L],
                batch, loss_fn,
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


def _move_opt_state(opt, device) -> None:
    """Page an optimizer's per-param state tensors between devices (Adam
    moments dominate full-FT memory at 8 B/param). Moving "back" targets
    each PARAM's own device — under pipeline parallel the blocks live on
    different cards and a global device string would silently migrate
    moments to the wrong one."""
    to_cpu = torch.device(device).type == "cpu"
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            tgt = torch.device("cpu") if to_cpu else p.device
            for k, v in st.items():
                if torch.is_tensor(v) and v.device != tgt:
                    st[k] = v.to(tgt)


def _train_summed(cfg, stack, cache, tok, log, teacher=None, moe=None):
    device = cfg.model.device
    n = stack.n_layers
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head,
                         tuned_lens_path=cfg.train.tuned_lens_path)
    anchor = _make_anchor(cfg, tok, teacher)
    online = teacher is not None
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
    offload = cfg.train.offload_adam

    step = accum = 0
    next_step = cfg.train.grad_accum
    pending_losses: list[list[torch.Tensor]] = []
    t0 = time.time()
    done = False
    for epoch in range(cfg.train.epochs):
        if done:
            break
        for items in loader:
            if done:
                break
            if isinstance(items, Batch):
                with (moe.teacher_phase() if moe else contextlib.nullcontext()):
                    targets = (teacher.aligned_targets_batch(items, device)
                               if online else items.hidden)
                if cfg.train.scramble_targets:
                    import random as _rnd
                    perm = list(range(1, n + 1))
                    _rnd.Random(cfg.train.seed).shuffle(perm)
                    targets = {L: targets[perm[L - 1]] for L in range(1, n + 1)}
                if moe is not None:
                    moe.set_maps(*_moe_row_maps(items, device))
                with (moe.student_phase() if moe else contextlib.nullcontext()):
                    layer_losses = _summed_batch(cfg, stack, loss_fn, items,
                                                 targets, device)
                accum += len(items.example_ids)
                _extend_pending_from_batch(pending_losses, layer_losses)
                if accum >= next_step:
                    _flush_train_log(log, epoch=epoch, step=step,
                                     accum=accum, pending=pending_losses,
                                     n_layers=n,
                                     batch_size=len(items.example_ids),
                                     batching=cfg.train.batching,
                                     **({"router_overlap": moe.overlap_flush()}
                                        if moe else {}))
                    if anchor is not None:
                        a_ids, a_base = anchor[0].next()
                        anchor_step(stack, n - cfg.train.readout_window_blocks + 1,
                                    a_ids, anchor[1], base_logits=a_base)
                    for L, opt in opts.items():
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        if offload:
                            _move_opt_state(opt, device)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if offload:
                            _move_opt_state(opt, "cpu")
                    step += 1
                    next_step += cfg.train.grad_accum
                    if cfg.train.max_steps and step >= cfg.train.max_steps:
                        done = True
                continue
            for it in items:
                if done:
                    break
                with (moe.teacher_phase() if moe else contextlib.nullcontext()):
                    targets = (teacher.aligned_targets(it, device) if online
                               else {L: it.hidden[L].to(device) for L in range(1, n + 1)})
                if cfg.train.scramble_targets:
                    # audit control: layer-permuted targets (see config)
                    import random as _rnd
                    perm = list(range(1, n + 1))
                    _rnd.Random(cfg.train.seed).shuffle(perm)
                    targets = {L: targets[perm[L - 1]] for L in range(1, n + 1)}
                if moe is not None:
                    moe.set_maps(*_moe_row_maps(it, device))
                with (moe.student_phase() if moe else contextlib.nullcontext()):
                    layer_losses = _summed_item(cfg, stack, loss_fn, it,
                                                targets, device)
                accum += 1
                pending_losses.append(layer_losses)
                if accum >= next_step:
                    _flush_train_log(log, epoch=epoch, step=step,
                                     accum=accum, pending=pending_losses,
                                     n_layers=n,
                                     **({"router_overlap": moe.overlap_flush()}
                                        if moe else {}))
                    if anchor is not None:
                        a_ids, a_base = anchor[0].next()
                        anchor_step(stack, n - cfg.train.readout_window_blocks + 1,
                                    a_ids, anchor[1], base_logits=a_base)
                    for L, opt in opts.items():
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        if offload:
                            _move_opt_state(opt, device)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        if offload:
                            _move_opt_state(opt, "cpu")
                    step += 1
                    next_step += cfg.train.grad_accum
                    if cfg.train.max_steps and step >= cfg.train.max_steps:
                        done = True
                        break
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n, partial=True,
                         **({"router_overlap": moe.overlap_flush()}
                            if moe else {}))
        if (epoch + 1) % cfg.eval.every_epochs == 0 or epoch == cfg.train.epochs - 1:
            r = recite_eval(stack.model, tok, records, limit=8,
                            rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")))
            log.log(kind="eval", epoch=epoch, cer=r["cer"], cer_flat=r["cer_flat"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"epoch {epoch}: eval CER {r['cer']:.3f} line-exact {r['line_exact']:.3f}")


class AnchorBank:
    """Tokenized neighbor-genre fragments (blank-line separated), cycled at
    optimizer-step boundaries for the anti-intrusion anchor."""

    def __init__(self, path, tok, device, max_tokens: int = 96):
        texts = [t.strip() for t in Path(path).read_text(encoding="utf-8").split("\n\n")
                 if t.strip()]
        if not texts:
            raise ValueError(f"no anchor fragments in {path}")
        self.ids = [torch.tensor(tok.encode(t, add_special_tokens=False)[:max_tokens],
                                 device=device) for t in texts]
        self.base_logits: list[torch.Tensor] | None = None
        self.i = 0

    @torch.no_grad()
    def precompute_base_logits(self, teacher: "OnlineTeacherSource"):
        """Base-model logits per fragment (anchor-KL targets), computed once
        through the frozen teacher (adapters-off or frozen copy)."""
        st = teacher.stack
        device = self.ids[0].device
        outs = []
        with teacher._ctx(), torch.autocast(device.type, dtype=torch.bfloat16):
            for ids in self.ids:
                pos = torch.arange(len(ids), device=device)[None]
                h = st.embed(ids[None])
                pe = st.rope(h, pos)
                for L in range(1, st.n_layers + 1):
                    h = st.run_block(L, h, pe)
                outs.append(st.lm_head(st.final_norm(h))[0].detach())
        self.base_logits = outs

    def next(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        j = self.i % len(self.ids)
        self.i += 1
        base = self.base_logits[j] if self.base_logits is not None else None
        return self.ids[j], base


def anchor_step(stack, L0, ids, w, base_logits=None, autocast=True):
    """Anti-intrusion anchor on a neighbor-genre fragment, gradient
    confined to the top readout window [L0..n] (input detached below the window,
    frozen norm/head): counters the readout trigger ("poetic Spanish ->
    recite the poem") exactly where catastrophic remembering showed it is
    installed. Returns the unweighted loss value.

    ``base_logits`` is required: KL(base || student) per position means
    "on neighbor-genre input, behave like the teacher/base model"."""
    if base_logits is None:
        raise ValueError("anchor_step requires teacher/base logits; reference-token CE is forbidden")
    device = ids.device
    pos = torch.arange(len(ids), device=device)[None]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=autocast):
        h = stack.embed(ids[None])
        pos_emb = stack.rope(h, pos)
        for L in range(1, L0):
            h = stack.run_block(L, h, pos_emb)
    h = h.detach()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        for L in range(L0, stack.n_layers + 1):
            h = stack.run_block(L, h, pos_emb)
        full_logits = stack.lm_head(stack.final_norm(h))[0]
        loss = F.kl_div(
            F.log_softmax(full_logits.float(), dim=-1),
            F.log_softmax(base_logits.float(), dim=-1),
            log_target=True, reduction="batchmean",
        )
    (w * loss).backward()
    return loss.detach()


def _make_anchor(cfg, tok, teacher=None):
    """Returns (bank, weight) or None. Anchor regularization is teacher KL."""
    w = cfg.train.anchor_kl_weight
    if w <= 0:
        return None
    if cfg.train.readout_window_blocks <= 0:
        raise ValueError("anchor weights need readout_window_blocks > 0 "
                         "(the anchor regularizes the top readout window)")
    bank = AnchorBank(cfg.train.anchor_path, tok, cfg.model.device)
    if cfg.train.anchor_kl_weight > 0:
        if teacher is None:
            raise ValueError("anchor_kl_weight needs an online teacher for "
                             "base logits: enable train.frozen_teacher_copy "
                             "or LoRA + train.online_teacher")
        bank.precompute_base_logits(teacher)
    return bank, w



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
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head,
                         tuned_lens_path=cfg.train.tuned_lens_path)
    ds = _make_dataset(cfg, None, tok, [], with_teacher_ids=True)
    records = ds.records
    loader = _loader(cfg, ds)
    opts = {
        L: torch.optim.AdamW(
            [p for p in stack.block_params(L) if p.requires_grad], lr=cfg.train.lr
        )
        for L in range(1, n + 1)
    }
    branch_gen = torch.Generator().manual_seed(cfg.train.seed + 1)

    step = accum = 0
    pending_losses: list[list[torch.Tensor]] = []
    branch_counts = {"teacher": 0, "student": 0}
    t0 = time.time()
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
                    targets = {
                        L: (stack.final_norm(t_states[L][0, it.t0: it.t0 + it.A])
                            if L == n else t_states[L][0, it.t0: it.t0 + it.A]).detach()
                        for L in range(1, n + 1)
                    }
                    layer_losses = _summed_item(cfg, stack, loss_fn, it,
                                                targets, device)
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
                    for L, opt in opts.items():
                        torch.nn.utils.clip_grad_norm_(stack.block_params(L), 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    step += 1
        _flush_train_log(log, epoch=epoch, step=step, accum=accum,
                         pending=pending_losses, n_layers=n,
                         p_teacher=round(p, 4),
                         teacher_items=branch_counts["teacher"],
                         student_items=branch_counts["student"],
                         partial=True)
        branch_counts = {"teacher": 0, "student": 0}
        if (epoch + 1) % cfg.eval.every_epochs == 0 or epoch == cfg.train.epochs - 1:
            r = recite_eval(stack.model, tok, records, limit=8)
            log.log(kind="eval", epoch=epoch, cer=r["cer"], cer_flat=r["cer_flat"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
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
    loss_fn = HiddenLoss(cfg.train.hidden_loss, stack.final_norm, stack.lm_head,
                         tuned_lens_path=cfg.train.tuned_lens_path)
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
                        loss_val, _ = last_block_step(
                            stack, h_in.detach(), pos_emb, target, it.s0, it.A,
                            loss_fn,
                        )
                    else:
                        loss_val, _ = local_block_step(
                            stack, L, h_in.detach(), pos_emb, target,
                            it.s0, it.A, loss_fn,
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
            r = recite_eval(stack.model, tok, records, limit=8,
                            rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")))
            log.log(kind="eval", layer=L, cer=r["cer"], cer_flat=r["cer_flat"], line_exact=r["line_exact"],
                    prefix_lines=r["prefix_lines"],
                    # per-epoch forgetting reference: CER says when the poem
                    # arrives, gen_ce says when the model starts paying for it
                    gen_ce=general_ce(stack.model, tok)["mean_ce"],
                    vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    vram_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    minutes=round((time.time() - t0) / 60, 1))
            print(f"after layer {L}: eval CER {r['cer']:.3f}")
