"""Online (per-step) teacher-state sources.

The disk cache stores aligned slices only; schedules that need full-sequence
or per-step teacher states (teacher_censored, mixed, online-LoRA summed)
draw them from a frozen teacher resident in the same process. Construction
belongs to ``TrainingRuntime.load_teacher``; this module only defines the
source itself.
"""

from __future__ import annotations

import contextlib

import torch

from ..data.dataset import Batch
from .steps import _capture_block_components


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
    def full_states_cpu(self, it, device) -> list[torch.Tensor]:
        """Uncensored ``h0..hn`` staged in host RAM one layer at a time.

        Pipeline-v3 teacher forcing needs full prefixes, whereas the durable
        v2 cache intentionally stores aligned rows only.  Copying each state
        before advancing keeps one teacher activation on GPU instead of
        retaining ``n+1`` complete sequence tensors there.  These states are
        immutable for the current answer and discarded before the next one.
        """
        t_ids = it.teacher_ids.to(device)[None]
        t_pos = torch.arange(t_ids.shape[1], device=device)[None]
        states = []
        with self._ctx(), torch.autocast(
                torch.device(device).type, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            states.append(h.detach().cpu())
            for L in range(1, self.stack.n_layers + 1):
                h = self.stack.run_block(L, h, pos_emb)
                states.append(h.detach().cpu())
        return states

    @torch.no_grad()
    def full_inputs_resident(self, it, device) -> list[torch.Tensor]:
        """Uncensored ``h0..h[n-1]`` retained on each block's device.

        Pipeline-v3 consumes these rows once per local token update. Keeping
        them answer-local on their owning GPUs avoids a tiny host-to-device
        transfer in every layer×token cell. Unlike ``full_states_cpu``, this
        returns block inputs only; v3 targets still come from the frozen disk
        cache.
        """
        t_ids = it.teacher_ids.to(device)[None]
        t_pos = torch.arange(t_ids.shape[1], device=device)[None]
        inputs = []
        with self._ctx(), torch.autocast(
                torch.device(device).type, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            for layer in range(1, self.stack.n_layers + 1):
                block_device = (
                    self.stack.block_devices[layer - 1]
                    if self.stack.block_devices is not None else
                    next(self.stack.blocks[layer - 1].parameters()).device)
                if h.device != block_device:
                    h = h.to(block_device, non_blocking=True)
                inputs.append(h.detach())
                h = self.stack.run_block(layer, h, pos_emb)
        return inputs

    @torch.no_grad()
    def full_inputs_resident_batch(self, batch: Batch, device) -> list[torch.Tensor]:
        """Uncensored block inputs for simultaneous-user v3.1 probes.

        Token sequences are right padded, so padding follows every real
        causal row and cannot affect it. Consumers remap each real prefix to
        their batched KV timeline and apply the censorship/padding mask there.
        """
        if batch.teacher_ids is None:
            raise ValueError("v3.1 teacher batch needs teacher_ids")
        t_ids = batch.teacher_ids.to(device)
        t_pos = torch.arange(t_ids.shape[1], device=device)[None].expand(
            t_ids.shape[0], -1)
        inputs = []
        with self._ctx(), torch.autocast(
                torch.device(device).type, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            for layer in range(1, self.stack.n_layers + 1):
                block_device = (
                    self.stack.block_devices[layer - 1]
                    if self.stack.block_devices is not None else
                    next(self.stack.blocks[layer - 1].parameters()).device)
                if h.device != block_device:
                    h = h.to(block_device, non_blocking=True)
                inputs.append(h.detach())
                h = self.stack.run_block(layer, h, pos_emb)
        return inputs

    @torch.no_grad()
    def full_inputs_pinned_batch(self, batch: Batch, device) -> list[torch.Tensor]:
        """Uncensored block inputs in bounded, host-pinned storage.

        Pipeline-v3.2 gathers only the current BxK tile back to the GPU.  The
        v3.1 resident variant retained n*[B,T,H] on device and therefore could
        not fit teacher-hidden 4B cohorts regardless of activation sharding.
        Copies are issued on the producing stream and synchronized once after
        the layer walk; returned tensors are immutable staging sources.
        """
        if batch.teacher_ids is None:
            raise ValueError("v3.2 teacher batch needs teacher_ids")
        device = torch.device(device)
        t_ids = batch.teacher_ids.to(device)
        t_pos = torch.arange(t_ids.shape[1], device=device)[None].expand(
            t_ids.shape[0], -1)
        inputs = []
        with self._ctx(), torch.autocast(device.type, dtype=torch.bfloat16):
            h = self.stack.embed(t_ids)
            pos_emb = self.stack.rope(h, t_pos)
            for layer in range(1, self.stack.n_layers + 1):
                staged = torch.empty(
                    h.shape, dtype=h.dtype, device="cpu", pin_memory=True)
                staged.copy_(h.detach(), non_blocking=True)
                inputs.append(staged)
                h = self.stack.run_block(layer, h, pos_emb)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        return inputs

    @torch.no_grad()
    def aligned_targets(self, it, device) -> dict[int, torch.Tensor]:
        states = self.full_states(it, device)
        return {
            L: self.stack.loss_view(L, states[L])[0, it.t0: it.t0 + it.A].detach()
            for L in range(1, self.stack.n_layers + 1)
        }

    @torch.no_grad()
    def aligned_targets_batch(self, batch: Batch, device,
                              capture_components: bool = False) -> dict:
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
                if capture_components:
                    with _capture_block_components(self.stack, L) as components:
                        h = self.stack.run_block(L, h, pos_emb)
                    for name in ("attn", "mlp"):
                        value = components[name]
                        out[(name, L)] = value[
                            row.to(value.device), idx.to(value.device)].detach()
                else:
                    h = self.stack.run_block(L, h, pos_emb)
                view = self.stack.loss_view(L, h)
                view_device = view.device
                out[L] = view[row.to(view_device), idx.to(view_device)].detach()
        return out


def _online_targets(stack, peft_model, it, device):
    """Back-compat wrapper (scripts import this): adapters-off aligned targets."""
    return OnlineTeacherSource(stack, peft_model=peft_model).aligned_targets(it, device)
