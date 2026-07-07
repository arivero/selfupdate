"""MoE routing modes (``train.moe_mode``): teacher_forced and router_aligned.

``dense_or_black_box`` needs nothing from this module: the router/expert
mechanism stays inside the block and block-output distillation is already
valid method evidence (see config.TrainConfig.moe_mode).

teacher_forced — the student's MoE layers replay the TEACHER's expert
selection. The online teacher's per-item full-sequence pass records every
MoE layer's top-k expert indices; during the student walk each MoE layer
routes each student row to the experts the teacher chose for the
corresponding teacher row (the same censored-row student->teacher map the
schedules use). Gate weights stay student-computed, restricted to the
forced expert set, so the differentiable part of routing still trains.

router_aligned — natural student routing, plus a per-layer regularizer
KL(teacher routing || student routing) with the UNIFORM weight
``train.moe_router_weight`` on every MoE layer (depth-uniform by
construction: the naming contract applies to routers too). Router
parameters keep their existing trainability — frozen under LoRA, so the
alignment acts through the block representation and the adapters-off
online teacher stays exactly the base model.

Both modes accumulate teacher/student top-k overlap per layer ON GPU
(no ``.item()`` in the block walk — the sync-bound lesson) and flush it
with the train log.

Kernel-agnosticism: hub kernels replace ``GptOssMLP.forward`` CLASS-wide
and never call the Python router submodule (scripts/moe_router_probe.py).
While a controller phase is active, gpt-oss MoE layers therefore run an
INSTANCE-level pure-Python forward with identical math; with no phase
active the original (possibly fused) class forward runs untouched, so
eval/generation keep their fast path. Gemma4 MoE decoder layers call
``self.router`` in plain Python, so wrapping the router module suffices.
A tripwire raises if an active phase sees no router calls.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

# the controller whose pending router losses the step functions drain;
# set only inside student_phase()
_ACTIVE: "MoEController | None" = None


def pending_router_loss():
    """Drained by every block/window step right before its backward: the
    graph of a pending KL term is part of the current step's graph, so it
    must join that step's total (leaking it across steps would retain the
    whole window graph)."""
    if _ACTIVE is None:
        return None
    return _ACTIVE._drain()


def dequantize_overrides(model_name: str, moe_mode: str) -> dict:
    """Extra ``from_pretrained`` kwargs needed for routing intervention.

    ``teacher_forced``/``router_aligned`` call ``mlp.experts(flat, idx,
    scores)`` directly with the plain 3-arg signature. gpt-oss's released
    checkpoints are MXFP4-quantized (``quant_method: mxfp4``), which
    replaces ``experts`` with ``Mxfp4GptOssExperts`` (needs a triton
    routing-kernel's ``routing_data``/``gather_idx``/``scatter_idx``, not
    this signature) and patches ``mlp.forward`` to match — incompatible
    with our forced-routing call. ``Mxfp4Config(dequantize=True)`` keeps
    the plain ``GptOssExperts`` interface (dense bf16 weights, ~2x the
    quantized footprint but still LoRA/single-card sized for gpt-oss-20b).
    dense_or_black_box never touches ``.experts`` directly, so it always
    keeps the quantized (smaller, kernel-fast) weights untouched."""
    if moe_mode == "dense_or_black_box":
        return {}
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    qc = getattr(cfg, "quantization_config", None) or getattr(
        getattr(cfg, "text_config", None), "quantization_config", None)
    if isinstance(qc, dict) and qc.get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config

        return {"quantization_config": Mxfp4Config(dequantize=True)}
    return {}


class _GptOssMoE:
    """gpt-oss family: ``block.mlp`` owns ``router`` (bare weight/bias
    Parameters, returns softmax over top-k logits) and ``experts``.

    Released gpt-oss checkpoints ship MXFP4-quantized experts
    (``Mxfp4GptOssExperts``): transformers' mxfp4 loader patches
    ``mlp.forward`` at the INSTANCE level (``MethodType(mlp_forward, mlp)``,
    see ``transformers.integrations.mxfp4.replace_with_mxfp4_linear``) to a
    triton-routing-kernel forward whose ``experts`` call needs
    ``routing_data``/``gather_idx``/``scatter_idx`` — a different calling
    convention than the plain ``(hidden_states, indices, weights)`` this
    adapter's forced/aligned routing uses. ``match()`` only tests for
    ``router``/``experts`` attributes, which the quantized module also has,
    so intervention modes require the caller to load the model with
    ``Mxfp4Config(dequantize=True)`` (see ``dequantize_overrides``) —
    that keeps ``experts`` as plain ``GptOssExperts``. The passthrough path
    below still captures the INSTANCE's actual forward (not the class
    method) so a non-dequantized model's normal (kernelized) fast path
    survives untouched whenever no phase is active."""

    def __init__(self, block):
        self.mlp = block.mlp

    @staticmethod
    def match(block) -> bool:
        mlp = getattr(block, "mlp", None)
        return (mlp is not None and hasattr(mlp, "router")
                and hasattr(mlp, "experts") and hasattr(mlp.router, "top_k"))

    def install(self, ctrl: "MoEController", L: int) -> None:
        mlp = self.mlp
        # the LIVE forward, which may already be an instance-level patch
        # (e.g. mxfp4's kernelized mlp_forward) — never the class method,
        # which can be stale relative to that patch.
        orig_forward = mlp.forward

        def forward(hidden_states):
            if ctrl.phase is None:
                return orig_forward(hidden_states)
            if not hasattr(mlp.experts, "gate_up_proj") or not isinstance(
                mlp.experts.gate_up_proj, torch.nn.Parameter
            ):
                raise RuntimeError(
                    f"{type(mlp.experts).__name__} is not the plain "
                    "GptOssExperts interface (quantized experts need a "
                    "kernel-specific routing call this adapter cannot make) "
                    "— load with Mxfp4Config(dequantize=True) for "
                    "teacher_forced/router_aligned (see moe.dequantize_overrides)")
            batch, seq, dim = hidden_states.shape
            flat = hidden_states.reshape(-1, dim)
            logits = F.linear(flat, mlp.router.weight, mlp.router.bias)
            nat_idx = torch.topk(logits, mlp.router.top_k, dim=-1).indices
            idx = ctrl.on_router(L, logits, nat_idx)
            scores = F.softmax(logits.gather(-1, idx), dim=-1,
                               dtype=logits.dtype)
            out = mlp.experts(flat, idx, scores)
            return out.reshape(batch, seq, dim), scores

        mlp.forward = forward


class _Gemma4MoE:
    """gemma4 MoE decoder layers: ``router``/``experts`` live on the layer;
    the router returns (probs over all experts, top-k weights with
    per-expert scale, top-k indices) and is called in plain Python."""

    def __init__(self, block):
        self.block = block

    @staticmethod
    def match(block) -> bool:
        return (hasattr(block, "router") and hasattr(block, "experts")
                and hasattr(block.router, "per_expert_scale"))

    def install(self, ctrl: "MoEController", L: int) -> None:
        router = self.block.router
        orig_cls_forward = type(router).forward

        def forward(hidden_states):
            probs, weights, nat_idx = orig_cls_forward(router, hidden_states)
            if ctrl.phase is None:
                return probs, weights, nat_idx
            logp = torch.log(probs.clamp_min(1e-9))
            idx = ctrl.on_router(L, logp, nat_idx)
            if idx is not nat_idx:  # teacher-forced expert set
                w = probs.gather(-1, idx)
                w = w / w.sum(dim=-1, keepdim=True)
                w = w * router.per_expert_scale[idx]
                return probs, w, idx
            return probs, weights, nat_idx

        router.forward = forward


class _Qwen35MoE:
    """qwen3_5_moe family: ``block.mlp`` is a SparseMoeBlock whose ``gate``
    (TopKRouter) is called in plain Python; scores are softmax-then-
    renormalized top-k probabilities. The shared expert is dense and always
    on — routing intervention only concerns ``gate``/``experts``."""

    def __init__(self, block):
        self.gate = block.mlp.gate

    @staticmethod
    def match(block) -> bool:
        mlp = getattr(block, "mlp", None)
        return (mlp is not None and hasattr(mlp, "gate")
                and hasattr(mlp, "experts") and hasattr(mlp.gate, "top_k"))

    def install(self, ctrl: "MoEController", L: int) -> None:
        gate = self.gate
        orig_cls_forward = type(gate).forward

        def forward(hidden_states):
            logits, scores, nat_idx = orig_cls_forward(gate, hidden_states)
            if ctrl.phase is None:
                return logits, scores, nat_idx
            idx = ctrl.on_router(L, logits, nat_idx)
            if idx is not nat_idx:  # teacher-forced expert set
                probs = F.softmax(logits, dtype=torch.float, dim=-1)
                w = probs.gather(-1, idx)
                w = (w / w.sum(dim=-1, keepdim=True)).to(logits.dtype)
                return logits, w, idx
            return logits, scores, nat_idx

        gate.forward = forward


class MoEController:
    """Owns the wrapped MoE layers of the STUDENT stack and the per-item
    teacher routing state. Phases:

    - ``teacher_phase()``: wraps the online teacher's adapters-off pass;
      records natural top-k indices (and log-probs for router_aligned)
      per MoE layer over the full teacher sequence.
    - ``student_phase()``: wraps one item/batch's schedule walk; requires
      ``set_maps`` first. Forces or aligns routing, accumulates overlap.

    Outside any phase the wrapped modules are exact passthroughs.
    """

    def __init__(self, stack, mode: str, router_weight: float = 0.0):
        if mode not in ("teacher_forced", "router_aligned"):
            raise ValueError(f"MoEController does not handle mode {mode!r}")
        self.mode = mode
        self.w = float(router_weight)
        self.phase = None
        self.adapters: dict[int, object] = {}
        for L in range(1, stack.n_layers + 1):
            block = stack.blocks[L - 1]
            if _GptOssMoE.match(block):
                ad = _GptOssMoE(block)
            elif _Qwen35MoE.match(block):
                ad = _Qwen35MoE(block)
            elif _Gemma4MoE.match(block):
                ad = _Gemma4MoE(block)
            else:
                continue
            ad.install(self, L)
            self.adapters[L] = ad
        if not self.adapters:
            raise ValueError(
                f"moe_mode={mode!r} but no MoE layers found in "
                f"{type(stack.model).__name__} — dense models must use "
                "dense_or_black_box (an arm that silently does nothing is "
                "a confound, not a control)")
        self.t_idx: dict[int, torch.Tensor] = {}   # L -> [Nt, k] long
        self.t_logp: dict[int, torch.Tensor] = {}  # L -> [Nt, E] bf16
        self.row_map: torch.Tensor | None = None   # [Ns] long, flat rows
        self.row_mask: torch.Tensor | None = None  # [Ns] bool
        self._pending: list[torch.Tensor] = []
        self._fired = 0
        # per-layer overlap accumulators, GPU-resident until flush
        self._ov_sum: dict[int, torch.Tensor] = {}
        self._ov_cnt: dict[int, torch.Tensor] = {}

    # ---- phases -------------------------------------------------------
    @contextlib.contextmanager
    def teacher_phase(self):
        self.t_idx.clear()
        self.t_logp.clear()
        self._fired = 0
        self.phase = "teacher"
        try:
            yield
        finally:
            self.phase = None
            if self._fired == 0:
                raise RuntimeError(
                    "MoE tripwire: teacher pass called no wrapped router — "
                    "is a fused-kernel path bypassing the Python MoE forward?")

    @contextlib.contextmanager
    def student_phase(self):
        global _ACTIVE
        if self.row_map is None:
            raise RuntimeError("student_phase needs set_maps() per item/batch")
        self._fired = 0
        self.phase = "student"
        _ACTIVE = self
        try:
            yield
        finally:
            self.phase = None
            _ACTIVE = None
            self.row_map = self.row_mask = None
            if self._fired == 0:
                raise RuntimeError(
                    "MoE tripwire: student walk called no wrapped router")
            if self._pending:
                self._pending.clear()
                raise RuntimeError(
                    "MoE tripwire: pending router losses were never drained "
                    "by a step function — graph leak")

    def set_maps(self, row_map: torch.Tensor, row_mask: torch.Tensor) -> None:
        """Flat student-row -> flat teacher-row map for the NEXT student
        phase (row b*S+j of the student batch reads teacher row map[b*S+j];
        mask marks real rows — padding rows are clamped junk)."""
        self.row_map = row_map
        self.row_mask = row_mask

    # ---- router callback ---------------------------------------------
    def on_router(self, L: int, logp: torch.Tensor,
                  nat_idx: torch.Tensor) -> torch.Tensor:
        """``logp``: [N, E] log-space router scores (unnormalized logits or
        log-probabilities — always re-normalized via log_softmax here).
        Returns the indices the layer must route with."""
        self._fired += 1
        if self.phase == "teacher":
            self.t_idx[L] = nat_idx.detach()
            if self.mode == "router_aligned":
                self.t_logp[L] = F.log_softmax(
                    logp.detach().float(), dim=-1).to(torch.bfloat16)
            return nat_idx
        t_idx = self.t_idx.get(L)
        if t_idx is None:
            raise RuntimeError(f"no teacher routing captured for layer {L}")
        rm = self.row_map.to(t_idx.device)
        mask = self.row_mask.to(nat_idx.device)
        if nat_idx.shape[0] != rm.numel():
            raise RuntimeError(
                f"router saw {nat_idx.shape[0]} rows but {rm.numel()} are "
                "mapped — a non-item forward (anchor/eval) ran inside "
                "student_phase")
        t_rows = t_idx[rm].to(nat_idx.device)  # [N, k]
        with torch.no_grad():
            hit = (nat_idx.unsqueeze(-1) == t_rows.unsqueeze(-2)).any(-1)
            frac = hit.float().mean(dim=-1)  # [N] natural-top-k overlap
            fsum = (frac * mask).sum()
            fcnt = mask.float().sum()
            if L in self._ov_sum:
                self._ov_sum[L] += fsum
                self._ov_cnt[L] += fcnt
            else:
                self._ov_sum[L] = fsum
                self._ov_cnt[L] = fcnt
        if self.mode == "teacher_forced":
            return t_rows
        if torch.is_grad_enabled() and self.w > 0:
            t_lp = self.t_logp[L][rm].to(logp.device)
            kl = F.kl_div(
                F.log_softmax(logp.float(), dim=-1)[mask],
                t_lp.float()[mask],
                log_target=True, reduction="batchmean")
            self._pending.append(self.w * kl)
        return nat_idx

    # ---- consumers ----------------------------------------------------
    def _drain(self):
        if not self._pending:
            return None
        total = sum(self._pending)
        self._pending.clear()
        return total

    def overlap_flush(self) -> dict[int, float]:
        """Per-MoE-layer mean teacher/student top-k overlap since the last
        flush. The only sync point — call at accum boundaries."""
        out = {L: round((self._ov_sum[L] / self._ov_cnt[L].clamp_min(1)).item(), 4)
               for L in sorted(self._ov_sum)}
        self._ov_sum.clear()
        self._ov_cnt.clear()
        return out
