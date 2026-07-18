"""Pipeline-v4: blockwise teacher-forced training with frozen teacher KV.

Every training loss is block-local against the cached teacher hidden states:
block L runs on the teacher's own ``i{L} = h[L-1]`` rows and is matched to
the teacher's ``h[L]`` at the same positions.  The attention context is the
teacher's OWN frozen K/V — adapters-off projections of the cached full-prefix
inputs — so gradients enter only through the query-side path of block L.
The student's trajectory is NEVER a loss input; it exists only for the
evaluation relay (M3) and the generation probes.

Censorship is pure attention censorship: every privileged key (the RAG
passage AND the prompt text announcing it, ``t_privileged``) is removed from
the additive attention mask.  Fill content is irrelevant because the fill is
never attended.

Because both the block input and the attention context are teacher-fixed,
there is NO sequential dependency between answer tokens and NO dependency
between layers: each layer processes every loss position of a whole cohort in
one batched pass, with exactly one optimizer write per block per cohort.
Whole-answer processing is exact for this objective, not a staleness
approximation — v3's B×K tile machinery does not apply.

Multi-GPU is layer-sharding (``train.v4_stage_splits`` / ``v4_stage_devices``
+ ``scripts/train.py --v4-stage``): independent processes, each loading the
full model on one card and training only its owned contiguous block range.
Lineage: progressive blockwise KD (Wang, Zhao, Li, Tan — IJCAI 2018); see
docs/training_pipeline_v4.md for the differentiators (same-model context
distillation; attention censorship; frozen teacher KV).
"""

from __future__ import annotations

import contextlib
import os
import random
import socket
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..eval.teacher_output import teacher_output_eval_sums
from .blocks import NO_PREPARED_ATTENTION_MASK
from .deepseek_ctx import (DeepseekRecorder, FrozenDeepseekCtx,
                           extended_additive_mask, gather_topk_at_qpos)
from .losses import HiddenLoss
from .moe import pending_router_loss
from .online_v3 import (_bk_bucketed_cohorts, _bk_layer_type,
                        _clear_block_grads, _immediate_sgd)
from .stop import stop_requested
from .telemetry import (
    ParameterDeltaTracker,
    _epoch_end_telemetry,
    _epoch_zero_telemetry,
)


class _FrozenKV:
    """Frozen full-sequence K/V for ONE block, duck-typed as an HF cache.

    Attention calls ``update`` with the freshly projected (and already
    RoPE-rotated) key/value states of the current rows.  In record mode (the
    one-off prefill over the full teacher sequence) those are stored.  In
    frozen mode the incoming projections are DISCARDED and the stored
    full-sequence tensors returned, so queries attend over teacher K/V and no
    gradient can enter through keys or values.
    """

    def __init__(self):
        self.keys = None    # [B, n_kv_heads, T, head_dim]
        self.values = None
        self.recording = True

    def update(self, key_states, value_states, layer_idx=None,
               cache_kwargs=None):
        if self.recording:
            self.keys = key_states.detach()
            self.values = value_states.detach()
            return self.keys, self.values
        if self.keys is None:
            raise RuntimeError("frozen teacher KV consumed before prefill")
        return self.keys, self.values

    # Transformers cache-protocol compatibility surface.
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def to(self, device):
        if self.keys is not None:
            self.keys = self.keys.to(device, non_blocking=True)
            self.values = self.values.to(device, non_blocking=True)
        return self

    def pin(self):
        if self.keys is not None and self.keys.device.type == "cpu":
            self.keys = self.keys.pin_memory()
            self.values = self.values.pin_memory()
        return self

    def staged_to(self, device) -> "_FrozenKV":
        kv = _FrozenKV()
        kv.keys = self.keys.to(device, non_blocking=True)
        kv.values = self.values.to(device, non_blocking=True)
        kv.recording = False
        return kv

    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return (self.keys.numel() + self.values.numel()
                ) * self.keys.element_size()


def _owned_range(cfg, n_layers: int) -> range:
    """One-based inclusive block range this process trains."""
    splits = list(cfg.train.v4_stage_splits or [])
    stage = cfg.train.v4_stage
    if stage < 0:
        return range(1, n_layers + 1)
    bounds = [0] + splits + [n_layers]
    if splits and splits[-1] >= n_layers:
        raise ValueError(
            f"v4_stage_splits {splits} outside 1..{n_layers - 1}")
    return range(bounds[stage] + 1, bounds[stage + 1] + 1)


class _V4Cohort:
    """Layer-independent tensors of one cohort, built once and reused.

    Everything here is teacher-coordinate.  ``qpos`` holds the query rows:
    the union of the training loss positions and the answer-predictor rows
    the CE/KL evaluation needs (the ``answer_offset - 1`` convention of
    ``_bk_answer_eval_coordinates``, so every teacher-realized answer token
    is counted exactly once per epoch).  ``loss_valid`` marks the training
    subset; ``eval_rows``/``eval_ids`` the evaluation subset.
    """

    def __init__(self, cfg, ds, indices: list[int], device):
        self.indices = list(indices)
        pairs = [ds.pairs[i] for i in self.indices]
        self.example_ids = [p.example_id for p in pairs]
        spans = [ds.cache.span(p.example_id) for p in pairs]
        self.t_len = [int(s["n_teacher"]) for s in spans]
        self.T = max(self.t_len)
        B = len(pairs)
        censor = cfg.mask.compaction == "flow_mask"

        keep = torch.zeros((B, self.T), dtype=torch.bool)
        qpos_rows, loss_rows, eval_marks, eval_ids = [], [], [], []
        for b, pair in enumerate(pairs):
            keep[b, : self.t_len[b]] = True
            if censor:
                for start, stop in _priv_ranges(pair):
                    keep[b, start:stop] = False
            t0 = pair.t_aligned.start
            ans0, ans1 = pair.t_answer.start, pair.t_answer.stop
            if cfg.train.v4_loss_positions == "answer":
                loss = list(range(ans0, ans1))
            elif cfg.train.v4_loss_positions == "aligned":
                loss = list(range(t0, pair.t_aligned.stop))
            else:
                raise NotImplementedError(
                    "v4_loss_positions=thinking_answer needs per-record "
                    "thinking-span metadata the dataset does not expose yet")
            if ans0 - 1 < t0:
                raise RuntimeError(
                    f"{pair.example_id}: answer evaluation requires a "
                    "shared_mid predictor row before the answer")
            # Evaluation predictor rows: positions p in
            # [ans0-1, t_aligned.stop-1) predict teacher_ids[p+1] — every
            # answer token exactly once, the _bk convention.
            ev0, ev1 = ans0 - 1, pair.t_aligned.stop - 1
            rows = sorted(set(loss) | set(range(ev0, ev1)))
            index_of = {p: i for i, p in enumerate(rows)}
            qpos_rows.append(rows)
            loss_rows.append([index_of[p] for p in loss])
            eval_marks.append([index_of[p] for p in range(ev0, ev1)])
            eval_ids.append([pair.teacher_ids[p + 1] for p in range(ev0, ev1)])

        self.Q = max(len(r) for r in qpos_rows)
        self.qpos = torch.zeros((B, self.Q), dtype=torch.long)
        self.loss_valid = torch.zeros((B, self.Q), dtype=torch.bool)
        for b, rows in enumerate(qpos_rows):
            self.qpos[b, : len(rows)] = torch.tensor(rows, dtype=torch.long)
            self.loss_valid[b, torch.tensor(loss_rows[b], dtype=torch.long)] = True
        self.eval_rows = [torch.tensor(r, dtype=torch.long) for r in eval_marks]
        self.eval_ids = [torch.tensor(i, dtype=torch.long) for i in eval_ids]
        self.n_eval = sum(len(i) for i in eval_ids)
        self.t0 = torch.tensor([p.t_aligned.start for p in pairs],
                               dtype=torch.long)
        self.teacher_ids = torch.zeros((B, self.T), dtype=torch.long)
        for b, pair in enumerate(pairs):
            self.teacher_ids[b, : len(pair.teacher_ids)] = torch.tensor(
                pair.teacher_ids, dtype=torch.long)
        self.keep = keep
        # Additive mask [B, 1, Q, T]: causal at each query row's own teacher
        # position, privileged and padded keys removed.  Padded query rows
        # (qpos 0) keep position 0 attendable, so no all-masked softmax row.
        k_pos = torch.arange(self.T)[None, None, :]
        allowed = (k_pos <= self.qpos[:, :, None]) & keep[:, None, :]
        self.device = device
        self._mask_cpu = allowed
        self.qpos_dev = self.qpos.to(device)
        self.loss_valid_dev = self.loss_valid.to(device)
        self.cells = int(self.loss_valid.sum())

    def additive_mask(self, dtype, window: int | None = None) -> torch.Tensor:
        allowed = self._mask_cpu.to(self.device)
        if window:
            k_pos = torch.arange(self.T, device=self.device)[None, None, :]
            allowed = allowed & (k_pos > (self.qpos_dev[:, :, None] - int(window)))
        mask = torch.zeros(allowed.shape, dtype=dtype, device=self.device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask[:, None]

    def gather_full_inputs(self, cache, layer: int) -> torch.Tensor:
        """[B, T, ...] padded full-sequence i{layer} (prefill input).
        Trailing dims follow the cached state (mHC boundaries are
        [T, hc_mult, H])."""
        rows = []
        for example_id, t_len in zip(self.example_ids, self.t_len):
            t = cache.teacher_input(example_id, layer)
            if t.shape[0] != t_len:
                raise RuntimeError(
                    f"{example_id}: i{layer:02d} length {t.shape[0]} != "
                    f"index n_teacher {t_len}")
            rows.append(t)
        out = torch.zeros((len(rows), self.T, *rows[0].shape[1:]),
                          dtype=rows[0].dtype)
        for b, t in enumerate(rows):
            out[b, : t.shape[0]] = t
        return out

    def gather_query_inputs(self, full_inputs: torch.Tensor) -> torch.Tensor:
        """[B, Q, ...] block-input rows at the query positions."""
        index = self.qpos.to(full_inputs.device)
        index = index.reshape(*index.shape,
                              *([1] * (full_inputs.dim() - 2)))
        index = index.expand(-1, -1, *full_inputs.shape[2:])
        return full_inputs.gather(1, index)

    def gather_targets(self, cache, layer: int) -> torch.Tensor:
        """[B, Q, ...] teacher h{layer} rows at the query positions."""
        out = None
        for b, example_id in enumerate(self.example_ids):
            h = cache.hidden(example_id, layer)
            if out is None:
                out = torch.zeros(
                    (len(self.example_ids), self.Q, *h.shape[1:]),
                    dtype=h.dtype)
            rel = self.qpos[b] - self.t0[b]
            rel = rel.clamp_(0, h.shape[0] - 1)
            out[b] = h.index_select(0, rel)
        return out


def _priv_ranges(pair) -> list[tuple[int, int]]:
    ranges = list(pair.t_privileged or [])
    if not ranges:
        # Single implicit privileged block between the shared prefix and the
        # aligned span: student coordinates [s0, ...] map to teacher
        # [s_aligned.start, t_aligned.start).
        ranges = [(pair.s_aligned.start, pair.t_aligned.start)]
    return [(int(a), int(b)) for a, b in ranges if int(b) > int(a)]


class _TeacherTensors:
    """Per-(layer, cohort) frozen tensors with residency management.

    ``gpu_corpus`` keeps them on the training device across epochs;
    ``cpu_stream`` keeps them pinned on the host and stages per visit.
    Teacher-frozen KV never changes, so whatever is cached is final until a
    ``student_refresh`` rebuild invalidates it.
    """

    def __init__(self, residency: str, device):
        self.residency = residency
        self.device = device
        self.store: dict[tuple[int, int], dict] = {}

    def get(self, layer: int, cohort_idx: int):
        if self.residency == "rebuild":
            return None
        return self.store.get((layer, cohort_idx))

    def put(self, layer: int, cohort_idx: int, kv: _FrozenKV,
            inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        entry = {"kv": kv, "inputs": inputs, "targets": targets}
        if self.residency == "rebuild":
            return entry  # never stored: rebuilt from the shm cache per visit
        if self.residency == "cpu_stream":
            kv = kv.to("cpu").pin()
            inputs = inputs.cpu().pin_memory()
            targets = targets.cpu().pin_memory()
            entry = {"kv": kv, "inputs": inputs, "targets": targets}
        self.store[(layer, cohort_idx)] = entry
        return entry

    def put_linear(self, layer: int, cohort_idx: int,
                   full_inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        if self.residency == "rebuild":
            return {"full_inputs": full_inputs, "targets": targets}
        if self.residency == "cpu_stream":
            full_inputs = full_inputs.cpu().pin_memory()
            targets = targets.cpu().pin_memory()
        entry = {"full_inputs": full_inputs, "targets": targets}
        self.store[(layer, cohort_idx)] = entry
        return entry

    def staged_linear(self, entry: dict):
        if self.residency != "cpu_stream":
            return entry["full_inputs"], entry["targets"]
        return (entry["full_inputs"].to(self.device, non_blocking=True),
                entry["targets"].to(self.device, non_blocking=True))

    def staged(self, entry: dict):
        if self.residency != "cpu_stream":
            return entry["kv"], entry["inputs"], entry["targets"]
        # Both _FrozenKV and FrozenDeepseekCtx implement staged_to.
        kv = entry["kv"].staged_to(self.device)
        return (kv, entry["inputs"].to(self.device, non_blocking=True),
                entry["targets"].to(self.device, non_blocking=True))

    def drop_layer(self, layer: int) -> None:
        for key in [k for k in self.store if k[0] == layer]:
            del self.store[key]


@torch.no_grad()
def _online_teacher_capture(cfg, stack, adapters_off, cohort, owned,
                            device, n_layers):
    """One adapters-off forward per cohort: teacher states computed by OUR
    runtime instead of read from a stored cache (owner contract: "just keep
    calculating it"). vLLM contributes only answer token ids; hidden states
    always come from this stack, numerically identical to what the builder
    would have stored minus the cache bf16 re-quantization.

    Returns owned-layer transient tensors: inputs {L: [B,T,H]}, targets
    {L: [B,Q,H]}, and (when the final layer is owned) the post-norm teacher
    rows at the eval positions.
    """
    B, T = len(cohort.indices), cohort.T
    ids_full = cohort.teacher_ids.to(device)
    # Batch-chunk the capture forward: a full-attention block over a long
    # cohort materializes O(chunk*heads*T*T) SDPA scores, which OOMs at the
    # whole cohort (B=32, T~4096) on 80 GB. Each item's forward is
    # independent (attention is within-sequence, causal), so a smaller chunk
    # is numerically exact — chunk>=B reproduces the historical single pass.
    chunk = cfg.train.v4_capture_micro_batch or B
    inputs_parts: dict = {layer: [] for layer in owned}
    targets_parts: dict = {layer: [] for layer in owned}
    eval_parts: list = []
    for b0 in range(0, B, chunk):
        b1 = min(b0 + chunk, B)
        ids = ids_full[b0:b1]
        pos = torch.arange(T, device=device)[None].expand(b1 - b0, -1)
        qpos_chunk = cohort.qpos_dev[b0:b1]
        ctx = (adapters_off() if adapters_off is not None
               else contextlib.nullcontext())
        with ctx:
            h = stack.embed(ids)
            pe = stack.rope(h, pos)
            # No prepared mask: with a plain rope, run_block leaves
            # attention_mask None and SDPA's is_causal fast path gives exact
            # causal attention (why the 0.6B online-vs-cache equivalence
            # held). With a gemma4-class rope BUNDLE, run_block instead
            # applies the per-layer-type mask from the bundle — REQUIRED so
            # sliding-window layers' teacher states are windowed, not
            # full-causal. NO_PREPARED here would discard that mask.
            for layer in range(1, n_layers + 1):
                if layer in owned:
                    inputs_parts[layer].append(h.clone())
                h = stack.run_block(layer, h, pe, position_ids=pos,
                                    input_ids=ids)
                if layer in owned:
                    view = h if layer < n_layers else stack.loss_view(
                        n_layers, h)
                    idx = qpos_chunk.reshape(
                        *qpos_chunk.shape, *([1] * (view.dim() - 2)))
                    idx = idx.expand(-1, -1, *view.shape[2:])
                    targets_parts[layer].append(view.gather(1, idx))
                    if layer == n_layers:
                        for bb in range(b0, b1):
                            r = cohort.eval_rows[bb].to(device)
                            positions = cohort.qpos_dev[bb].index_select(0, r)
                            eval_parts.append(
                                view[bb - b0].index_select(0, positions))
    # When capture chunking is enabled (the memory-lean path), offload the
    # per-owned-layer full inputs to host: under item_major, block L's
    # training runs while blocks L+1..end of the owned range still hold their
    # [B,T,H] capture inputs on the card. On gemma-4-26B the first owned
    # block is a full_attention layer, so its training peak coincides with
    # all owned inputs resident (~6 GB) and OOMs an 80 GB card by a few
    # hundred MB. build_layer_cohort streams the one needed layer back per
    # visit. Epoch 1 only (epochs 2+ read the cpu_stream store, capture=None).
    offload = bool(cfg.train.v4_capture_micro_batch)
    inputs = {layer: (torch.cat(inputs_parts[layer], 0).to("cpu")
                      if offload else torch.cat(inputs_parts[layer], 0))
              for layer in owned}
    targets = {layer: torch.cat(targets_parts[layer], 0) for layer in owned}
    eval_rows_teacher = eval_parts if n_layers in owned else None
    return {"inputs": inputs, "targets": targets,
            "eval_rows_teacher": eval_rows_teacher}


def _resolve_residency(cfg, cohorts, ds, stack, owned) -> str:
    # online source uses the SAME store and sizing: epoch-1 captures are
    # retained per residency (cache-after-first-production) or re-captured
    # each epoch under "rebuild" — the owner's calibration point. The
    # measured capture seconds land in the v4_epoch prep split.
    if cfg.train.v4_teacher_residency != "auto":
        return cfg.train.v4_teacher_residency
    hidden = stack.text_config.hidden_size
    n_kv = getattr(stack.text_config, "num_key_value_heads", None) or \
        stack.text_config.num_attention_heads
    head_dim = getattr(stack.text_config, "head_dim", None) or (
        hidden // stack.text_config.num_attention_heads)
    total_positions = sum(
        int(ds.cache.span(p.example_id)["n_teacher"]) for p in ds.pairs)
    # mHC boundaries carry hc_mult streams: inputs/targets scale with it.
    hc = int(getattr(stack.text_config, "hc_mult", 0) or 1)
    per_layer = total_positions * (2 * n_kv * head_dim
                                   + 2 * hidden * max(hc, 1)) * 2
    # BOTH loop orders accumulate every owned layer's tensors in the store
    # across the epoch (that persistence IS the epoch-2 speedup), so the
    # honest requirement is all owned layers. Sizing for one resident layer
    # OOM'd the 27B elephant (54 GB weights + 16 accumulating layers).
    needed = per_layer * len(owned)
    free, _total = torch.cuda.mem_get_info(
        torch.device(cfg.model.device))
    if needed < 0.5 * free:
        return "gpu_corpus"
    # Host check: pinned staging must fit beside the sibling stages. At the
    # 27B full corpus this is ~480 GB per stage — pinning that much kills
    # the node; fall back to rebuild-per-visit (one extra forward per
    # (layer, cohort); the shm cache serves the reads at RAM speed).
    stages = max(len(cfg.train.v4_stage_splits or []) + 1, 1)
    with open("/proc/meminfo") as fh:
        available_kb = next(
            int(line.split()[1]) for line in fh
            if line.startswith("MemAvailable"))
    if needed < 0.3 * available_kb * 1024 / stages:
        return "cpu_stream"
    return "rebuild"



def _student_ids(ds, cohort) -> torch.Tensor:
    """[B, T] censored student token ids of one cohort (flow_mask: same
    length as the teacher sequence)."""
    B, T = len(cohort.indices), cohort.T
    ids = torch.zeros((B, T), dtype=torch.long)
    for b, i in enumerate(cohort.indices):
        pair = ds.pairs[i]
        sid = torch.tensor(pair.student_ids, dtype=torch.long)
        if sid.shape[0] != cohort.t_len[b]:
            raise RuntimeError(
                f"{pair.example_id}: flow_mask student sequence "
                f"length {sid.shape[0]} != teacher {cohort.t_len[b]}")
        ids[b, : sid.shape[0]] = sid
    return ids


@torch.no_grad()
def _relay_boundary_h(cfg, stack, ds, cohort, boundaries_in, idx, device):
    if boundaries_in is not None:
        return boundaries_in[idx].to(device)
    return stack.embed(_student_ids(ds, cohort).to(device))


@torch.no_grad()
def _relay_segment(cfg, stack, ds, cohorts, device, owned,
                   boundaries_in: dict | None, rotator=None) -> dict:
    """Run this stage's owned blocks of the CENSORED student forward.

    ``boundaries_in`` maps cohort index -> [B, T, H] hidden states at the
    stage boundary (None = first stage, which embeds the student ids).
    Returns the same mapping at this stage's output boundary.  This is the
    deployment-matched walk: flow attention mask, full causal sequence,
    the student's own states — never teacher tensors.

    With a ``rotator`` (weights paged per layer) the walk is LAYER-outer:
    cohort-outer would page the whole owned shard once per cohort (~4 TB
    per relay at 397B). All cohorts' boundary hiddens stay resident
    instead (~10 GB at 27B scale) — the cheap side of that trade.
    """
    deepseek = getattr(stack, "needs_deepseek_masks", False)
    out = {}
    if rotator is None:
        for idx, cohort in enumerate(cohorts):
            B, T = len(cohort.indices), cohort.T
            keep = cohort.keep.to(device)
            pos = torch.arange(T, device=device)[None].expand(B, -1)
            h = _relay_boundary_h(cfg, stack, ds, cohort, boundaries_in,
                                  idx, device)
            # Hash-MoE routing in the relay uses the STUDENT (censored)
            # ids — this is the deployment walk.
            ids = (_student_ids(ds, cohort).to(device) if deepseek else None)
            pe = stack.rope(h, pos)
            for layer in owned:
                h = stack.run_block(layer, h, pe, position_ids=pos,
                                    flow_keep=keep, causal_length=T,
                                    input_ids=ids)
            # Stream each cohort's boundary to host immediately: holding all
            # cohorts' finals on the card (~30-50 GB at 26B/31B) is what
            # pushed stage 0 over the edge on 2026-07-18 (with the missing
            # no_grad compounding it). Consumers (.to(device) in the eval
            # tail, .cpu() in the envelope write) already accept host
            # tensors.
            out[idx] = h.cpu()
        return out
    hs, keeps, poss, idss = {}, {}, {}, {}
    for idx, cohort in enumerate(cohorts):
        B, T = len(cohort.indices), cohort.T
        keeps[idx] = cohort.keep.to(device)
        poss[idx] = torch.arange(T, device=device)[None].expand(B, -1)
        hs[idx] = _relay_boundary_h(cfg, stack, ds, cohort, boundaries_in,
                                    idx, device)
        idss[idx] = (_student_ids(ds, cohort).to(device)
                     if deepseek else None)
    owned_list = list(owned)
    for pos_i, layer in enumerate(owned_list):
        rotator.activate(layer)
        if pos_i + 1 < len(owned_list):
            rotator.prefetch(owned_list[pos_i + 1])
        for idx, cohort in enumerate(cohorts):
            pe = stack.rope(hs[idx], poss[idx])
            hs[idx] = stack.run_block(
                layer, hs[idx], pe, position_ids=poss[idx],
                flow_keep=keeps[idx], causal_length=cohort.T,
                input_ids=idss[idx])
        rotator.evict(layer)
    return hs


@torch.no_grad()
def _relay_eval_tail(cfg, stack, ds, cohorts, cache, device, log,
                     epoch: int, finals: dict, trajectory: str,
                     serviced_at_epoch: int | None = None,
                     teacher_rows_by_cohort: dict | None = None) -> None:
    """Frozen-head CE/KL over the answer-predictor rows of the final states."""
    n = stack.n_layers
    ce = torch.zeros((), dtype=torch.float64, device=device)
    kl = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    for idx, cohort in enumerate(cohorts):
        view = stack.loss_view(n, finals[idx].to(device))
        rows_v, rows_t, row_ids = [], [], []
        stashed = (teacher_rows_by_cohort or {}).get(idx)
        for b, example_id in enumerate(cohort.example_ids):
            r = cohort.eval_rows[b]
            positions = cohort.qpos[b].index_select(0, r)
            rows_v.append(view[b].index_select(0, positions.to(device)))
            if stashed is not None:
                rows_t.append(stashed[b].to(device))
            else:
                teacher_h = cache.hidden(example_id, n)
                rel = (positions - cohort.t0[b]).clamp_(
                    0, teacher_h.shape[0] - 1)
                rows_t.append(teacher_h.index_select(0, rel).to(device))
            row_ids.append(cohort.eval_ids[b])
        c, k, cnt = teacher_output_eval_sums(
            torch.cat(rows_v).detach().float(),
            torch.cat(rows_t).detach().float(),
            torch.cat(row_ids).to(device), stack.lm_head)
        ce += c.double()
        kl += k.double()
        count += cnt
    log.log(
        kind="student_trajectory_eval", epoch=epoch,
        CE_eval_loss=float(ce.item() / max(count, 1)),
        KL_eval_loss=float(kl.item() / max(count, 1)),
        answer_token_count=count,
        dataset_item_count=len(ds.pairs),
        dataset_coverage="whole_training_set_once_per_call",
        token_coverage="every_teacher_realized_answer_token",
        answer_only=True,
        evaluation_only=True,
        validation_subset=False,
        used_for_backward=False,
        optimizer_weight=0.0,
        aggregation="token_weighted_mean",
        trajectory=trajectory,
        serviced_at_epoch=serviced_at_epoch,
        CE_target="teacher_realized_answer_token_ids",
        KL_direction="teacher_to_student",
        vocabulary_head="frozen",
    )


@torch.no_grad()
def _student_trajectory_eval(cfg, stack, ds, cohorts, cache, device, log,
                             epoch: int,
                             teacher_rows_by_cohort: dict | None = None,
                             rotator=None) -> None:
    """Single-process deployment-matched CE/KL: whole walk in one call.
    Under rotary PPP1 the rotator pages each block in for its layer-outer
    pass — without it the walk would touch CPU-resident masters."""
    finals = _relay_segment(cfg, stack, ds, cohorts, device,
                            range(1, stack.n_layers + 1), None,
                            rotator=rotator)
    _relay_eval_tail(cfg, stack, ds, cohorts, cache, device, log, epoch,
                     finals, trajectory="student_censored_flow_full_walk",
                     teacher_rows_by_cohort=teacher_rows_by_cohort)



def _launch_identity() -> str:
    """Identity of THIS coordinated launch, shared by all its stages.

    The launcher exports SELFUPDATE_V4_LAUNCH_ID to every stage; a
    single-process run mints its own. Every relay/adapter file is stamped
    with it and consumers REFUSE a mismatch: on a shared machine, a stale
    stage from an aborted set must never feed tensors into a newer set
    (owner defect-class report, 2026-07-17 — the hard-killed stale stage 0
    could have done exactly this).
    """
    return os.environ.get("SELFUPDATE_V4_LAUNCH_ID", f"solo-{os.getpid()}")


class _RelayFiles:
    """Atomic tensor-file exchange between v4 stage processes.

    Everything goes through the shared run directory (Lustre or /dev/shm —
    wherever runs/ lives): write to a sibling .tmp, rename into place, poll
    for existence on the consumer side.  This is the same publish discipline
    as the node-epoch0 cache.  A future cross-machine stage set only changes
    WHERE this directory lives (InfiniBand-backed instead of local), nothing
    else — see docs/training_pipeline_v4.md, future scale-out.

    Provenance: every file carries safetensors metadata
    {launch_id, producer_stage, epoch}; ``read`` asserts the launch_id.
    """

    def __init__(self, base_dir: Path):
        root = os.environ.get("SELFUPDATE_V4_RELAY_ROOT")
        if root:
            # Full-corpus boundaries are ~GBs per stage per epoch: exchange
            # them through node-local RAM, never Lustre. Consumers delete
            # consumed files, so the footprint stays ~2 epochs in flight.
            self.dir = Path(root) / Path(base_dir).name / "relay"
        else:
            self.dir = Path(base_dir) / "relay"
        self.launch_id = _launch_identity()

    def path(self, epoch: int, name: str) -> Path:
        return self.dir / f"e{epoch:04d}" / name

    def write(self, path: Path, tensors: dict, *, stage: int, epoch: int,
              to_stage: int | None = None) -> None:
        """Post a tensor file with a full envelope.

        The envelope is the postal address of the exchange (owner metaphor,
        2026-07-17): FROM host+stage of THIS launch, TO the addressee stage
        (None = broadcast, e.g. adapter publications any stage may read),
        for one epoch of one run. The cross-machine (InfiniBand) relay of
        the scale-out plan keeps this envelope unchanged — only the
        directory moves.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        save_file(tensors, str(tmp), metadata={
            "launch_id": self.launch_id,
            "from_host": socket.gethostname(),
            "from_stage": str(stage),
            "to_stage": "broadcast" if to_stage is None else str(to_stage),
            "epoch": str(epoch),
        })
        tmp.rename(path)

    def read(self, path: Path, *, expect_epoch: int | None = None,
             as_stage: int | None = None) -> dict:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
            if meta.get("launch_id") != self.launch_id:
                raise RuntimeError(
                    f"relay envelope mismatch at {path}: from launch "
                    f"{meta.get('launch_id')!r} (host "
                    f"{meta.get('from_host')!r}, stage "
                    f"{meta.get('from_stage')!r}), this process is launch "
                    f"{self.launch_id!r} — a stale stage from another launch "
                    "is writing into this run's exchange")
            addressee = meta.get("to_stage")
            if (as_stage is not None and addressee not in
                    ("broadcast", str(as_stage))):
                raise RuntimeError(
                    f"relay envelope at {path} is addressed to stage "
                    f"{addressee!r}, but stage {as_stage} tried to read it")
            if (expect_epoch is not None
                    and meta.get("epoch") != str(expect_epoch)):
                raise RuntimeError(
                    f"relay envelope at {path} carries epoch "
                    f"{meta.get('epoch')!r}, expected {expect_epoch}")
            return {key: handle.get_tensor(key) for key in handle.keys()}

    def wait(self, path: Path, timeout_s: float = 3600.0,
             poll_s: float = 2.0) -> Path:
        deadline = time.time() + timeout_s
        while not path.exists():
            if time.time() > deadline:
                raise RuntimeError(
                    f"relay timeout after {timeout_s:.0f}s waiting for "
                    f"{path}; a stage died or stalled — inspect its log")
            time.sleep(poll_s)
        return path


class _RelayServicer:
    """Non-blocking student-trajectory relay for one stage process.

    The blocking design serialized every stage behind the slowest one
    (stage 0's eval battery) — measured as whole-node ~8% while two cards
    idled at the barrier (2026-07-17). Now each epoch boundary SUBMITS the
    relay and immediately returns to training; pending relays are serviced
    whenever their predecessor boundary has arrived, and drained (blocking)
    only after the final epoch so the last CE/KL always lands.

    Consequence, stated not hidden: a stage may service epoch e's relay
    after it has trained past e, so the segment runs on slightly newer
    weights. The skew is bounded by pipeline depth, evaluation-only, and
    recorded in the eval row as ``serviced_at_epoch`` per the owner's sync
    contract ("until the next gpu has already trained at that level").
    """

    def __init__(self, cfg, stack, ds, cohorts, cache, device, log,
                 run_dir: Path, owned, teacher_eval_rows: dict | None = None,
                 rotator=None):
        self.rotator = rotator
        # Keep the SHARED reference: an empty dict is falsy, so `... or {}`
        # would swap in a fresh dict and sever the link to the training loop's
        # `teacher_eval_rows`, which is populated cohort-by-cohort DURING the
        # run. That severance made the last stage's CE/KL relay fall back to
        # `cache.hidden` and die on the index-only online cache (2026-07-17).
        self.teacher_eval_rows = (teacher_eval_rows
                                  if teacher_eval_rows is not None else {})
        self.cfg, self.stack, self.ds = cfg, stack, ds
        self.cohorts, self.cache = cohorts, cache
        self.device, self.log, self.owned = device, log, owned
        self.stage = cfg.train.v4_stage
        self.stages = len(cfg.train.v4_stage_splits or []) + 1
        self.rf = _RelayFiles(run_dir.parent)
        self.pending: list[int] = []
        self.trained_epochs = 0

    def submit(self, epoch: int) -> None:
        self.trained_epochs = max(self.trained_epochs, epoch)
        if self.stage == 0:
            # Producer: own segment starts from embeddings — no wait ever.
            self._produce(epoch, None)
        else:
            self.pending.append(epoch)
            self.service(block=False)

    def service(self, block: bool = False) -> None:
        while self.pending:
            epoch = self.pending[0]
            path = self.rf.path(epoch, f"stage{self.stage - 1}.st")
            if not path.exists():
                if not block:
                    return
                self.rf.wait(path)
            loaded = self.rf.read(path, expect_epoch=epoch,
                                  as_stage=self.stage)
            boundaries = {int(k[1:]): v for k, v in loaded.items()}
            self._produce(epoch, boundaries)
            # Consumed: delete our input so an infinite run's relay/ stays
            # bounded (every file has exactly one addressee).
            path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                path.parent.rmdir()
            self.pending.pop(0)

    def drain(self) -> None:
        """Best-effort: a dead or stopped predecessor must never cost this
        stage its trained adapters. The 2026-07-18 e500 finale lost the
        500-epoch adapters of three stages exactly this way — stage 0
        stopped gracefully at 311, its siblings trained to 500, then
        blocked here for their pending relay evals and were killed by the
        wait timeout BEFORE certification/checkpoint. Relay evals are
        telemetry; checkpoints are the run. Abandon, log, proceed."""
        try:
            self.service(block=True)
        except RuntimeError as err:
            self.log.log(kind="relay_drain_abandoned",
                         pending_epochs=list(self.pending),
                         error=str(err)[:300])
            self.pending.clear()

    def _produce(self, epoch: int, boundaries) -> None:
        out = _relay_segment(self.cfg, self.stack, self.ds, self.cohorts,
                             self.device, self.owned, boundaries,
                             rotator=self.rotator)
        if self.stage == self.stages - 1:
            _relay_eval_tail(self.cfg, self.stack, self.ds, self.cohorts,
                             self.cache, self.device, self.log, epoch, out,
                             trajectory="student_censored_flow_staged_relay",
                             serviced_at_epoch=self.trained_epochs,
                             teacher_rows_by_cohort=self.teacher_eval_rows)
        else:
            self.rf.write(self.rf.path(epoch, f"stage{self.stage}.st"),
                          {f"c{idx}": t.detach().cpu()
                           for idx, t in out.items()},
                          stage=self.stage, epoch=epoch,
                          to_stage=self.stage + 1)


def _owned_adapter_tensors(stack, owned) -> dict:
    """This stage's trainable parameters, keyed stably by block + local name."""
    tensors = {}
    for layer in owned:
        block = stack.blocks[layer - 1]
        for name, param in block.named_parameters():
            if param.requires_grad:
                tensors[f"L{layer:03d}.{name}"] = param.detach().cpu()
    return tensors


def _subprocess_battery(cfg, stack, log, epoch: int, run_dir: Path,
                        owned, baseline, rotator=None):
    """Plan B6: every stage publishes adapters and releases VRAM; stage 0
    spawns scripts/v4_battery.py (full model, device_map=auto over all
    cards, existing telemetry probes) and signals done; stages resume."""
    import os
    import subprocess
    import sys as _sys

    # Single-process (rotary PPP1) normalizes to stage 0: it publishes,
    # spawns, and has no siblings to ack/notify.
    stage = max(cfg.train.v4_stage, 0)
    stages = len(cfg.train.v4_stage_splits or []) + 1
    rf = _RelayFiles(run_dir.parent)
    rf.write(rf.path(epoch, f"adapters_stage{stage}.st"),
             _owned_adapter_tensors(stack, owned), stage=stage, epoch=epoch)
    if rotator is not None:
        for L in list(rotator._staged):
            rotator.evict(L)
    torch.cuda.empty_cache()
    rf.write(rf.path(epoch, f"battery_ack_stage{stage}.st"),
             {"ack": torch.zeros(1)}, stage=stage, epoch=epoch)
    if stage != 0:
        rf.wait(rf.path(epoch, "battery_done.st"))
        return baseline
    for k in range(1, stages):
        rf.wait(rf.path(epoch, f"battery_ack_stage{k}.st"))
    paths = os.environ.get("SELFUPDATE_V4_CONFIG", "")
    base_p, _, exp_p = paths.partition("::")
    if not base_p or not exp_p:
        raise RuntimeError(
            "v4_battery_mode=subprocess needs SELFUPDATE_V4_CONFIG="
            "<base>::<experiment> in the environment (scripts/train.py "
            "exports it)")
    script = Path(__file__).resolve().parents[3] / "scripts" / "v4_battery.py"
    logf = run_dir / f"battery_e{epoch:04d}.log"
    child_env = dict(os.environ)
    if cfg.train.v4_stage < 0:
        # Rotary PPP1: the battery child must stay on the rotor's OWN
        # card — device_map=auto over every GPU tramples concurrent runs
        # (three rotors + a PPP5 stage shared one node, 2026-07-18).
        # Inside the child, auto-placement spills the remainder to CPU.
        own = torch.device(cfg.model.device).index or 0
        child_env["CUDA_VISIBLE_DEVICES"] = str(own)
    with open(logf, "ab") as fh:
        rc = subprocess.run(
            [_sys.executable, str(script), "--config", base_p,
             "--experiment", exp_p, "--run-dir", str(run_dir),
             "--epoch", str(epoch), "--stages", str(stages)],
            stdout=fh, stderr=fh, env=child_env).returncode
    if rc != 0:
        raise RuntimeError(
            f"battery subprocess failed rc={rc}; see {logf} — the "
            "per-epoch battery is non-negotiable, a run without it "
            "must not continue silently")
    rf.write(rf.path(epoch, "battery_done.st"), {"done": torch.zeros(1)},
             stage=0, epoch=epoch)
    return baseline


def _staged_epoch_battery(cfg, stack, tok, log, epoch: int, run_dir: Path,
                          owned, baseline, started_at: float,
                          rotator=None):
    """Owner-mandated per-epoch battery in staged mode.

    Every stage publishes its owned adapter tensors; stage 0 waits for all
    of them, grafts the foreign-block adapters onto its own full model
    (harmless for its training — v4 never reads foreign blocks), and runs
    the SAME recall/standard-damage probes as v3.  Other stages return
    immediately and keep training.
    """
    if cfg.train.v4_battery_mode == "subprocess":
        return _subprocess_battery(cfg, stack, log, epoch, run_dir,
                                   owned, baseline, rotator=rotator)
    stage = cfg.train.v4_stage
    stages = len(cfg.train.v4_stage_splits or []) + 1
    rf = _RelayFiles(run_dir.parent)
    if cfg.train.v4_stage_scoped:
        # Graft mode requires stage 0 to hold the FULL model; under stage-
        # scoped loading foreign blocks are meta. validate steers scoped
        # runs to v4_battery_mode=subprocess; this loud row remains as the
        # last-resort marker — a skipped battery must never be mistaken
        # for a run without the obligation.
        if stage == 0:
            log.log(kind="epoch_battery_skipped", epoch=epoch,
                    reason="v4_stage_scoped_graft_impossible",
                    owner_law="per-epoch battery is NON-NEGOTIABLE; this "
                              "row marks debt, not permission")
        return baseline
    if stage != 0:
        # Stage 0 is the only consumer; it needs no copy of its own.
        rf.write(rf.path(epoch, f"adapters_stage{stage}.st"),
                 _owned_adapter_tensors(stack, owned), stage=stage,
                 epoch=epoch)
        return baseline
    n = stack.n_layers
    with torch.no_grad():
        for other in range(1, stages):
            path = rf.wait(rf.path(epoch, f"adapters_stage{other}.st"))
            grafted = rf.read(path, expect_epoch=epoch, as_stage=0)
            path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                path.parent.rmdir()
            for key, value in grafted.items():
                layer_tag, _, local = key.partition(".")
                layer = int(layer_tag[1:])
                if layer in owned:
                    raise RuntimeError(
                        f"stage {other} published block {layer}, owned here")
                params = dict(stack.blocks[layer - 1].named_parameters())
                params[local].copy_(value.to(params[local].device))
    return _epoch_end_telemetry(cfg, stack, tok, log, epoch=epoch - 1,
                                baseline=baseline, started_at=started_at)


def train_online_v4(cfg, stack, tok, log, cache, peft_model=None,
                    run_dir: Path | None = None) -> bool:
    """Run the v4 walk.  Returns True when stopped cooperatively."""
    if cfg.train.pipeline_version != 4:
        raise ValueError("train_online_v4 requires pipeline_version=4")
    if cfg.train.max_steps:
        raise NotImplementedError(
            "pipeline-v4 has no step cap; bound work with epochs")
    if cfg.train.batching != "bucketed":
        raise NotImplementedError(
            "pipeline-v4 cohorts are length-bucketed; set batching=bucketed")
    from ..data.dataset import DistillDataset

    ds = DistillDataset(
        cfg.data.examples_path, cache, tok,
        need_layers=[],
        with_teacher_ids=False,
        pad_random=False,
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        item_cache_items=cfg.cache.item_cache_items,
    )
    online_source = cfg.train.v4_teacher_source == "online"
    store_source = cfg.train.v4_teacher_source == "store"
    if (cfg.train.v4_teacher_source == "cache"
            and not cache.has_full_teacher_inputs):
        raise ValueError(
            "pipeline-v4 cache source needs "
            "cache.store_full_teacher_inputs=true; or set "
            "v4_teacher_source=online/store (index-only cache)")
    teacher_eval_rows: dict = {}
    n = stack.n_layers
    owned = _owned_range(cfg, n)
    # Frozen-KV contract tripwire: KV-sharing layers (gemma-class
    # num_kv_shared_layers > 0) read another layer's KV through the
    # shared_kv_states side channel and never call past_key_values.update,
    # so _FrozenKV would stay empty and fail later with a generic consume
    # error. The 2026-07 gemma-4 targets set 0; fail loudly and by name if
    # a future variant does not.
    shared_kv = int(getattr(stack.text_config, "num_kv_shared_layers", 0)
                    or 0)
    if shared_kv:
        raise NotImplementedError(
            f"pipeline-v4 frozen teacher KV does not support KV-sharing "
            f"layers yet (num_kv_shared_layers={shared_kv}): the shared "
            f"layers bypass past_key_values.update; a shared-kv arm of "
            f"_FrozenKV is required")
    device = torch.device(cfg.model.device)
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    B = cfg.train.micro_batch

    # Fixed cohort composition and row order; only the visit order shuffles
    # per epoch.  Within-cohort order is irrelevant to numerics here — the
    # update is one summed write per block per cohort — and a fixed
    # composition is what lets frozen per-cohort KV persist across epochs.
    cohort_indices = _bk_bucketed_cohorts(ds, B, cfg.train.seed)
    cohorts = [
        _V4Cohort(cfg, ds, indices, device) for indices in cohort_indices]
    residency = _resolve_residency(cfg, cohorts, ds, stack, owned)
    if store_source and residency == "rebuild":
        # Capture-once data has no per-epoch recapture to rebuild from;
        # validate already rejects an EXPLICIT rebuild — this guards the
        # auto policy resolving there under memory pressure.
        residency = "cpu_stream"
    tensors = _TeacherTensors(residency, device)

    # Frozen teacher projections = LoRA adapters disabled. The PEFT handle
    # comes from the runtime; a full-FT v4 (no adapters) would need a frozen
    # teacher copy instead, which validate.py currently rejects.
    if cfg.train.lora.enabled and peft_model is None:
        raise ValueError(
            "pipeline-v4 with LoRA needs the runtime's peft_model handle to "
            "compute adapters-off teacher projections")
    adapters_off = peft_model.disable_adapter if peft_model is not None else None

    # MoE routing intervention (teacher_forced / router_aligned): the wrapped
    # routers record teacher top-k during the SAME adapters-off capture that
    # produces the teacher hiddens, and replay/align during the student pass.
    # dense_or_black_box needs nothing: the block runs whole with the
    # student's own routing (experts are nn.Parameter — LoRA never touches
    # them, and the router Linear is not a LoRA target).
    moe_ctrl = None
    moe_routing: dict[int, dict] = {}  # cohort_idx -> {"idx": {L: T}, "logp": {L: T}}
    if cfg.train.moe_mode != "dense_or_black_box":
        from .moe import MoEController
        moe_ctrl = MoEController(stack, cfg.train.moe_mode,
                                 cfg.train.moe_router_weight)

    optimizers: dict[int, torch.optim.AdamW] = {}
    if cfg.train.v4_optimizer == "adam":
        for layer in owned:
            params = [p for p in stack.block_params(layer) if p.requires_grad]
            optimizers[layer] = torch.optim.AdamW(
                params, lr=cfg.train.lr,
                betas=tuple(cfg.train.v4_adam_betas),
                eps=cfg.train.v4_adam_eps,
                weight_decay=cfg.train.v4_adam_weight_decay)
        if cfg.train.init_from:
            # Warm start restores momentum (plan B0). Missing file = cold
            # moments, logged not raised; a param-group mismatch raises
            # loudly inside load_state_dict.
            mpath = (Path("runs") / cfg.train.init_from / "checkpoint"
                     / "adam_moments.pt")
            if mpath.exists():
                saved = torch.load(mpath, map_location="cpu",
                                   weights_only=False)
                restored = [L for L in optimizers if L in saved]
                for L in restored:
                    optimizers[L].load_state_dict(saved[L])
                log.log(kind="v4_adam_moments", action="restored",
                        layers=sorted(restored),
                        missing_layers=sorted(
                            L for L in optimizers if L not in saved))
            else:
                log.log(kind="v4_adam_moments", action="cold_start",
                        reason=f"no {mpath}")

    # Weight rotation (plan B4): when the stage-scoped load left owned
    # frozen block weights on host (v4_weight_residency rotate, or auto
    # deciding rotate), page each block's rotation unit — frozen weights
    # one-way, Adam moments both ways — per layer_major visit. An empty
    # masters map means the runtime placed everything resident.
    rotator = None
    if cfg.train.v4_stage_scoped:
        from .rotation import BlockRotator
        rot = BlockRotator(stack, owned, device, optimizers)
        if rot.masters:
            rotator = rot
            log.log(kind="v4_rotation",
                    rotated_blocks=sorted(rot.masters),
                    rotated_bytes=sum(
                        t.numel() * t.element_size()
                        for e in rot.masters.values() for t in e.values()),
                    moments_rotate=cfg.train.v4_optimizer == "adam")

    log.log(
        kind="pipeline_v4_contract",
        objective="student_block_L(teacher_h[L-1]) vs teacher_h[L]",
        attention_context="frozen_teacher_kv_full_sequence",
        linear_attention_rule=(
            "full_sequence_teacher_forced_own_recurrence_flow_censored"),
        sliding_attention_rule="frozen_teacher_kv_windowed_mask",
        kv_gradient="none_query_side_only",
        kv_source=cfg.train.v4_kv_source,
        censorship="privileged_keys_removed_from_attention",
        loss_positions=cfg.train.v4_loss_positions,
        update_law="one_write_per_block_per_cohort_unaveraged_sum",
        loop_order=cfg.train.v4_loop_order,
        teacher_residency=residency,
        owned_blocks=[owned.start, owned.stop - 1],
        v4_stage=cfg.train.v4_stage,
        optimizer=cfg.train.v4_optimizer,
        cohorts=len(cohorts),
        dataset_items=len(ds.pairs),
    )

    def build_layer_cohort(layer: int, cohort_idx: int,
                           capture: dict | None = None) -> dict:
        cohort = cohorts[cohort_idx]
        entry = tensors.get(layer, cohort_idx)
        if entry is not None:
            return entry
        if store_source:
            raise RuntimeError(
                f"store mode: no captured entry for layer {layer} cohort "
                f"{cohort_idx} — the capture relay must prefill every "
                "(owned layer, cohort) pair; a dropped entry means the "
                "residency policy evicted capture-once data")
        layer_type = _bk_layer_type(stack, layer)
        if capture is not None:
            # Capture inputs may live on host (memory-lean offload path);
            # stream the one needed owned layer back to the card. .to(device)
            # is a no-op when already resident. Drop the host reference so
            # sibling owned layers' inputs are the only ones left staged.
            full_inputs = capture["inputs"].pop(layer).to(device)
            targets = capture["targets"].pop(layer).to(device)
        else:
            full_inputs = cohort.gather_full_inputs(cache, layer).to(device)
            targets = cohort.gather_targets(cache, layer).to(device)
        if layer_type == "linear_attention":
            # Recurrent mixers have no K/V to freeze: the layer runs the
            # FULL teacher-forced sequence with its own (trainable)
            # recurrence, censored by flow_keep row-zeroing. Store the full
            # inputs; no prefill.
            return tensors.put_linear(layer, cohort_idx, full_inputs, targets)
        if getattr(stack, "needs_deepseek_masks", False):
            # DeepSeek-V4 record pass: real typed cache layers run the
            # compressor's genuine window arithmetic; the indexer's top-k is
            # captured by hook.  Fresh recorder PER CHUNK — the typed cache
            # treats successive calls as time-continuation, chunks are batch
            # slices.  Mask-free is exact for everything recorded: sliding
            # K=V and compressed entries are projections of the input, and
            # the indexer applies its own causal mask internally.
            Bc = len(cohort.indices)
            chunk = cfg.train.v4_capture_micro_batch or Bc
            ids_full = cohort.teacher_ids.to(device)
            ctx = (contextlib.nullcontext() if adapters_off is None
                   else adapters_off())
            kv_parts, entry_parts, topk_parts = [], [], []
            with torch.no_grad(), ctx:
                for b0 in range(0, Bc, chunk):
                    b1 = min(b0 + chunk, Bc)
                    fi = full_inputs[b0:b1]
                    pos = torch.arange(cohort.T, device=device)[None].expand(
                        b1 - b0, -1)
                    rec = DeepseekRecorder(stack, layer)
                    try:
                        stack.run_block(
                            layer, fi, stack.rope(fi, pos), position_ids=pos,
                            past_key_values=rec.shim, use_cache=True,
                            prepared_attention_mask=NO_PREPARED_ATTENTION_MASK,
                            input_ids=ids_full[b0:b1])
                    finally:
                        rec.close()
                    kv_c, entries_c, topk_c = rec.harvest()
                    kv_parts.append(kv_c)
                    entry_parts.append(entries_c)
                    topk_parts.append(gather_topk_at_qpos(
                        topk_c, cohort.qpos_dev[b0:b1]))
            frozen = FrozenDeepseekCtx(
                torch.cat(kv_parts, 0) if len(kv_parts) > 1 else kv_parts[0],
                (None if entry_parts[0] is None else
                 (torch.cat(entry_parts, 0) if len(entry_parts) > 1
                  else entry_parts[0])),
                (None if topk_parts[0] is None else
                 (torch.cat(topk_parts, 0) if len(topk_parts) > 1
                  else topk_parts[0])),
                layer - 1)
            inputs_q = cohort.gather_query_inputs(full_inputs)
            del full_inputs
            return tensors.put(layer, cohort_idx, frozen, inputs_q, targets)
        kv = _FrozenKV()
        Bc = len(cohort.indices)
        # Batch-chunk the prefill for the same reason as the capture: the
        # block still runs its full attention + MoE over the whole teacher
        # sequence (only the projected K/V survive; the output is discarded),
        # so a full-attention gemma4 block over the longest cohort (B=32,
        # T~5070) OOMs an 80 GB card. The stored K/V are [B, n_kv, T, hd] and
        # per-item independent, so building them in chunks and concatenating
        # along the batch dim is numerically exact. chunk>=Bc = one pass.
        chunk = cfg.train.v4_capture_micro_batch or Bc
        refresh = cfg.train.v4_kv_source == "student_refresh"
        ctx = (contextlib.nullcontext() if refresh or adapters_off is None
               else adapters_off())
        key_parts, val_parts = [], []
        with torch.no_grad(), ctx:
            for b0 in range(0, Bc, chunk):
                b1 = min(b0 + chunk, Bc)
                fi = full_inputs[b0:b1]
                pos = torch.arange(cohort.T, device=device)[None].expand(
                    b1 - b0, -1)
                rope_c = stack.rope(fi, pos)
                kv_c = _FrozenKV()
                # Mask-free fast path: the prefill's attention OUTPUT is
                # discarded — only the K/V stored at update() matter, and
                # they are projected from the input before any attention
                # math. The causal_length path would materialize a
                # [B,1,T,T] additive mask; the sentinel avoids it.
                stack.run_block(
                    layer, fi, rope_c, position_ids=pos,
                    past_key_values=kv_c, use_cache=True,
                    prepared_attention_mask=NO_PREPARED_ATTENTION_MASK)
                key_parts.append(kv_c.keys)
                val_parts.append(kv_c.values)
        kv.keys = torch.cat(key_parts, 0) if len(key_parts) > 1 else key_parts[0]
        kv.values = (torch.cat(val_parts, 0)
                     if len(val_parts) > 1 else val_parts[0])
        kv.recording = False
        inputs_q = cohort.gather_query_inputs(full_inputs)
        del full_inputs
        return tensors.put(layer, cohort_idx, kv, inputs_q, targets)

    def moe_student_ctx(layer: int, cohort_idx: int, row_map, row_mask):
        """Arm the controller for ONE owned MoE layer's student pass: load
        this cohort's captured teacher routing, install the flat student->
        teacher row map, and return the student_phase context."""
        routing = moe_routing.get(cohort_idx)
        if routing is None or layer not in routing["idx"]:
            raise RuntimeError(
                f"no captured teacher routing for layer {layer}; the online "
                "capture must precede every MoE student step")
        moe_ctrl.t_idx = {layer: routing["idx"][layer]}
        moe_ctrl.t_logp = ({layer: routing["logp"][layer]}
                           if layer in routing["logp"] else {})
        moe_ctrl.set_maps(row_map, row_mask)
        return moe_ctrl.student_phase()

    def layer_cohort_step(layer: int, cohort_idx: int, epoch_state: dict,
                          epoch_lr: float, capture: dict | None = None
                          ) -> None:
        cohort = cohorts[cohort_idx]
        layer_type = _bk_layer_type(stack, layer)
        moe_step = moe_ctrl is not None and layer in moe_ctrl.adapters
        router_extra = None
        prep_started = time.perf_counter()
        entry = build_layer_cohort(layer, cohort_idx, capture)
        if layer_type == "linear_attention":
            full_inputs, targets = tensors.staged_linear(entry)
            B = full_inputs.shape[0]
            pos = torch.arange(cohort.T, device=device)[None].expand(B, -1)
            keep = cohort.keep.to(device)
            if moe_step:
                # Full-sequence pass: student rows ARE teacher rows.
                row_map = torch.arange(B * cohort.T, device=device)
                row_mask = (torch.arange(cohort.T, device=device)[None]
                            < torch.tensor(cohort.t_len,
                                           device=device)[:, None]
                            ).reshape(-1)
                ctx = moe_student_ctx(layer, cohort, row_map, row_mask)
            else:
                ctx = contextlib.nullcontext()
            torch.cuda.synchronize(device)
            epoch_state["_prep_s"] = (epoch_state.get("_prep_s", 0.0)
                                      + time.perf_counter() - prep_started)
            exec_started = time.perf_counter()
            with ctx:
                out_full = stack.run_block(
                    layer, full_inputs, stack.rope(full_inputs, pos),
                    position_ids=pos, flow_keep=keep, causal_length=cohort.T)
                if moe_step:
                    router_extra = pending_router_loss()
            out = cohort.gather_query_inputs(out_full)
            del out_full
        else:
            kv, inputs_q, targets = tensors.staged(entry)
            rope_q = stack.rope(inputs_q, cohort.qpos_dev)
            input_ids_q = None
            if isinstance(kv, FrozenDeepseekCtx):
                # Every V4 layer's K=V branch is sliding-windowed; the
                # compressed-entry columns (causality + censorship +
                # teacher-forced indexer selection) extend the mask.
                rate = None
                if layer_type != "sliding_attention":
                    rate = stack.text_config.compress_rates[layer_type]
                mask = extended_additive_mask(
                    cohort, kv, rate, stack.text_config.sliding_window,
                    inputs_q.dtype)
                # Hash-MoE layers route by the row's own token id.
                input_ids_q = cohort.teacher_ids.to(device).gather(
                    1, cohort.qpos_dev)
            else:
                window = None
                if layer_type in ("sliding_attention", "chunked_attention"):
                    window = (getattr(stack.text_config, "sliding_window",
                                      None)
                              or getattr(stack.text_config,
                                         "attention_chunk_size", None))
                mask = cohort.additive_mask(inputs_q.dtype, window=window)
            if moe_step:
                # Query-row pass: student row (b, j) sits at teacher
                # position qpos[b, j]; padded rows have qpos 0 (never a
                # real query — aligned spans start after the prefix).
                Bq = inputs_q.shape[0]
                row_map = (torch.arange(Bq, device=device)[:, None]
                           * cohort.T + cohort.qpos_dev).reshape(-1)
                row_mask = (cohort.qpos_dev > 0).reshape(-1)
                ctx = moe_student_ctx(layer, cohort, row_map, row_mask)
            else:
                ctx = contextlib.nullcontext()
            torch.cuda.synchronize(device)
            epoch_state["_prep_s"] = (epoch_state.get("_prep_s", 0.0)
                                      + time.perf_counter() - prep_started)
            exec_started = time.perf_counter()
            with ctx:
                out = stack.run_block(
                    layer, inputs_q.requires_grad_(False), rope_q,
                    position_ids=cohort.qpos_dev,
                    past_key_values=kv, use_cache=False,
                    prepared_attention_mask=mask, input_ids=input_ids_q)
                if moe_step:
                    router_extra = pending_router_loss()
        view = stack.loss_view(layer, out)
        target = targets.to(view.dtype)
        valid = cohort.loss_valid_dev
        flat_view = view[valid]
        flat_target = target[valid]
        mean_loss = loss_fn(flat_view, flat_target,
                            normed=(layer == n), layer=layer)
        summed = mean_loss * flat_view.shape[0]
        if router_extra is not None:
            # router_aligned only: the pre-weighted KL(teacher||student)
            # regularizer joins THIS step's backward (drained inside the
            # phase so the graph never leaks across steps).
            summed = summed + router_extra
        params = _clear_block_grads(stack, layer)
        summed.backward()
        if cfg.train.v4_optimizer == "adam":
            opt = optimizers[layer]
            if cfg.train.v4_grad_clip > 0:
                # clip_grad_norm_ rescales grads in place and RETURNS the
                # pre-clip total norm — the honest diagnostic to log.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, cfg.train.v4_grad_clip)
            else:
                with torch.no_grad():
                    grads = [p.grad for p in params if p.grad is not None]
                    norms = torch._foreach_norm(grads, 2) if grads else []
                    grad_sq = (torch.stack(norms).float().square().sum()
                               if grads else torch.zeros((), device=device))
                grad_norm = grad_sq.sqrt()
            opt.step()
            opt.zero_grad(set_to_none=True)
        else:
            grad_norm = _immediate_sgd(params, epoch_lr)
        torch.cuda.synchronize(device)
        epoch_state["_exec_s"] = (epoch_state.get("_exec_s", 0.0)
                                  + time.perf_counter() - exec_started)
        state = epoch_state.setdefault(layer, {
            "loss_sum": torch.zeros((), dtype=torch.float64, device=device),
            "cells": 0,
            "grad_sq": torch.zeros((), dtype=torch.float64, device=device),
            "writes": 0,
        })
        state["loss_sum"] += mean_loss.detach().double() * flat_view.shape[0]
        state["cells"] += flat_view.shape[0]
        state["grad_sq"] += grad_norm.double().square()
        state["writes"] += 1
        try:
            util = torch.cuda.utilization(device)
        except Exception:
            util = -1
        if util >= 0:
            samples = epoch_state.setdefault("_util", [])
            samples.append(util)
            # Mid-epoch self-abort (owner, 2026-07-17): stop as soon as the
            # evidence is in, not at the epoch boundary. Warmup of 128
            # cohort-steps covers the tensor-build first pass; then a
            # rolling last-128 mean below the floor is a FAIL. External
            # watchers see the same signal in the v4_epoch rows and the
            # sample_gpu_telemetry.sh CSV.
            floor = cfg.train.v4_min_train_gpu_util
            # Epoch 1 (0-indexed 0) is exempt, mirroring the epoch-boundary
            # gate below: the first epoch's cache/prefill warm-up (v4
            # online teacher recaptures per cohort) is capture-bound, not a
            # steady-state utilization signal.
            if (floor and epoch_state.get("_epoch", 0) > 0
                    and len(samples) >= 256 and len(samples) % 64 == 0):
                rolling = sum(samples[-128:]) / 128.0
                if rolling < floor:
                    raise RuntimeError(
                        f"UTILIZATION GATE (mid-epoch): rolling "
                        f"training-phase GPU utilization {rolling:.1f}% < "
                        f"{floor:.0f}% floor after {len(samples)} cohort "
                        f"steps (goal 90%). Aborting now rather than "
                        f"finishing an idle epoch.")
        if layer == n:
            # CE/KL eval over the answer-predictor rows, streaming, before
            # any later write touches this block again this epoch.
            with torch.no_grad():
                rows_v, rows_t, ids = [], [], []
                for b in range(len(cohort.indices)):
                    r = cohort.eval_rows[b].to(device)
                    rows_v.append(view[b].index_select(0, r))
                    rows_t.append(target[b].index_select(0, r))
                    ids.append(cohort.eval_ids[b])
                ce, kl, count = teacher_output_eval_sums(
                    torch.cat(rows_v).detach().float(),
                    torch.cat(rows_t).detach().float(),
                    torch.cat(ids).to(device), stack.lm_head)
            ev = epoch_state.setdefault("_eval", {
                "ce": torch.zeros((), dtype=torch.float64, device=device),
                "kl": torch.zeros((), dtype=torch.float64, device=device),
                "count": 0})
            ev["ce"] += ce.double()
            ev["kl"] += kl.double()
            ev["count"] += count
        del out, view, target

    stopped = False
    expected_eval = sum(c.n_eval for c in cohorts)
    # Per-epoch particular evaluations (recall corpora incl. epoch zero,
    # standard damage, parameter deltas) are non-negotiable (owner,
    # 2026-07-17).  They need every trained layer in one model, so staged
    # multi-process runs defer them to the merged-adapter pass (M3); the
    # single-process mode runs them exactly as v3 does.
    single_process = cfg.train.v4_stage == -1
    relay = None
    if (not single_process and cfg.train.v4_relay_every_cohorts
            and run_dir is not None):
        relay = _RelayServicer(cfg, stack, ds, cohorts, cache, device, log,
                               run_dir, owned, rotator=rotator,
                               teacher_eval_rows=teacher_eval_rows)
    if store_source:
        # Capture ONCE, before any epoch: the relay fills this stage's
        # per-(layer, cohort) store; every epoch then runs with zero
        # teacher forwards (the measured 3.2x lever at 27B).
        from .v4_store import capture_relay_store
        capture_relay_store(
            cfg, stack, ds, cohorts, tensors, adapters_off, device,
            run_dir, log, owned=owned, n_layers=n, moe_ctrl=moe_ctrl,
            moe_routing=moe_routing, teacher_eval_rows=teacher_eval_rows,
            rotator=rotator)
    started_at = time.time()
    tracker = ParameterDeltaTracker(stack)
    baseline = None
    if cfg.train.v4_battery_mode == "subprocess":
        if int(cfg.eval.every_epochs) > int(cfg.train.epochs):
            # Declared debt, not silence (owner kill-doomed-runs policy,
            # 2026-07-18): a smoke whose eval cadence exceeds its epoch
            # count wants TIMING data, not batteries — at 122B PPP1 the
            # one-card epoch-0 battery is hours of CPU-offload generation
            # before the first training epoch.
            log.log(kind="epoch_battery_skipped", epoch=0,
                    reason="eval_cadence_beyond_run_epochs")
        else:
            # Epoch-zero baseline under the subprocess battery: every
            # stage participates (publish zero-init adapters, release
            # VRAM, ack); the subprocess evaluates the base model. This
            # INCLUDES rotary PPP1 (single_process + scoped): an
            # in-process epoch-0 probe would run a full model.forward
            # against CPU-mastered rotated blocks — the 2026-07-18 g31b
            # PPP1 crash ("cuda:0 and cpu").
            baseline = _subprocess_battery(cfg, stack, log, 0, run_dir,
                                           owned, baseline, rotator=rotator)
    elif single_process or cfg.train.v4_stage == 0:
        # Stage 0 runs epoch zero directly: LoRA is zero-init everywhere,
        # so its full resident model IS the base model at this point.
        baseline = _epoch_zero_telemetry(cfg, stack, tok, log, started_at)
    tracker.log(log, epoch=0, phase="epoch0", started_at=started_at)
    for epoch in range(cfg.train.epochs):
        if stopped:
            break
        epoch_started = time.time()
        epoch_lr = cfg.train.lr
        epoch_state: dict = {"_util": [], "_epoch": epoch}
        rng = random.Random(cfg.train.seed + epoch)
        visit = list(range(len(cohorts)))
        rng.shuffle(visit)
        if (cfg.train.v4_kv_source == "student_refresh"
                and cfg.train.v4_kv_refresh_epochs
                and epoch and epoch % cfg.train.v4_kv_refresh_epochs == 0):
            tensors.store.clear()
        if cfg.train.v4_loop_order == "layer_major":
            owned_list = list(owned)
            for pos, layer in enumerate(owned_list):
                if stopped:
                    break
                if rotator is not None:
                    rotate_started = time.perf_counter()
                    rotator.activate(layer)
                    if pos + 1 < len(owned_list):
                        rotator.prefetch(owned_list[pos + 1])
                    epoch_state["_prep_s"] = (
                        epoch_state.get("_prep_s", 0.0)
                        + time.perf_counter() - rotate_started)
                for cohort_idx in visit:
                    layer_cohort_step(layer, cohort_idx, epoch_state, epoch_lr)
                    if stop_requested():
                        stopped = True
                        break
                if rotator is not None:
                    rotator.evict(layer)
                if residency == "gpu_corpus" and len(owned) > 1:
                    free, _ = torch.cuda.mem_get_info(device)
                    if free < 8 << 30:
                        tensors.drop_layer(layer)
        else:
            for cohort_idx in visit:
                if stopped:
                    break
                capture = None
                if online_source:
                    missing = any(
                        tensors.get(layer, cohort_idx) is None
                        for layer in owned)
                    if missing:
                        cap_started = time.perf_counter()
                        if moe_ctrl is not None:
                            with moe_ctrl.teacher_phase():
                                capture = _online_teacher_capture(
                                    cfg, stack, adapters_off,
                                    cohorts[cohort_idx], owned, device, n)
                            # Harvest routing for OWNED MoE layers only:
                            # foreign layers are never stepped here, and
                            # teacher_phase clears t_idx on its next entry.
                            moe_routing[cohort_idx] = {
                                "idx": {L: moe_ctrl.t_idx[L]
                                        for L in moe_ctrl.t_idx
                                        if L in owned},
                                "logp": {L: moe_ctrl.t_logp[L]
                                         for L in moe_ctrl.t_logp
                                         if L in owned},
                            }
                        else:
                            capture = _online_teacher_capture(
                                cfg, stack, adapters_off, cohorts[cohort_idx],
                                owned, device, n)
                        epoch_state["_capture_s"] = (
                            epoch_state.get("_capture_s", 0.0)
                            + time.perf_counter() - cap_started)
                        if capture["eval_rows_teacher"] is not None:
                            teacher_eval_rows[cohort_idx] = [
                                r.detach() for r in
                                capture["eval_rows_teacher"]]
                for layer in owned:
                    layer_cohort_step(layer, cohort_idx, epoch_state,
                                      epoch_lr, capture)
                if stop_requested():
                    stopped = True
        # One host sync per epoch: flush per-layer telemetry.
        layer_losses = {}
        grad_norms = {}
        token_events = 0
        for layer in owned:
            state = epoch_state.get(layer)
            if state is None:
                continue
            cells = max(state["cells"], 1)
            layer_losses[str(layer)] = float(state["loss_sum"].item() / cells)
            grad_norms[str(layer)] = float(state["grad_sq"].sqrt().item())
            token_events += state["cells"]
        elapsed = max(time.time() - epoch_started, 1e-9)
        util_samples = epoch_state.get("_util", [])
        train_util = (sum(util_samples) / len(util_samples)
                      if util_samples else None)
        log.log(kind="v4_epoch", epoch=epoch + 1,
                partial=bool(stopped),
                # Rotation observability (owner, 2026-07-18: "rotations
                # should also be optimisable for max GPU usage") — honest
                # per-epoch stall/traffic, drained per epoch.
                **(rotator.take_counters() if rotator is not None
                   and hasattr(rotator, "take_counters") else {}),
                layer_losses=layer_losses,
                token_events=token_events,
                token_events_per_second=token_events / elapsed,
                physical_writes=sum(
                    s["writes"] for k, s in epoch_state.items()
                    if isinstance(k, int)),
                train_phase_gpu_util=train_util,
                train_util_samples=len(util_samples),
                prep_seconds=round(epoch_state.get("_prep_s", 0.0), 3),
                capture_seconds=round(
                    epoch_state.get("_capture_s", 0.0), 3),
                exec_seconds=round(epoch_state.get("_exec_s", 0.0), 3),
                prep_fraction=round(
                    epoch_state.get("_prep_s", 0.0)
                    / max(epoch_state.get("_prep_s", 0.0)
                          + epoch_state.get("_exec_s", 0.0), 1e-9), 4),
                epoch_seconds=elapsed)
        # Utilization gate (owner, 2026-07-17): training-phase mean below
        # the configured floor is a FAIL — abort loudly, never let an idle
        # card masquerade as a run. Epoch 1 is exempt (cache/prefill warm-up).
        floor = cfg.train.v4_min_train_gpu_util
        if (floor and epoch > 0 and not stopped and train_util is not None
                and train_util < floor):
            raise RuntimeError(
                f"UTILIZATION GATE: training-phase GPU utilization "
                f"{train_util:.1f}% < {floor:.0f}% floor at epoch "
                f"{epoch + 1} (goal is 90%). This run is a FAIL by owner "
                f"criterion; profile the walk instead of letting it idle.")
        log.log(kind="v4_gradient_norm", epoch=epoch + 1,
                grad_norms=grad_norms)
        if moe_ctrl is not None:
            # One extra host sync at the epoch boundary (allowed there):
            # mean teacher/student top-k overlap per owned MoE layer.
            log.log(kind="v4_moe_overlap", epoch=epoch + 1,
                    moe_mode=cfg.train.moe_mode,
                    teacher_topk_overlap={
                        str(L): v
                        for L, v in moe_ctrl.overlap_flush().items()})
        ev = epoch_state.get("_eval")
        if ev is not None and not stopped:
            count = max(ev["count"], 1)
            log.log(
                kind="teacher_output_eval", epoch=epoch + 1,
                CE_eval_loss=float(ev["ce"].item() / count),
                KL_eval_loss=float(ev["kl"].item() / count),
                answer_token_count=ev["count"],
                expected_answer_token_count=expected_eval,
                dataset_item_count=len(ds.pairs),
                dataset_coverage="whole_training_set_once_per_completed_epoch",
                token_coverage="every_teacher_realized_answer_token",
                answer_only=True,
                evaluation_only=True,
                validation_subset=False,
                used_for_backward=False,
                optimizer_weight=0.0,
                aggregation="token_weighted_mean",
                temporal_semantics=(
                    "streaming_pre_final_block_write_at_each_cohort_visit"),
                trajectory="teacher_forced_blockwise",
                CE_target="teacher_realized_answer_token_ids",
                KL_direction="teacher_to_student",
                vocabulary_head="frozen",
            )
        tracker.log(log, epoch=epoch + 1,
                    phase=f"after_epoch_{epoch + 1}", started_at=started_at)
        boundary_started = time.perf_counter()
        if single_process and not stopped:
            if cfg.train.v4_relay_every_cohorts and (
                    (epoch + 1) % cfg.train.v4_relay_every_cohorts == 0):
                _student_trajectory_eval(
                    cfg, stack, ds, cohorts, cache, device, log,
                    epoch=epoch + 1,
                    teacher_rows_by_cohort=teacher_eval_rows,
                    rotator=rotator)
            if cfg.train.v4_stage_scoped:
                # Rotary PPP1: the model is never resident, so the direct
                # telemetry probes cannot run in-process. Spawn the
                # subprocess battery at the eval cadence; mark the
                # off-cadence epochs loudly (the per-epoch battery law is
                # visible debt here, not silence).
                every = max(int(cfg.eval.every_epochs), 1)
                if (epoch + 1) % every == 0:
                    baseline = _subprocess_battery(
                        cfg, stack, log, epoch + 1, run_dir, owned,
                        baseline, rotator=rotator)
                else:
                    log.log(kind="epoch_battery_skipped", epoch=epoch + 1,
                            reason="ppp1_rotate_battery_at_eval_cadence")
            else:
                baseline = _epoch_end_telemetry(
                    cfg, stack, tok, log, epoch=epoch, baseline=baseline,
                    started_at=started_at)
        elif not stopped:
            if run_dir is None:
                raise ValueError("staged pipeline-v4 needs run_dir for the "
                                 "relay/battery exchange")
            if relay is not None and (
                    (epoch + 1) % max(cfg.train.v4_relay_every_cohorts, 1)
                    == 0):
                relay.submit(epoch + 1)
            baseline = _staged_epoch_battery(
                cfg, stack, tok, log, epoch + 1, run_dir, owned, baseline,
                started_at, rotator=rotator)
        if not stopped:
            log.log(kind="v4_epoch_boundary", epoch=epoch + 1,
                    boundary_seconds=round(
                        time.perf_counter() - boundary_started, 3))
    if relay is not None and not stopped:
        relay.drain()
    if optimizers and run_dir is not None:
        # Warm starts keep momentum (plan B0): per-block AdamW state,
        # CPU-serialized, outside the epoch loop (no hot-loop syncs).
        ckpt = Path(run_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        payload = {layer: _cpu_state_dict(opt)
                   for layer, opt in optimizers.items() if opt.state}
        if payload:
            torch.save(payload, ckpt / "adam_moments.pt")
            log.log(kind="v4_adam_moments", action="saved",
                    layers=sorted(payload))
    return stopped


def _cpu_state_dict(opt) -> dict:
    """Optimizer state_dict with every tensor moved to CPU (checkpoint
    serialization; the restore side's load_state_dict casts back to the
    params' device, and rotation's _moments_to keeps paging afterwards)."""
    def mv(x):
        if torch.is_tensor(x):
            return x.cpu()
        if isinstance(x, dict):
            return {k: mv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [mv(v) for v in x]
        return x
    return mv(opt.state_dict())


def certify_locality_v4(cfg, stack, tok, cache, run_dir, items: int = 4,
                        peft_model=None):
    """Measured locality certification for the v4 objective.

    For a sample of (item, layer) cells, run one v4 step's backward and
    verify the gradient touches exactly the current block: zero gradient on
    every other block, the embedding, the final norm, and the LM head.
    """
    from ..data.dataset import DistillDataset

    n = stack.n_layers
    owned = _owned_range(cfg, n)
    # store regenerates exactly like online here: one adapters-off forward
    # per sampled item (teacher states are the same computation the relay
    # captured; certification needs no store plumbing).
    online_source = cfg.train.v4_teacher_source in ("online", "store")
    if cfg.train.v4_stage_scoped and cfg.train.v4_teacher_source != "cache":
        # The certification capture walks EVERY layer; foreign blocks are
        # meta under stage-scoped loading. A cert relay is future work —
        # report the debt loudly instead of crashing the end of a run.
        return {
            "items": 0,
            "skipped": "stage_scoped_store_certification_pending_relay",
            "passed": False,
            "owner_note": "locality certification debt, not evidence",
        }
    device = torch.device(cfg.model.device)
    loss_fn = HiddenLoss.from_config(cfg.train, stack)
    ds = DistillDataset(
        cfg.data.examples_path, cache, tok, need_layers=[],
        with_teacher_ids=False, pad_random=False,
        cache_source_compaction=cfg.cache.source_compaction,
        student_compaction=cfg.mask.compaction,
        item_cache_items=cfg.cache.item_cache_items)
    sample_layers = sorted({owned.start, (owned.start + owned.stop - 1) // 2,
                            owned.stop - 1})
    adapters_off = peft_model.disable_adapter if peft_model is not None else None
    local_sq = 0.0
    cross_sq = 0.0
    vocab_sq = 0.0
    checked = 0
    vocab_params = (list(stack.embed_tokens.parameters())
                    + list(stack.final_norm.parameters())
                    + list(stack.lm_head.parameters())
                    + (list(stack.hc_head.parameters())
                       if getattr(stack, "hc_head", None) is not None
                       else []))
    for item_index in range(min(items, len(ds.pairs))):
        cohort = _V4Cohort(cfg, ds, [item_index], device)
        # Online runs carry no hidden cache; regenerate this item's teacher
        # inputs/targets with one adapters-off forward (the same source the
        # training walk uses) instead of reading the absent cache shards.
        online_cap = (
            _online_teacher_capture(cfg, stack, adapters_off, cohort,
                                    owned, device, n)
            if online_source else None)
        for layer in sample_layers:
            # Under weight rotation the sampled block's frozen masters live
            # on host; page them in for the certification step and restore
            # after (the certification must never leave placement changed).
            paged = []
            for p in stack.block_params(layer):
                if p.device.type == "cpu" and not p.is_meta:
                    paged.append((p, p.data))
                    p.data = p.data.to(device)
            paged_buf = []
            for _, buf in stack.blocks[layer - 1].named_buffers():
                if buf.device.type == "cpu" and not buf.is_meta:
                    paged_buf.append((buf, buf.data))
                    buf.data = buf.data.to(device)
            for foreign in range(1, n + 1):
                for p in stack.block_params(foreign):
                    p.grad = None
            for p in vocab_params:
                p.grad = None
            if online_cap is not None:
                full_inputs = online_cap["inputs"][layer].to(device)
                targets = online_cap["targets"][layer].to(device)
            else:
                full_inputs = cohort.gather_full_inputs(cache, layer).to(device)
                targets = cohort.gather_targets(cache, layer).to(device)
            layer_type = _bk_layer_type(stack, layer)
            if layer_type == "linear_attention":
                # Same routing as the walk: recurrent mixers take the full
                # teacher-forced sequence, no KV object.
                B = full_inputs.shape[0]
                pos = torch.arange(cohort.T, device=device)[None].expand(B, -1)
                out_full = stack.run_block(
                    layer, full_inputs, stack.rope(full_inputs, pos),
                    position_ids=pos, flow_keep=cohort.keep.to(device),
                    causal_length=cohort.T)
                out = cohort.gather_query_inputs(out_full)
            elif getattr(stack, "needs_deepseek_masks", False):
                # Same record/serve procedure as the training walk.
                pos = torch.arange(cohort.T, device=device)[None]
                ids_full = cohort.teacher_ids.to(device)
                ctx = (adapters_off() if adapters_off is not None
                       else contextlib.nullcontext())
                rec = DeepseekRecorder(stack, layer)
                try:
                    with torch.no_grad(), ctx:
                        stack.run_block(
                            layer, full_inputs, stack.rope(full_inputs, pos),
                            position_ids=pos, past_key_values=rec.shim,
                            use_cache=True,
                            prepared_attention_mask=NO_PREPARED_ATTENTION_MASK,
                            input_ids=ids_full)
                finally:
                    rec.close()
                kv_t, entries_t, topk_t = rec.harvest()
                frozen = FrozenDeepseekCtx(
                    kv_t, entries_t,
                    gather_topk_at_qpos(topk_t, cohort.qpos_dev), layer - 1)
                inputs_q = cohort.gather_query_inputs(full_inputs)
                rope_q = stack.rope(inputs_q, cohort.qpos_dev)
                rate = None
                if layer_type != "sliding_attention":
                    rate = stack.text_config.compress_rates[layer_type]
                mask = extended_additive_mask(
                    cohort, frozen, rate, stack.text_config.sliding_window,
                    inputs_q.dtype)
                out = stack.run_block(
                    layer, inputs_q, rope_q, position_ids=cohort.qpos_dev,
                    past_key_values=frozen, use_cache=False,
                    prepared_attention_mask=mask,
                    input_ids=ids_full.gather(1, cohort.qpos_dev))
            else:
                kv = _FrozenKV()
                pos = torch.arange(cohort.T, device=device)[None]
                ctx = (adapters_off() if adapters_off is not None
                       else contextlib.nullcontext())
                with torch.no_grad(), ctx:
                    stack.run_block(layer, full_inputs,
                                    stack.rope(full_inputs, pos),
                                    position_ids=pos, past_key_values=kv,
                                    use_cache=True,
                                    prepared_attention_mask=NO_PREPARED_ATTENTION_MASK)
                kv.recording = False
                inputs_q = cohort.gather_query_inputs(full_inputs)
                rope_q = stack.rope(inputs_q, cohort.qpos_dev)
                window = None
                if layer_type in ("sliding_attention", "chunked_attention"):
                    window = (getattr(stack.text_config, "sliding_window", None)
                              or getattr(stack.text_config,
                                         "attention_chunk_size", None))
                mask = cohort.additive_mask(inputs_q.dtype, window=window)
                out = stack.run_block(
                    layer, inputs_q, rope_q, position_ids=cohort.qpos_dev,
                    past_key_values=kv, use_cache=False,
                    prepared_attention_mask=mask)
            view = stack.loss_view(layer, out)
            valid = cohort.loss_valid_dev
            loss = loss_fn(view[valid], targets.to(view.dtype)[valid],
                           normed=(layer == n), layer=layer)
            (loss * int(valid.sum())).backward()
            for p in stack.block_params(layer):
                if p.grad is not None:
                    local_sq += float(p.grad.float().square().sum())
            for foreign in range(1, n + 1):
                if foreign == layer:
                    continue
                for p in stack.block_params(foreign):
                    if p.grad is not None:
                        cross_sq += float(p.grad.float().square().sum())
            for p in vocab_params:
                if p.grad is not None:
                    vocab_sq += float(p.grad.float().square().sum())
            checked += 1
            for p in stack.block_params(layer):
                p.grad = None
            for p, cpu_data in paged:
                p.data = cpu_data
            for buf, cpu_data in paged_buf:
                buf.data = cpu_data
    passed = (local_sq > 0 and cross_sq == 0.0 and vocab_sq == 0.0)
    return {
        "items": checked,
        "gradient_contract": "teacher_forced_blockwise_frozen_teacher_kv",
        "final_logit_training": False,
        "local_grad_norm": local_sq ** 0.5,
        "cross_block_leak_grad_norm": cross_sq ** 0.5,
        "frozen_vocab_grad_norm": vocab_sq ** 0.5,
        "local_signal_present_in_every_block": local_sq > 0,
        "passed": bool(passed),
    }
