"""Frozen teacher context for DeepSeek-V4 (MLA + compressor stack), plan B8.

DeepSeek-V4 attention has three key-side series per layer: the sliding K=V
branch (shared-KV MQA, window ``config.sliding_window``), the compressed
long-range entries (HCA rate 128 / CSA rate 4), and — on CSA layers — the
lightning-indexer keys that pick top-k compressed entries per query.  The
v4 contract (queries student-side, keys/values teacher-frozen, censorship
by key removal) maps onto them as:

* sliding K=V      -> recorded at teacher prefill, served frozen
                      (exactly the ``_FrozenKV`` record/consume contract);
* compressed KV    -> recorded at teacher prefill, served frozen; a
                      compressed entry whose source window contains a
                      censored (privileged or padded) position is removed
                      from the student's attention entirely;
* indexer routing  -> TEACHER-FORCED: the teacher's top-k selection is
                      recorded and baked into the extended mask.  The
                      student never re-scores (its selection would be a
                      key-side decision from student weights).

The serving mechanism needs no forked modeling code.  The compressors are
stateful only through the cache object: ``store_compression_weights``
peels window chunks, ``update_compressor_states`` appends and returns the
running entries.  A frozen cache layer that returns ZERO-WIDTH chunks
makes the model's own compressor code emit nothing and hand back the
recorded teacher entries; the indexer additionally receives an EMPTY
series, so its scorer short-circuits.  Compressed-slot visibility
(causality + censorship + teacher top-k) is enforced by pre-extending the
prepared additive mask to ``T + n_entries`` columns: DeepSeek attention
only concatenates its internal ``block_bias`` when the incoming mask is
narrower than the post-concat KV axis, so a full-width mask governs the
compressed slots outright.

The record pass reuses the REAL ``DeepseekV4HCACache``/``CSACache`` layer
objects so buffers/overlap/entry-count semantics are the model's own, and
captures the indexer's top-k via a forward hook.
"""

from __future__ import annotations

import torch

DEEPSEEK_LAYER_TYPES = ("compressed_sparse_attention",
                        "heavily_compressed_attention",
                        "sliding_attention")


def is_deepseek_v4(text_config) -> bool:
    return getattr(text_config, "model_type", "").startswith("deepseek_v4")


def _cache_layer_cls(layer_type: str):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4CSACache, DeepseekV4HCACache)
    if layer_type == "compressed_sparse_attention":
        return DeepseekV4CSACache
    if layer_type == "heavily_compressed_attention":
        return DeepseekV4HCACache
    return None  # plain sliding layer: no compressor state


class _RecordShim:
    """Cache-shim for ONE block's teacher record pass (full sequence, fresh
    state).  ``update`` intercepts at the Cache level and records the roped
    full-sequence K(=V); ``layers[idx]`` hands the compressor the real typed
    cache layer so its window/buffer/overlap arithmetic is the model's own.
    """

    def __init__(self, text_config, layer_idx0: int, layer_type: str):
        cls = _cache_layer_cls(layer_type)
        self.layer = cls(text_config) if cls is not None else None
        self.layers = {layer_idx0: self.layer}
        self.kv = None

    def update(self, key_states, value_states, layer_idx=None,
               cache_kwargs=None):
        self.kv = key_states.detach()
        return self.kv, self.kv

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0


class DeepseekRecorder:
    """Record one block's frozen-context artifacts over one full-sequence
    chunk.  MUST be fresh per chunk: the typed cache treats successive
    forward calls as time-continuation, while capture chunks are along the
    batch axis."""

    def __init__(self, stack, layer: int):
        idx0 = layer - 1
        self.layer_type = stack.layer_types[idx0]
        self.shim = _RecordShim(stack.text_config, idx0, self.layer_type)
        self.topk = None
        self._handle = None
        indexer = getattr(
            getattr(stack.blocks[idx0].self_attn, "compressor", None),
            "indexer", None)
        if indexer is not None:
            def _grab(module, args, output):
                self.topk = output.detach()
            self._handle = indexer.register_forward_hook(_grab)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def harvest(self):
        """(kv [B,1,T,hd], entries [B,n_e,hd] | None, topk [B,T,k] | None)."""
        if self.shim.kv is None:
            raise RuntimeError("deepseek record pass stored no sliding K=V")
        entries = None
        if self.shim.layer is not None:
            entries = self.shim.layer.compressed_kv.get("compressor")
            if entries is not None:
                entries = entries.detach()
        if self.layer_type == "compressed_sparse_attention" \
                and self.topk is None:
            raise RuntimeError(
                "CSA record pass captured no indexer top-k (hook missed)")
        return self.shim.kv, entries, self.topk


class _FrozenDeepseekLayer:
    """Serve side of the typed-cache duck-type: zero-width chunks in,
    recorded teacher entries out."""

    def __init__(self, entries):
        self.entries = entries  # [B, n_e, hd] or None (pure sliding layer)

    def store_compression_weights(self, name, kv, gate):
        return kv[:, :0], gate[:, :0], 0

    def update_compressor_states(self, name, compressed):
        if compressed.shape[1] != 0:
            raise RuntimeError(
                "frozen deepseek ctx received a non-empty compressed chunk; "
                "store_compression_weights was bypassed")
        if name == "indexer":
            # Empty indexer series: the scorer sees zero keys and selection
            # short-circuits.  Routing is teacher-forced via the extended
            # prepared mask instead.
            return compressed
        if self.entries is None:
            raise RuntimeError("frozen deepseek ctx has no recorded entries")
        return self.entries

    def update_overlap_state(self, name, chunk_kv, chunk_gate, head_dim):
        raise RuntimeError(
            "unreachable: zero-width chunks never reach overlap state")


class FrozenDeepseekCtx:
    """Frozen teacher context for ONE block, duck-typed as an HF cache.

    ``update`` discards the student's key/value projection and returns the
    recorded teacher K(=V) — no gradient can enter key-side.  ``layers``
    serves the frozen compressed entries.  ``topk`` rides along for the
    extended-mask builder (already gathered at the cohort's query rows,
    [B, Q, k] with -1 sentinels).
    """

    def __init__(self, kv, entries, topk, layer_idx0: int):
        self.kv = kv
        self.entries = entries
        self.topk = topk
        self.layer_idx0 = layer_idx0
        self.layers = {layer_idx0: _FrozenDeepseekLayer(entries)}

    def update(self, key_states, value_states, layer_idx=None,
               cache_kwargs=None):
        if self.kv is None:
            raise RuntimeError("frozen deepseek ctx consumed before prefill")
        return self.kv, self.kv

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if self.kv is None else int(self.kv.shape[2])

    @property
    def n_entries(self) -> int:
        return 0 if self.entries is None else int(self.entries.shape[1])

    # -- residency management (mirrors _FrozenKV) ------------------------

    def to(self, device):
        self.kv = self.kv.to(device, non_blocking=True)
        if self.entries is not None:
            self.entries = self.entries.to(device, non_blocking=True)
        if self.topk is not None:
            self.topk = self.topk.to(device, non_blocking=True)
        self.layers[self.layer_idx0].entries = self.entries
        return self

    def pin(self):
        if self.kv.device.type == "cpu":
            self.kv = self.kv.pin_memory()
            if self.entries is not None:
                self.entries = self.entries.pin_memory()
            if self.topk is not None:
                self.topk = self.topk.pin_memory()
            self.layers[self.layer_idx0].entries = self.entries
        return self

    def staged_to(self, device) -> "FrozenDeepseekCtx":
        return FrozenDeepseekCtx(
            self.kv.to(device, non_blocking=True),
            (self.entries.to(device, non_blocking=True)
             if self.entries is not None else None),
            (self.topk.to(device, non_blocking=True)
             if self.topk is not None else None),
            self.layer_idx0)

    def nbytes(self) -> int:
        total = self.kv.numel() * self.kv.element_size()
        if self.entries is not None:
            total += self.entries.numel() * self.entries.element_size()
        if self.topk is not None:
            total += self.topk.numel() * self.topk.element_size()
        return total


def gather_topk_at_qpos(topk: torch.Tensor | None,
                        qpos: torch.Tensor) -> torch.Tensor | None:
    """[B,T,k] full-sequence teacher selection -> [B,Q,k] at the query rows.
    Stored this way: int64 over the full sequence is the dominant memory
    term (T=5k -> ~650 MB/cohort), while Q rows are a few percent of it."""
    if topk is None:
        return None
    idx = qpos.to(topk.device)[:, :, None].expand(-1, -1, topk.shape[-1])
    return topk.gather(1, idx).to(torch.int32)


def extended_additive_mask(cohort, ctx: FrozenDeepseekCtx,
                           compress_rate: int | None, sliding_window: int,
                           dtype) -> torch.Tensor:
    """[B,1,Q,T+n_e] additive mask for one DeepSeek block's query pass.

    Columns 0..T-1: the cohort's causal+censor mask intersected with the
    sliding window (every V4 layer's K=V branch is windowed).  Columns
    T..T+n_e-1 (compressed entries): entry ``e`` is attendable iff

    * causal: ``e < (qpos + 1) // rate`` (the model's own rule);
    * clean: every source position in ``[e*rate, (e+1)*rate)`` is a keep —
      censored or padded source tokens poison the whole entry (the
      key-removal law applied at entry granularity, conservatively);
    * CSA: ``e`` is in the TEACHER's top-k selection for this query row.
    """
    base = cohort.additive_mask(dtype, window=sliding_window)
    n_e = ctx.n_entries
    if n_e == 0:
        return base
    device = base.device
    qpos = cohort.qpos_dev
    B, Q = qpos.shape
    e_idx = torch.arange(n_e, device=device)
    allowed = e_idx[None, None, :] < ((qpos + 1) // compress_rate)[:, :, None]
    keep = cohort.keep.to(device)
    clean = keep[:, : n_e * compress_rate].reshape(B, n_e, compress_rate)
    allowed = allowed & clean.all(-1)[:, None, :]
    if ctx.topk is not None:
        sel = torch.zeros((B, Q, n_e + 1), dtype=torch.bool, device=device)
        safe = ctx.topk.long().clamp(min=-1)
        sel.scatter_(-1, torch.where(safe >= 0, safe,
                                     torch.full_like(safe, n_e)), True)
        allowed = allowed & sel[..., :n_e]
    ext = torch.zeros((B, 1, Q, n_e), dtype=dtype, device=device)
    ext.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return torch.cat([base, ext], dim=-1)
