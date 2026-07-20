"""Per-block "activations-in / activations-out" contract.

``BlockStack`` exposes the pieces of a loaded HF causal LM as independently
runnable stages: embedding, rotary position embeddings, each decoder block,
final norm, lm_head. Blocks are 1-based to match the teacher-cache convention
(``h{L}`` = output of block L; ``h{n_layers}`` is post-final-norm).

This interface is the 120B streaming story: a future weight-streaming runner
only needs to implement load()/offload() per block behind the same
``run_block`` signature. At small scale everything is resident and those are
no-ops.

Batch-1, unpadded sequences: attention_mask=None takes the causal sdpa fast
path inside the attention layers.
"""

from __future__ import annotations

import inspect
from collections import UserDict

import torch


# ``None`` means "no prepared override" because cached execution may still
# need to construct a flow/causal mask. This sentinel explicitly requests the
# model's mask-free causal path for an intact cached-attention cell.
NO_PREPARED_ATTENTION_MASK = object()


def _is_text_stack(value) -> bool:
    return all(hasattr(value, attr) for attr in ("embed_tokens", "layers", "norm"))


def _resolve_text_stack(model):
    """Resolve a decoder text tower in text-only and multimodal composites.

    Qwen3.6 multimodal checkpoints do not promise that the text tower is
    reachable as ``model.model``.  Prefer explicit text-bearing attributes,
    then inspect only a small module tree while excluding vision/audio
    branches.  Returning the first object with the complete text-stack
    contract keeps the rest of ``BlockStack`` architecture-neutral.
    """
    getter = getattr(model, "get_text_model", None)
    if callable(getter):
        candidate = getter()
        if _is_text_stack(candidate):
            return candidate
    preferred = (
        "language_model", "text_model", "text_tower", "transformer",
        "model",
    )
    roots = [model]
    direct = getattr(model, "model", None)
    if direct is not None:
        roots.insert(0, direct)
    seen = set()
    for root in roots:
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        if _is_text_stack(root):
            return root
        for name in preferred:
            candidate = getattr(root, name, None)
            if candidate is not None and _is_text_stack(candidate):
                return candidate
    # Last-resort bounded traversal for composite wrappers.  Do not descend
    # through explicitly non-text towers: it is easy to accidentally select a
    # vision transformer that happens to expose a ``layers`` attribute.
    blocked = ("vision", "image", "audio", "projector", "perceiver")
    frontier = [(model, 0)]
    seen.clear()
    while frontier:
        current, depth = frontier.pop(0)
        if current is None or id(current) in seen or depth > 3:
            continue
        seen.add(id(current))
        if _is_text_stack(current):
            return current
        children = getattr(current, "named_children", lambda: ())()
        for name, child in children:
            if any(part in name.lower() for part in blocked):
                continue
            frontier.append((child, depth + 1))
    raise NotImplementedError(
        f"{type(model).__name__} has no resolvable text tower with "
        ".embed_tokens/.layers/.norm")


class BlockStack:
    def __init__(self, model):
        self.model = model
        inner = _resolve_text_stack(model)
        self.text_model = inner
        # This module layout is shared by the Qwen/Llama/DeepSeek/GLM HF
        # ports. The resolver above also handles Qwen3.6 multimodal text
        # composites without assuming ``model.model.*``.
        for attr, owner in (("embed_tokens", inner), ("layers", inner),
                            ("norm", inner)):
            if not hasattr(owner, attr):
                raise NotImplementedError(
                    f"{type(inner).__name__} lacks .{attr}; add an arch adapter "
                    "(see docs/scaling.md)"
                )
        lm_head_owner = model if hasattr(model, "lm_head") else inner
        if not hasattr(lm_head_owner, "lm_head"):
            raise NotImplementedError(
                f"{type(model).__name__} lacks a frozen text lm_head; add an "
                "arch adapter (see docs/scaling.md)")
        self.embed_tokens = inner.embed_tokens
        model_config = getattr(model, "config", None)
        self.text_config = getattr(
            inner, "config", getattr(model_config, "text_config", model_config))
        self.layer_types = list(getattr(self.text_config, "layer_types", []) or [])
        # MLA-style models compute rotary inside attention; rotary_emb is optional
        self.rotary_emb = getattr(inner, "rotary_emb", None)
        self.rotary_needs_layer_type = (
            self.rotary_emb is not None
            and "layer_type" in inspect.signature(self.rotary_emb.forward).parameters
        )
        self.needs_gemma4_masks = (
            getattr(self.text_config, "model_type", "") == "gemma4_text"
            and bool(self.layer_types)
        )
        # DeepSeek-V4: eager-only attention (no SDPA is_causal fast path), so
        # full-model walks need an explicit causal mask; every layer type
        # shares the one sliding-window mask (the compressors extend it
        # internally via block_bias).  mHC threads hc_mult parallel residual
        # streams between blocks — the block boundary is [B, T, hc_mult, H].
        self.needs_deepseek_masks = (
            getattr(self.text_config, "model_type", "")
            .startswith("deepseek_v4") and bool(self.layer_types)
        )
        self.hc_mult = int(getattr(self.text_config, "hc_mult", 0) or 0)
        self.hc_head = getattr(inner, "hc_head", None)
        self.blocks = list(inner.layers)
        self._block_params = [list(block.parameters()) for block in self.blocks]
        self._accepts_past_key_values = []
        self._accepts_shared_kv_states = []
        self._accepts_per_layer_input = []
        self._accepts_input_ids = []
        for block in self.blocks:
            params = inspect.signature(block.forward).parameters
            # A layer whose forward is (…, **kwargs) passes them straight to its
            # attention (deepseek_v4: DeepseekV4DecoderLayer.forward ->
            # self_attn(…, **kwargs)); it therefore accepts past_key_values iff
            # its attention does. Checking only explicit params was a false
            # negative that broke the deepseek store-fill recorder shim.
            # Kept SPECIFIC to past_key_values: shared_kv_states/input_ids stay
            # explicit-only — the deepseek attention rejects shared_kv_states.
            has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in params.values())
            attn = getattr(block, "self_attn", None)
            attn_pkv = attn is not None and "past_key_values" in \
                inspect.signature(attn.forward).parameters
            self._accepts_past_key_values.append(
                "past_key_values" in params or (has_var_kw and attn_pkv))
            self._accepts_shared_kv_states.append("shared_kv_states" in params)
            self._accepts_per_layer_input.append("per_layer_input" in params)
            self._accepts_input_ids.append("input_ids" in params)
        self.final_norm = inner.norm
        self.lm_head = lm_head_owner.lm_head
        self.n_layers = len(self.blocks)
        # Fallback for architectures that accept shared_kv_states without a
        # per-layer rotary bundle. Gemma4 supplies a fresh mapping in rope();
        # this path is latent on the current models but must still initialize.
        self._shared_kv_states = None
        self.frozen_input_modules = [
            module for module in (
                getattr(inner, "embed_tokens_per_layer", None),
                getattr(inner, "per_layer_model_projection", None),
                getattr(inner, "per_layer_projection_norm", None),
            ) if module is not None
        ]

    def freeze_non_blocks(self) -> None:
        """Embedding, final norm and lm_head stay at init: block-only training
        keeps the localization readout clean and matches the teacher's frozen
        h{n_layers} (post-norm with the initial norm weights)."""
        self.embed_tokens.requires_grad_(False)
        self.final_norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        if self.hc_head is not None:
            # The mHC stream-collapse head sits between the last block and
            # the frozen norm/lm_head: same frozen-vocabulary treatment.
            self.hc_head.requires_grad_(False)
        for module in self.frozen_input_modules:
            module.requires_grad_(False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # KV sharing across layer types (gemma4-class) lives in the per-call
        # rope() bundle (a fresh UserDict each walk), never on the instance —
        # instance state here once risked stale wrong-length teacher KV
        # across items (2026-07-10 review, latent).
        with torch.no_grad():
            h = self.embed_tokens(input_ids)
            if self.hc_mult > 1:
                # mHC boundary convention: the inter-block state carries all
                # hc_mult streams; the model seeds them as copies of the
                # embedding (DeepseekV4Model.forward does the same expand).
                h = h.unsqueeze(2).expand(-1, -1, self.hc_mult,
                                          -1).contiguous()
            return h

    def embed_and_per_layer_inputs(self, input_ids: torch.Tensor):
        """Embed once and prepare Gemma per-layer inputs when configured."""
        with torch.no_grad():
            raw = self.embed_tokens(input_ids)
            per_layer = self.per_layer_inputs(input_ids, raw=raw)
            hidden = raw
            if self.hc_mult > 1:
                hidden = hidden.unsqueeze(2).expand(
                    -1, -1, self.hc_mult, -1).contiguous()
            return hidden, per_layer

    def per_layer_inputs(self, input_ids: torch.Tensor, *, raw=None):
        """Frozen per-layer token features, or ``None`` for ordinary LMs."""
        if not int(getattr(self.text_config,
                           "hidden_size_per_layer_input", 0) or 0):
            return None
        with torch.no_grad():
            if raw is None:
                raw = self.embed_tokens(input_ids)
            token_part = self.text_model.get_per_layer_inputs(input_ids, raw)
            return self.text_model.project_per_layer_inputs(raw, token_part)

    def is_kv_shared_layer(self, L: int) -> bool:
        attention = getattr(self.blocks[L - 1], "self_attn", None)
        return bool(getattr(attention, "is_kv_shared_layer", False))

    def shared_kv_source(self, L: int) -> int:
        """Nearest preceding producer for a shared-KV decoder layer."""
        if not self.is_kv_shared_layer(L):
            return L
        layer_type = (self.layer_types[L - 1]
                      if L - 1 < len(self.layer_types) else None)
        for source in range(L - 1, 0, -1):
            source_type = (self.layer_types[source - 1]
                           if source - 1 < len(self.layer_types) else None)
            if source_type == layer_type and not self.is_kv_shared_layer(source):
                return source
        raise RuntimeError(
            f"shared-KV layer {L} has no preceding {layer_type!r} producer")

    def shared_kv_types_through(self, stop: int) -> list[str]:
        """Shared-KV mappings guaranteed to exist after block ``stop``."""
        if not any(self.is_kv_shared_layer(layer)
                   for layer in range(1, self.n_layers + 1)):
            return []
        produced = set()
        for layer in range(1, min(int(stop), self.n_layers) + 1):
            attention = getattr(self.blocks[layer - 1], "self_attn", None)
            if bool(getattr(attention, "store_full_length_kv", False)):
                produced.add(self.layer_types[layer - 1])
        return sorted(produced)

    def rope(self, hidden: torch.Tensor, position_ids: torch.Tensor, *,
             shared_kv_states=None):
        if self.rotary_emb is None:
            return None  # attention computes rotary internally (MLA-style)
        if self.rotary_needs_layer_type:
            bundle = {"position_ids": position_ids}
            rope_types = sorted(
                name[: -len("_inv_freq")]
                for name in dir(self.rotary_emb)
                if name.endswith("_inv_freq") and not name.startswith("_"))
            # A yarn/scaled rope keeps auxiliary "*_original" inv_freq buffers
            # (deepseek_v4: compress_original, main_original) that are NOT valid
            # layer_type dispatch keys — self.rotary_emb.rope_type is only
            # {main, compress}. Calling rotary_emb(layer_type="compress_original")
            # raises KeyError (modeling_rope_utils indexes rope_type[layer_type]).
            # Restrict to the real dispatch keys the model itself uses.
            _valid = getattr(self.rotary_emb, "rope_type", None)
            if isinstance(_valid, dict):
                rope_types = [t for t in rope_types if t in _valid]
            if (rope_types
                    and getattr(self.text_config, "model_type", "")
                    .startswith("deepseek")):
                # MLA-family contract (deepseek_v4): attention consumes the
                # WHOLE dict — position_embeddings[self.rope_layer_type] —
                # so precompute every rope type's (cos, sin) once per walk.
                with torch.no_grad():
                    bundle["rope_dict"] = {
                        t: self.rotary_emb(hidden, position_ids,
                                           layer_type=t)
                        for t in rope_types}
            if self.needs_gemma4_masks:
                from transformers.masking_utils import (
                    create_causal_mask,
                    create_sliding_window_causal_mask,
                )

                mask_kwargs = {
                    "config": self.text_config,
                    "inputs_embeds": hidden,
                    "attention_mask": None,
                    "past_key_values": None,
                    "position_ids": position_ids,
                }
                bundle["attention_masks"] = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }
                bundle["shared_kv_states"] = (
                    shared_kv_states if shared_kv_states is not None
                    else UserDict())
            if self.needs_deepseek_masks:
                from transformers.masking_utils import (
                    create_sliding_window_causal_mask,
                )

                # One sliding-window causal mask serves every V4 layer type
                # (DeepseekV4Model.forward builds exactly one); rope arrives
                # as hidden [B,T,hc,H] but the mask helper wants [B,T,H].
                flat = hidden[..., 0, :] if hidden.dim() == 4 else hidden
                mask = create_sliding_window_causal_mask(
                    config=self.text_config, inputs_embeds=flat,
                    attention_mask=None, past_key_values=None,
                    position_ids=position_ids)
                bundle["attention_masks"] = {
                    t: mask for t in set(self.layer_types)}
            return bundle
        with torch.no_grad():
            return self.rotary_emb(hidden, position_ids)

    def run_block(self, L: int, hidden, position_embeddings, position_ids=None,
                  *, flow_keep=None, past_key_values=None, use_cache=False,
                  causal_length=None, prepared_attention_mask=None,
                  input_ids=None, shared_kv_states=None,
                  per_layer_input=None):
        """Forward decoder block L (1-based) on ``[B,n,H]`` states.

        ``flow_keep`` is the pipeline-v3 information-flow mask over the full
        key/value history (1 = ordinary token, 0 = censored privileged row).
        Censored query rows are zeroed before and after every block; attention
        layers also receive the corresponding key/value mask.  The explicit
        row zeroing is necessary for recurrent/linear-attention mixers, whose
        two-dimensional padding helper is ineffective at batch size one in
        current Transformers.
        """
        attention_mask = None
        layer_type = (
            self.layer_types[L - 1] if L - 1 < len(self.layer_types)
            else getattr(self.blocks[L - 1], "layer_type", None)
            or getattr(getattr(self.blocks[L - 1], "self_attn", None),
                       "layer_type", None)
        )
        if self.rotary_needs_layer_type and isinstance(position_embeddings, dict):
            bundle = position_embeddings
            position_ids = bundle["position_ids"]
            masks = bundle.get("attention_masks")
            if masks is not None:
                attention_mask = masks[layer_type]
            if shared_kv_states is None:
                shared_kv_states = bundle.get("shared_kv_states")
            if "rope_dict" in bundle:
                # deepseek-style: the attention module indexes the dict by
                # its own rope_layer_type; never collapse to one pair.
                position_embeddings = bundle["rope_dict"]
            else:
                with torch.no_grad():
                    position_embeddings = self.rotary_emb(
                        hidden, position_ids, layer_type=layer_type)
        local_keep = None
        keep_bcast = None
        if flow_keep is not None:
            flow_keep = flow_keep.to(hidden.device)
            local_keep = flow_keep[:, -hidden.shape[1]:].to(hidden.dtype)
            # [B, S] -> broadcastable over any trailing state dims (mHC
            # stream stacks are [B, S, hc_mult, H]).
            keep_bcast = local_keep.reshape(
                *local_keep.shape, *([1] * (hidden.dim() - 2)))
            hidden = hidden * keep_bcast
        if prepared_attention_mask is NO_PREPARED_ATTENTION_MASK:
            attention_mask = None
        elif prepared_attention_mask is not None:
            attention_mask = prepared_attention_mask.to(hidden.device)
        elif flow_keep is not None or past_key_values is not None:
            # Linear/recurrent mixers consume the ordinary 2-D keep mask.
            # Full/sliding attention needs the additive causal form and must
            # include the already-cached key/value length.
            if layer_type == "linear_attention":
                # Transformers applies this mask directly to the current
                # hidden chunk.  With cached BxK execution ``flow_keep`` also
                # contains the prefix and therefore cannot broadcast against
                # [B,K,H]; the prefix has already been committed to recurrent
                # state.  Mask only the current query rows.
                attention_mask = local_keep
            else:
                # A shared DynamicCache is updated layer by layer. Calling
                # Transformers' top-level mask helper *after* an earlier
                # layer has appended this same chunk makes that helper count
                # the chunk as past and doubles K (Q=85 -> K=170). The full
                # model avoids this by building its mask once before the
                # layer walk. Build the equivalent additive mask from the
                # caller-declared causal length, which is stable throughout
                # this block walk and never inspects another layer's cache.
                q_len = hidden.shape[1]
                kv_len = (flow_keep.shape[1] if flow_keep is not None
                          else causal_length)
                if kv_len is None:
                    raise ValueError(
                        "cached block execution needs causal_length")
                if kv_len < q_len:
                    raise ValueError(
                        f"causal_length {kv_len} shorter than query {q_len}")
                past_len = kv_len - q_len
                window = None
                if layer_type in ("sliding_attention", "chunked_attention",
                                  "compressed_sparse_attention",
                                  "heavily_compressed_attention"):
                    window = getattr(self.text_config, "sliding_window", None)
                    if window is None:
                        window = getattr(
                            self.text_config, "attention_chunk_size", None)
                # The configured cache class, rather than layer_type alone,
                # determines physical K length. A plain DynamicCache may grow
                # a full DynamicLayer lazily, while Gemma shared-KV configs
                # pre-create DynamicSlidingWindowLayer producers. Consumers
                # have no cache layer of their own and use the transported
                # producer tensor directly.
                physical_kv_len = None
                if (self.is_kv_shared_layer(L)
                        and shared_kv_states is not None
                        and layer_type in shared_kv_states):
                    physical_kv_len = int(
                        shared_kv_states[layer_type][0].shape[-2])
                elif past_key_values is not None:
                    cache_layer_index = self.shared_kv_source(L) - 1
                    cache_layers = getattr(past_key_values, "layers", ())
                    if cache_layer_index < len(cache_layers):
                        cache_layer = cache_layers[cache_layer_index]
                        get_sizes = getattr(cache_layer, "get_mask_sizes", None)
                        if get_sizes is not None:
                            physical_kv_len = int(get_sizes(q_len)[0])
                if physical_kv_len is not None:
                    if physical_kv_len < q_len:
                        raise ValueError(
                            f"physical KV length {physical_kv_len} shorter "
                            f"than query {q_len} at layer {L}")
                    kv_len = physical_kv_len
                    past_len = kv_len - q_len
                q_pos = torch.arange(
                    past_len, kv_len, device=hidden.device)[:, None]
                k_pos = torch.arange(kv_len, device=hidden.device)[None, :]
                allowed = k_pos <= q_pos
                # DeepSeek-V4's compressed layer types keep a sliding K=V
                # branch too (the compressor extends KV internally and
                # carries its own block_bias for those extra columns).
                if layer_type in ("sliding_attention", "chunked_attention",
                                  "compressed_sparse_attention",
                                  "heavily_compressed_attention"):
                    if window:
                        allowed &= k_pos > (q_pos - int(window))
                # Out-of-place: the batch dimension arrives by broadcasting.
                # The historical in-place `&=` on an .expand()ed view only
                # ever worked at B=1 (stride-0 views reject in-place writes);
                # batched flow_keep callers (pipeline-v4 relay) tripped it.
                if flow_keep is not None:
                    physical_keep = flow_keep[:, -kv_len:]
                    allowed = allowed[None] & physical_keep[:, None, :].bool()
                else:
                    allowed = allowed[None].expand(hidden.shape[0], -1, -1)
                attention_mask = torch.zeros(
                    allowed.shape, dtype=hidden.dtype, device=hidden.device)
                attention_mask.masked_fill_(
                    ~allowed, torch.finfo(hidden.dtype).min)
                attention_mask = attention_mask[:, None]
        kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": use_cache,
        }
        if self._accepts_input_ids[L - 1]:
            # DeepSeek-V4 hash-MoE layers route by a frozen token-id ->
            # expert table and CRASH on input_ids=None; every caller of a
            # deepseek block must supply the row-aligned token ids.
            kwargs["input_ids"] = input_ids
        if self._accepts_per_layer_input[L - 1]:
            kwargs["per_layer_input"] = per_layer_input
        if past_key_values is not None:
            if not self._accepts_past_key_values[L - 1]:
                raise NotImplementedError(
                    f"{type(self.blocks[L - 1]).__name__} does not expose "
                    "past_key_values; use history_policy=recompute_prefix")
            kwargs["past_key_values"] = past_key_values
        if self._accepts_shared_kv_states[L - 1]:
            if shared_kv_states is not None:
                kwargs["shared_kv_states"] = shared_kv_states
            else:
                if self._shared_kv_states is None:
                    self._shared_kv_states = {}
                kwargs["shared_kv_states"] = self._shared_kv_states
        if self.needs_deepseek_masks:
            # DeepSeek-V4 runs its HCA/FFN compressors in fp32 (Sinkhorn
            # stability) and hands fp32 `collapsed` to the bf16 attention/MLP
            # linears — the model is designed to run under autocast, which
            # casts the fp32 activation for the bf16 matmul. Our raw-bf16 walk
            # is not, so scope autocast around EVERY deepseek block forward:
            # store-fill, training, teacher capture, and eval alike (the single
            # choke point, so no call site can forget it).
            with torch.autocast(device_type=hidden.device.type,
                                dtype=torch.bfloat16):
                out = self.blocks[L - 1](hidden, **kwargs)
        else:
            out = self.blocks[L - 1](hidden, **kwargs)
        if keep_bcast is not None:
            out = out * keep_bcast
        return out

    def block_params(self, L: int) -> list[torch.nn.Parameter]:
        return self._block_params[L - 1]

    def loss_view(self, L: int, block_out: torch.Tensor) -> torch.Tensor:
        """What to compare against the cached teacher h{L}: raw block output,
        except the last layer, whose cached target is post-final-norm."""
        if L != self.n_layers:
            return block_out
        norm_parameter = next(self.final_norm.parameters(), None)
        if norm_parameter is not None and block_out.device != norm_parameter.device:
            block_out = block_out.to(norm_parameter.device, non_blocking=True)
        if self.hc_head is not None and block_out.dim() == 4:
            # mHC: collapse the hc_mult streams (frozen hyper-head) before
            # the shared final norm, mirroring DeepseekV4Model.forward.
            block_out = self.hc_head(block_out)
        return self.final_norm(block_out)
