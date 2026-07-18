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


def _named_parameters_all(module):
    """Preserve tied-weight aliases when the framework supports it."""
    try:
        return module.named_parameters(remove_duplicate=False)
    except TypeError:  # older torch/Transformers compatibility
        return module.named_parameters()


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
    def __init__(self, model, hook_free_walk: bool = False):
        self.model = model
        inner = _resolve_text_stack(model)
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
        self._accepts_input_ids = []
        for block in self.blocks:
            params = inspect.signature(block.forward).parameters
            self._accepts_past_key_values.append("past_key_values" in params)
            self._accepts_shared_kv_states.append("shared_kv_states" in params)
            self._accepts_input_ids.append("input_ids" in params)
        self.final_norm = inner.norm
        self.lm_head = lm_head_owner.lm_head
        self.n_layers = len(self.blocks)
        # Hook-free walk (explicit pipeline placement only): call each
        # block's pre-hook forward and do the boundary moves ourselves —
        # accelerate's per-call dispatch is ~8% of the PP2 walk (issues.md
        # 2026-07-10). Full-model forwards (evals, generate) keep their
        # hooks and are unaffected. Never engaged when a hook offloads
        # WEIGHTS (device_map=auto spill), where dispatch is load-bearing,
        # nor for per-layer-rope bundles (gemma4-style), which recompute
        # rope per block anyway.
        self.hook_free_walk = False
        self.block_devices = None
        self._block_calls = self.blocks
        self._pe_src = None
        self._pe_map: dict = {}
        # Fallback for architectures that accept shared_kv_states without a
        # per-layer rotary bundle. Gemma4 supplies a fresh mapping in rope();
        # this path is latent on the current models but must still initialize.
        self._shared_kv_states = None
        if hook_free_walk and not self.rotary_needs_layer_type:
            devices, calls, plain = [], [], True
            for b in self.blocks:
                p = next(b.parameters(), None)
                devices.append(p.device if p is not None else torch.device("cpu"))
                hook = getattr(b, "_hf_hook", None)
                if hook is not None and getattr(hook, "offload", False):
                    plain = False
                calls.append(getattr(b, "_old_forward", None) or b)
            if plain:
                self.hook_free_walk = True
                self.block_devices = devices
                self._block_calls = calls

    def _pos_emb_on(self, pe, dev):
        """Per-device cache of the rope tensors for the CURRENT positional
        context (keyed by identity; a new rope() output resets the map, and
        the held reference makes id-reuse impossible while cached)."""
        if pe is None or not isinstance(pe, tuple):
            return pe
        if pe is not self._pe_src:
            self._pe_src = pe
            self._pe_map = {}
        got = self._pe_map.get(dev)
        if got is None:
            got = tuple(t.to(dev, non_blocking=True) if t.device != dev else t
                        for t in pe)
            self._pe_map[dev] = got
        return got

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

    def rope(self, hidden: torch.Tensor, position_ids: torch.Tensor):
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
                bundle["shared_kv_states"] = UserDict()
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
                  input_ids=None):
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
        shared_kv_states = None
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
            shared_kv_states = bundle.get("shared_kv_states")
            if "rope_dict" in bundle:
                # deepseek-style: the attention module indexes the dict by
                # its own rope_layer_type; never collapse to one pair.
                position_embeddings = bundle["rope_dict"]
            else:
                with torch.no_grad():
                    position_embeddings = self.rotary_emb(
                        hidden, position_ids, layer_type=layer_type)
        if self.hook_free_walk:
            dev = self.block_devices[L - 1]
            if hidden.device != dev:
                hidden = hidden.to(dev, non_blocking=True)
            position_embeddings = self._pos_emb_on(position_embeddings, dev)
            if torch.is_tensor(position_ids) and position_ids.device != dev:
                position_ids = position_ids.to(dev, non_blocking=True)
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
                    window = getattr(self.text_config, "sliding_window", None)
                    if window is None:
                        window = getattr(
                            self.text_config, "attention_chunk_size", None)
                    if window:
                        allowed &= k_pos > (q_pos - int(window))
                # Out-of-place: the batch dimension arrives by broadcasting.
                # The historical in-place `&=` on an .expand()ed view only
                # ever worked at B=1 (stride-0 views reject in-place writes);
                # batched flow_keep callers (pipeline-v4 relay) tripped it.
                if flow_keep is not None:
                    allowed = allowed[None] & flow_keep[:, None, :].bool()
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
        out = self._block_calls[L - 1](hidden, **kwargs)
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

    # -- PPn model-adapter surface ---------------------------------------

    @property
    def ordered_text_blocks(self):
        """The complete ordered text-block list owned by this stack."""
        return tuple(self.blocks)

    def legal_cut_positions(self) -> tuple[int, ...]:
        """Return conservative contiguous cuts for arbitrary-stage PP.

        A model exposing shared KV state has an undeclared cross-layer edge;
        until its adapter supplies a dependency-aware state transport, no
        cut is legal.  Ordinary full/linear Qwen blocks are private and may
        be cut at every block boundary.
        """
        if any(getattr(self, "_accepts_shared_kv_states", ())):
            return ()
        return tuple(range(1, self.n_layers))

    def new_causal_state(self, *, max_cache_len: int | None = None):
        """Construct a state object for adapter users without importing v3."""
        from transformers import DynamicCache, StaticCache

        if max_cache_len is None:
            return DynamicCache(config=self.text_config)
        return StaticCache(config=self.text_config, max_cache_len=max_cache_len)

    def detach_causal_state(self, state) -> None:
        """Detach state tensors recursively after a stage-local tile write."""
        def detach(value):
            if torch.is_tensor(value):
                return value.detach()
            if isinstance(value, tuple):
                return tuple(detach(item) for item in value)
            if isinstance(value, list):
                return [detach(item) for item in value]
            if isinstance(value, dict):
                return {key: detach(item) for key, item in value.items()}
            return value

        for layer in getattr(state, "layers", ()):
            for name, value in vars(layer).items():
                detached = detach(value)
                if detached is not value:
                    setattr(layer, name, detached)

    def parameter_ownership(self) -> dict[str, str]:
        """Map every named parameter to one block or a frozen owner."""
        block_ids = {
            id(parameter): index
            for index, block in enumerate(self.blocks, start=1)
            for parameter in block.parameters()
        }
        result = {}
        for name, parameter in _named_parameters_all(self.model):
            index = block_ids.get(id(parameter))
            result[name] = f"block:{index}" if index is not None else "frozen_vocabulary_or_input"
        return result

    def checkpoint_ownership(self) -> dict[str, object]:
        """Describe stage-owned tensors and tied frozen-vocabulary aliases."""
        aliases: dict[int, list[str]] = {}
        for name, parameter in _named_parameters_all(self.model):
            aliases.setdefault(id(parameter), []).append(name)
        return {
            "parameter_ownership": self.parameter_ownership(),
            "tied_weight_aliases": [names for names in aliases.values()
                                    if len(names) > 1],
            "frozen_modules": ["embed_tokens", "final_norm", "lm_head"],
        }
