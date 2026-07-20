"""In-process evaluation over the live pipeline-v4 stage owners.

This is evaluation pipeline parallelism, not the v4 training dataflow.  Every
rank enters the same ordered NCCL collectives at an epoch boundary.  Rank 0
tokenizes and embeds, each rank executes only its contiguous owned blocks, and
the last rank applies the frozen final norm / vocabulary head.  No foreign
block is materialized and no optimizer is reachable from this module.

Pipeline-v4.6 has no reconstructed-model fallback. Architecture-specific state
(rotated weights, shared KV, per-layer token inputs, hybrid caches and mHC
boundaries) is carried by the live owners or rejected as a launch defect before
the first evaluation collective.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import math
import os
import time
from collections import UserDict

import torch
import torch.nn.functional as F


def _tensor_digest(named_tensors) -> str:
    """Byte-exact digest for the small trainable adapter surface."""
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        # Stream large vocabulary matrices to host.  Byte exactness does not
        # require a multi-GB CPU copy (or a second full-size GPU contiguous
        # allocation) at an epoch boundary.
        for chunk in tensor.detach().reshape(-1).split(1 << 22):
            raw = chunk.contiguous().view(torch.uint8).cpu()
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


class DistributedBattery:
    """Synchronous live-weight standard scoring and greedy generation.

    Boundary tensors use a communicator dedicated to evaluation.  Broadcasts
    are deliberate: every rank executes an identical collective sequence, and
    a local compute failure is reduced before any peer enters the following
    payload collective.  This costs more fabric traffic than adjacent P2P but
    makes the failure protocol auditable and prevents a sibling waiting on a
    send that a failed owner never posted.
    """

    backend_name = "live_stage_owned_distributed_pp"
    weight_source = "live_stage_owned_weights"

    def __init__(self, cfg, stack, tokenizer, log, owned, *, rotator=None,
                 ds=None, cohorts=None, adapters_off=None):
        import torch.distributed as dist

        self.cfg = cfg
        self.stack = stack
        self.tokenizer = tokenizer
        self.log = log
        self.owned = owned
        self.ds = ds
        self.cohorts = cohorts
        self.adapters_off = adapters_off
        self.rotator = rotator
        self.device = torch.device(cfg.model.device)
        self.stage = int(cfg.train.v4_stage)
        self.stages = len(cfg.train.v4_stage_splits or []) + 1
        self.last_stage = self.stages - 1
        self.dist = dist
        self.timings: dict[str, float] = {}
        self._epoch = None

        timeout = datetime.timedelta(seconds=int(cfg.train.v4_nccl_timeout_s))
        if not dist.is_initialized():
            # Single-node launches historically needed no process group, so
            # the launcher leaves MASTER_* unset.  Derive a launch-specific
            # port to avoid colliding with another run on the same node.
            launch = os.environ.get("SELFUPDATE_V4_LAUNCH_ID", "v4")
            digest = int(hashlib.sha256(launch.encode()).hexdigest()[:8], 16)
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(20000 + digest % 20000))
            dist.init_process_group(
                "nccl", rank=self.stage, world_size=self.stages,
                timeout=timeout, device_id=self.device)
        if dist.get_rank() != self.stage or dist.get_world_size() != self.stages:
            raise RuntimeError(
                "distributed battery rank/world does not match v4 stage "
                f"ownership: pg={dist.get_rank()}/{dist.get_world_size()} "
                f"cfg={self.stage}/{self.stages}")
        # Never reuse the boundary or subprocess-adapter communicator.
        self.group = dist.new_group(
            ranks=list(range(self.stages)), backend="nccl", timeout=timeout)

    @property
    def is_writer(self) -> bool:
        return self.stage == 0

    def _failure_guard(self, label: str, fn):
        value = None
        error = None
        try:
            value = fn()
        except BaseException as exc:  # every sibling still reaches all_reduce
            error = exc
        failed = torch.tensor(
            [int(error is not None)], dtype=torch.int32, device=self.device)
        self.dist.all_reduce(failed, op=self.dist.ReduceOp.MAX,
                             group=self.group)
        if int(failed.item()):
            if error is not None:
                raise RuntimeError(
                    f"distributed battery {label} failed on stage "
                    f"{self.stage}: {error}") from error
            raise RuntimeError(
                f"distributed battery {label} failed on a sibling stage")
        return value

    def guard_phase(self, label: str, fn):
        """Run a rank-local phase before the next distributed payload.

        Evaluation helpers use this public spelling for decode, scoring and
        durable logging.  A failure is reduced here, while every rank is still
        at the same protocol boundary, rather than being discovered by an
        outer catch after a sibling has entered the next payload collective.
        """
        return self._failure_guard(label, fn)

    def _broadcast_header(self, values: list[int], src: int = 0) -> list[int]:
        n = len(values)
        header = self._failure_guard(
            "broadcast_header_prepare",
            lambda: (torch.tensor(values, dtype=torch.long, device=self.device)
                     if self.stage == src else
                     torch.empty(n, dtype=torch.long, device=self.device)))
        self.dist.broadcast(header, src=src, group=self.group)
        return [int(x) for x in header.tolist()]

    def _broadcast_tensor(self, tensor: torch.Tensor | None, *, src: int,
                          shape: tuple[int, ...], dtype) -> torch.Tensor:
        tensor = self._failure_guard(
            "broadcast_tensor_prepare",
            lambda: (torch.empty(shape, dtype=dtype, device=self.device)
                     if self.stage != src else
                     tensor.to(self.device).contiguous()))
        self.dist.broadcast(tensor, src=src, group=self.group)
        return tensor

    def _broadcast_rank0_inputs(self, input_ids, attention_mask, position_ids):
        if self.stage == 0:
            header = [input_ids.shape[0], input_ids.shape[1]]
        else:
            header = [0, 0]
        batch, width = self._broadcast_header(header)
        shape = (batch, width)
        ids = self._broadcast_tensor(
            input_ids if self.stage == 0 else None, src=0, shape=shape,
            dtype=torch.long)
        mask = self._broadcast_tensor(
            attention_mask if self.stage == 0 else None, src=0, shape=shape,
            dtype=torch.long)
        pos = self._broadcast_tensor(
            position_ids if self.stage == 0 else None, src=0, shape=shape,
            dtype=torch.long)
        return ids, mask, pos

    def _owned_adapter_digest(self) -> str:
        tensors = []
        for layer in self.owned:
            for name, param in self.stack.blocks[layer - 1].named_parameters():
                if param.requires_grad and not param.is_meta:
                    tensors.append((f"L{layer:03d}.{name}", param))
        return _tensor_digest(tensors)

    def _owned_adapter_count(self) -> int:
        return sum(
            int(param.requires_grad and not param.is_meta)
            for layer in self.owned
            for param in self.stack.blocks[layer - 1].parameters())

    def _expected_owned_adapter_keys(self) -> set[str]:
        # Stage-scoped loading leaves foreign tensors on meta, but the PEFT
        # parameter topology (and requires_grad flags) is still complete.  The
        # same key law is used by adapter publication/grafting.
        return {
            f"L{layer:03d}.{name}"
            for layer in self.owned
            for name, param in self.stack.blocks[layer - 1].named_parameters()
            if param.requires_grad
        }

    def _live_owned_adapter_keys(self) -> set[str]:
        return {
            f"L{layer:03d}.{name}"
            for layer in self.owned
            for name, param in self.stack.blocks[layer - 1].named_parameters()
            if param.requires_grad and not param.is_meta
        }

    def _assert_own_gpu_only(self) -> bool | None:
        """Tripwire for foreign contexts; None means it could not be verified."""
        import subprocess

        try:
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=10).stdout
            gpus = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=10).stdout
        except Exception:
            return None
        try:
            index_of = {}
            for line in gpus.strip().splitlines():
                index, uuid = (part.strip() for part in line.split(","))
                index_of[uuid] = int(index)
            own_rows = [
                (uuid, pid)
                for uuid, pid in
                (tuple(part.strip() for part in line.split(","))
                 for line in apps.strip().splitlines() if line.strip())
                if pid.isdigit() and int(pid) == os.getpid()
            ]
        except (TypeError, ValueError):
            return None
        foreign = sorted({
            index_of[uuid] for uuid, _pid in own_rows
            if uuid in index_of and index_of[uuid] != self.device.index
        })
        if foreign:
            raise RuntimeError(
                f"distributed battery stage {self.stage} opened foreign "
                f"CUDA devices {foreign}; owned device is {self.device.index}")
        if (self.device.index is None or not apps.strip() or not gpus.strip()
                or not own_rows):
            return None
        return True

    def _frozen_vocab_digest(self) -> str:
        """Byte-exact digest of the named frozen vocabulary surface."""
        tensors = []
        modules = [
            ("embedding", self.stack.embed_tokens),
            ("final_norm", self.stack.final_norm),
            ("lm_head", self.stack.lm_head),
            ("hc_head", getattr(self.stack, "hc_head", None)),
        ]
        modules.extend(
            (f"frozen_input_{index}", module)
            for index, module in enumerate(
                getattr(self.stack, "frozen_input_modules", [])))
        for prefix, module in modules:
            if module is None:
                continue
            for name, tensor in module.named_parameters():
                if tensor.is_meta:
                    raise RuntimeError(f"frozen vocabulary tensor is meta: {prefix}.{name}")
                tensors.append((f"{prefix}.{name}", tensor))
            for name, tensor in module.named_buffers():
                if tensor.is_meta:
                    raise RuntimeError(f"frozen vocabulary buffer is meta: {prefix}.{name}")
                tensors.append((f"{prefix}.{name}", tensor))
        return _tensor_digest(tensors)

    def _verify_entry(self, epoch: int) -> None:
        launch = os.environ.get("SELFUPDATE_V4_LAUNCH_ID", "")
        self._failure_guard(
            "launch_identity",
            lambda: (_ for _ in ()).throw(RuntimeError(
                "distributed battery requires SELFUPDATE_V4_LAUNCH_ID"))
            if not launch else None)
        launch_hash = int(hashlib.sha256(launch.encode()).hexdigest()[:15], 16)
        mine = torch.tensor(
            [epoch, self.owned.start, self.owned.stop - 1, launch_hash],
            dtype=torch.long, device=self.device)
        rows = [torch.empty_like(mine) for _ in range(self.stages)]
        self.dist.all_gather(rows, mine, group=self.group)
        expected_bounds = [0] + list(self.cfg.train.v4_stage_splits or []) + [
            self.stack.n_layers]
        for rank, row in enumerate(rows):
            got = [int(x) for x in row.tolist()]
            expected = [epoch, expected_bounds[rank] + 1,
                        expected_bounds[rank + 1], launch_hash]
            if got != expected:
                raise RuntimeError(
                    f"distributed battery epoch/launch/ownership mismatch at "
                    f"rank {rank}: got={got}, expected={expected}")
        def validate_adapter_keys():
            expected_keys = self._expected_owned_adapter_keys()
            live_keys = self._live_owned_adapter_keys()
            if live_keys != expected_keys:
                raise RuntimeError(
                    f"stage {self.stage} live adapter key-set mismatch: "
                    f"missing={sorted(expected_keys - live_keys)[:8]} "
                    f"unexpected={sorted(live_keys - expected_keys)[:8]}")
            return live_keys
        live_keys = self._failure_guard(
            "complete_adapter_keyset", validate_adapter_keys)
        local_count = len(live_keys)
        count = torch.tensor([local_count], dtype=torch.long,
                             device=self.device)
        counts = [torch.empty_like(count) for _ in range(self.stages)]
        self.dist.all_gather(counts, count, group=self.group)
        if any(int(value.item()) <= 0 for value in counts):
            raise RuntimeError(
                "distributed battery found a stage with no live trainable "
                f"adapter tensors: {[int(x.item()) for x in counts]}")
        local_digest = self._failure_guard(
            "adapter_fingerprint", self._owned_adapter_digest)
        digest_tensor = torch.tensor(
            list(bytes.fromhex(local_digest)),
            dtype=torch.uint8, device=self.device)
        digests = [torch.empty_like(digest_tensor) for _ in range(self.stages)]
        self.dist.all_gather(digests, digest_tensor, group=self.group)
        self.adapter_digests = [bytes(x.tolist()).hex() for x in digests]
        vocab_digest = self._failure_guard(
            "vocabulary_fingerprint", self._frozen_vocab_digest)
        vocab = torch.tensor(list(bytes.fromhex(vocab_digest)),
                             dtype=torch.uint8, device=self.device)
        vocabs = [torch.empty_like(vocab) for _ in range(self.stages)]
        self.dist.all_gather(vocabs, vocab, group=self.group)
        if any(not torch.equal(vocabs[0], other) for other in vocabs[1:]):
            raise RuntimeError(
                "embedding/final-norm/lm-head fingerprints differ across PP ranks")
        frozen_error = None
        modules = [
            ("embedding", self.stack.embed_tokens),
            ("final_norm", self.stack.final_norm),
            ("lm_head", self.stack.lm_head),
            ("hc_head", getattr(self.stack, "hc_head", None)),
        ]
        modules.extend(
            (f"frozen_input_{index}", module)
            for index, module in enumerate(
                getattr(self.stack, "frozen_input_modules", [])))
        for name, module in modules:
            if module is None:
                continue
            if any(parameter.requires_grad for parameter in module.parameters()):
                frozen_error = RuntimeError(
                    f"distributed battery requires frozen {name}")
                break
        self._failure_guard(
            "frozen_vocabulary_contract",
            lambda: (_ for _ in ()).throw(frozen_error)
            if frozen_error is not None else None)

    def _run_pipeline(self, input_ids: torch.Tensor | None,
                      attention_mask: torch.Tensor | None,
                      position_ids: torch.Tensor | None, *, caches=None,
                      use_cache: bool = False) -> torch.Tensor:
        input_ids, attention_mask, position_ids = self._broadcast_rank0_inputs(
            input_ids, attention_mask, position_ids)
        # Final-rank scoring needs the exact token targets, while rank 0 is
        # the only tokenizer owner.  Keep the just-broadcast tensor for the
        # duration of this backend call.
        self._last_input_ids = input_ids
        batch, width = input_ids.shape
        hidden_size = int(self.stack.text_config.hidden_size)
        hidden_dtype = self.stack.embed_tokens.weight.dtype
        hc_mult = int(getattr(self.stack, "hc_mult", 0) or 0)
        boundary_tail = ((hc_mult, hidden_size) if hc_mult > 1
                         else (hidden_size,))

        embedded = self._failure_guard(
            "embedding_and_frozen_layer_inputs",
            lambda: self.stack.embed_and_per_layer_inputs(input_ids)
            if self.stage == 0 else None)
        if self.stage == 0:
            hidden, per_layer_inputs = embedded
        else:
            hidden = per_layer_inputs = None
        hidden = self._broadcast_tensor(
            hidden if self.stage == 0 else None, src=0,
            shape=(batch, width, *boundary_tail), dtype=hidden_dtype)

        shared_types = (sorted(set(self.stack.layer_types))
                        if any(self.stack.is_kv_shared_layer(layer)
                               for layer in range(1, self.stack.n_layers + 1))
                        else [])
        shared_kv_states = UserDict()

        def synchronize_shared_kv(owner: int) -> None:
            """Broadcast Gemma's full-length shared-KV side channel.

            Every rank executes the same header/tensor sequence. The mapping is
            transient for this token step; persistent prefix state remains only
            in the producing owner's ordinary DynamicCache layer.
            """
            for layer_type in shared_types:
                present = (self.stage == owner
                           and layer_type in shared_kv_states)
                has_value = self._broadcast_header(
                    [int(present)] if self.stage == owner else [0],
                    src=owner)[0]
                if not has_value:
                    shared_kv_states.pop(layer_type, None)
                    continue
                if self.stage == owner:
                    key, value = shared_kv_states[layer_type]
                    header = list(key.shape)
                else:
                    key = value = None
                    header = [0, 0, 0, 0]
                shape = tuple(self._broadcast_header(header, src=owner))
                key = self._broadcast_tensor(
                    key, src=owner, shape=shape, dtype=hidden_dtype)
                value = self._broadcast_tensor(
                    value, src=owner, shape=shape, dtype=hidden_dtype)
                shared_kv_states[layer_type] = (key, value)

        executed = torch.zeros(
            self.stack.n_layers, dtype=torch.int32, device=self.device)
        if self.rotator is not None:
            self._failure_guard(
                "rotation_prefetch_first_owned_block",
                lambda: self.rotator.prefetch(self.owned.start))
        bounds = [0] + list(self.cfg.train.v4_stage_splits or []) + [
            self.stack.n_layers]
        for owner in range(self.stages):
            owner_start, owner_stop = bounds[owner], bounds[owner + 1]
            per_owner = None
            ple_dim = int(getattr(
                self.stack.text_config, "hidden_size_per_layer_input", 0) or 0)
            if ple_dim:
                ple_shape = (batch, width, owner_stop - owner_start, ple_dim)
                source_ple = (per_layer_inputs[:, :, owner_start:owner_stop]
                              if self.stage == 0 else None)
                per_owner = self._broadcast_tensor(
                    source_ple, src=0, shape=ple_shape, dtype=hidden_dtype)

            def compute_stage():
                nonlocal hidden
                if self.stage != owner:
                    return None
                rope = self.stack.rope(
                    hidden, position_ids,
                    shared_kv_states=shared_kv_states)
                owned_layers = list(self.owned)
                for local_index, layer in enumerate(owned_layers):
                    activated = False
                    try:
                        if self.rotator is not None:
                            self.rotator.activate(layer)
                            activated = True
                            if local_index + 1 < len(owned_layers):
                                self.rotator.prefetch(
                                    owned_layers[local_index + 1])
                        hidden = self.stack.run_block(
                            layer, hidden, rope, position_ids=position_ids,
                            flow_keep=attention_mask.bool(),
                            past_key_values=caches,
                            use_cache=use_cache,
                            causal_length=attention_mask.shape[1],
                            input_ids=input_ids,
                            shared_kv_states=shared_kv_states,
                            per_layer_input=(
                                per_owner[:, :, layer - owner_start - 1]
                                if per_owner is not None else None))
                        executed[layer - 1] += 1
                    finally:
                        if activated:
                            self.rotator.evict(layer)
                return hidden
            owner_hidden = self._failure_guard(
                f"stage_{owner}_blocks", compute_stage)
            hidden = self._broadcast_tensor(
                owner_hidden if self.stage == owner else None, src=owner,
                shape=(batch, width, *boundary_tail), dtype=hidden_dtype)
            def validate_shared_kv():
                if self.stage != owner:
                    return
                for layer_type, (key, value) in shared_kv_states.items():
                    if key.ndim != 4 or value.shape != key.shape:
                        raise RuntimeError(
                            f"shared KV {layer_type!r} must be matching 4-D "
                            f"tensors, got {key.shape}/{value.shape}")
                    if key.dtype != hidden_dtype or value.dtype != hidden_dtype:
                        raise RuntimeError(
                            f"shared KV {layer_type!r} dtype "
                            f"{key.dtype}/{value.dtype} != {hidden_dtype}")
            self._failure_guard(
                f"stage_{owner}_shared_kv_validation", validate_shared_kv)
            synchronize_shared_kv(owner)

        self.dist.all_reduce(executed, op=self.dist.ReduceOp.SUM,
                             group=self.group)
        if not torch.equal(executed, torch.ones_like(executed)):
            raise RuntimeError(
                "distributed battery layer execution count is not exactly one: "
                f"{executed.tolist()}")
        return hidden

    def token_lengths(self, tokenizer, texts: list[str], *,
                      add_special_tokens: bool = False) -> list[int]:
        """Collective token-length service; only rank 0 invokes tokenizer."""
        values = self._failure_guard(
            "token_lengths",
            lambda: [len(tokenizer.encode(
                        t, add_special_tokens=add_special_tokens))
                     for t in texts] if self.stage == 0 else None)
        return self._broadcast_header(values if self.stage == 0 else [0] * len(texts))

    def stop_token_id(self, tokenizer) -> int:
        """Collective stop-token service; tokenizer use stays on rank 0."""
        from ..chatfmt import stop_token_id

        value = self._failure_guard(
            "stop_token_id",
            lambda: stop_token_id(tokenizer) if self.stage == 0 else None)
        return self._broadcast_header(
            [int(value)] if self.stage == 0 else [0])[0]

    def score_pairs(self, tokenizer, pairs: list[tuple[str, str]],
                    batch_size: int) -> list[float]:
        """Teacher-forced normalized continuation log likelihood."""
        started = time.perf_counter()
        scores: list[float] = []
        for begin in range(0, len(pairs), batch_size):
            batch = pairs[begin:begin + batch_size]

            def prepare():
                if self.stage != 0:
                    return None
                texts = [prompt + choice for prompt, choice in batch]
                enc = tokenizer(
                    texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False)
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                pos = torch.arange(ids.shape[1], device=self.device)[None]
                pos = pos.expand(ids.shape[0], -1)
                starts, ends = [], []
                for prompt, choice in batch:
                    starts.append(len(tokenizer.encode(
                        prompt, add_special_tokens=False)))
                    ends.append(starts[-1] + len(tokenizer.encode(
                        choice, add_special_tokens=False)))
                return ids, mask, pos, starts, ends

            prepared = self._failure_guard("standard_tokenize", prepare)
            if self.stage == 0:
                ids, mask, pos, starts, ends = prepared
            else:
                ids = mask = pos = None
                starts = ends = [0] * len(batch)
            bounds = self._broadcast_header(
                ([x for pair in zip(starts, ends) for x in pair]
                 if self.stage == 0 else [0] * (2 * len(batch))))
            starts, ends = bounds[::2], bounds[1::2]
            hidden = self._run_pipeline(ids, mask, pos)

            def finish_scores():
                if self.stage != self.last_stage:
                    return None
                view = self.stack.loss_view(self.stack.n_layers, hidden)
                logits = self.stack.lm_head(view)
                local = []
                for row, (start, end) in enumerate(zip(starts, ends)):
                    if start <= 0 or end <= start:
                        local.append(-math.inf)
                        continue
                    row_logits = logits[row, start - 1:end - 1].float()
                    targets = self._last_input_ids[row, start:end].to(
                        row_logits.device)
                    nll = F.cross_entropy(
                        row_logits, targets, reduction="sum").item()
                    local.append(-nll / (end - start))
                return local

            local = self._failure_guard("standard_frozen_head", finish_scores)
            result = (torch.tensor(local, dtype=torch.float64,
                                   device=self.device)
                      if self.stage == self.last_stage else
                      torch.empty(len(batch), dtype=torch.float64,
                                  device=self.device))
            self.dist.broadcast(result, src=self.last_stage, group=self.group)
            scores.extend(float(x) for x in result.tolist())
        self.timings["standard_scoring_seconds"] = (
            self.timings.get("standard_scoring_seconds", 0.0)
            + time.perf_counter() - started)
        return scores

    def _new_cache(self):
        from transformers import DynamicCache

        return DynamicCache(config=self.stack.text_config)

    @staticmethod
    def _cache_layer_bytes(layer) -> int:
        """Count tensor state retained by one Transformers cache layer."""
        # Config-aware DynamicCache constructors pre-create cache-layer
        # objects.  Sliding layers contain a scalar window tensor even before
        # they have seen a token; that is immutable cache metadata, not
        # retained sequence state.  Ownership concerns initialized layers.
        if (hasattr(layer, "is_initialized")
                and not bool(layer.is_initialized)):
            return 0
        if (hasattr(layer, "is_conv_states_initialized")
                and hasattr(layer, "is_recurrent_states_initialized")
                and not (bool(layer.is_conv_states_initialized)
                         or bool(layer.is_recurrent_states_initialized))):
            return 0
        seen: set[int] = set()

        def walk(value) -> int:
            if torch.is_tensor(value):
                if id(value) in seen:
                    return 0
                seen.add(id(value))
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(walk(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(walk(item) for item in value)
            return 0

        return walk(vars(layer))

    def _assert_cache_ownership(self, cache) -> None:
        """Only this rank's persistent cache layers may retain state.

        Shared-KV consumer layers intentionally have no DynamicCache entry;
        their producer's full-length K/V is transported transiently during the
        same token step and the producer alone retains the prefix cache.
        """
        retained = {
            index + 1: self._cache_layer_bytes(layer)
            for index, layer in enumerate(cache.layers)
        }
        foreign = {layer: size for layer, size in retained.items()
                   if layer not in self.owned and size}
        missing = [layer for layer in self.owned
                   if not self.stack.is_kv_shared_layer(layer)
                   if retained.get(layer, 0) <= 0]
        if foreign or missing:
            raise RuntimeError(
                f"cache ownership violation at stage {self.stage}: "
                f"foreign_bytes={foreign}, owned_layers_without_state={missing}")

    def _adapter_disable_flags(self) -> tuple[bool, ...]:
        return tuple(
            bool(module.disable_adapters)
            for module in self.stack.model.modules()
            if hasattr(module, "disable_adapters"))

    def _cohort_inputs(self, cohort, *, censored: bool):
        """Prepare one fixed vLLM sequence on rank 0 only."""
        if self.stage != 0:
            return None, None, None
        if censored:
            ids = torch.zeros_like(cohort.teacher_ids)
            for row, index in enumerate(cohort.indices):
                source = torch.tensor(
                    self.ds.pairs[index].student_ids, dtype=torch.long)
                if source.numel() != cohort.t_len[row]:
                    raise RuntimeError(
                        f"{cohort.example_ids[row]}: censored sequence length "
                        f"{source.numel()} != teacher {cohort.t_len[row]}")
                ids[row, :source.numel()] = source
            mask = cohort.keep.long().clone()
        else:
            ids = cohort.teacher_ids.clone()
            rows = torch.arange(cohort.T)[None, :]
            lengths = torch.tensor(cohort.t_len)[:, None]
            mask = (rows < lengths).long()
        pos = torch.arange(cohort.T)[None].expand(len(cohort.indices), -1)
        return (ids.to(self.device), mask.to(self.device),
                pos.to(self.device))

    def _fixed_sequence_metric(self, cohort, student_hidden, teacher_hidden):
        """Final-rank output-distance and vLLM reproduction aggregates."""
        if self.stage != self.last_stage:
            return None
        from .teacher_output import teacher_output_eval_sums

        student_view = self.stack.loss_view(
            self.stack.n_layers, student_hidden)
        teacher_view = self.stack.loss_view(
            self.stack.n_layers, teacher_hidden)
        student_rows, teacher_rows, target_rows, lengths = [], [], [], []
        for row in range(len(cohort.indices)):
            marks = cohort.eval_rows[row].to(self.device)
            positions = cohort.qpos[row].index_select(
                0, cohort.eval_rows[row]).to(self.device)
            # ``marks`` documents the qpos-coordinate contract; the full
            # sequence forwards index by the corresponding absolute position.
            if marks.numel() != positions.numel():
                raise RuntimeError("evaluation-row coordinate mismatch")
            student_rows.append(student_view[row].index_select(0, positions))
            teacher_rows.append(teacher_view[row].index_select(0, positions))
            target_rows.append(cohort.eval_ids[row].to(self.device))
            lengths.append(int(positions.numel()))
        values = teacher_output_eval_sums(
            torch.cat(student_rows).detach().float(),
            torch.cat(teacher_rows).detach().float(),
            torch.cat(target_rows), self.stack.lm_head,
            answer_lengths=lengths)
        ce, kl, count, sm, tm, se, te = values
        return torch.stack((
            ce.double(), kl.double(),
            torch.tensor(float(count), dtype=torch.float64,
                         device=self.device),
            sm.double(), tm.double(), se.double(), te.double(),
            torch.tensor(float(len(lengths)), dtype=torch.float64,
                         device=self.device),
        ))

    def _log_fixed_sequence(self, *, epoch: int, censored: bool,
                            totals: torch.Tensor) -> None:
        if not self.is_writer:
            return
        ce, kl, count, sm, tm, se, te, answers = totals.tolist()
        count_d = max(count, 1.0)
        answer_d = max(answers, 1.0)
        common = {
            "epoch": epoch,
            "CE_eval_loss": ce / count_d,
            "KL_eval_loss": kl / count_d,
            "student_argmax_acceptance": sm / count_d,
            "teacher_argmax_acceptance": tm / count_d,
            "student_exact_seq_rate": se / answer_d,
            "teacher_exact_seq_rate": te / answer_d,
            "student_exact_seq_match_answers": int(se),
            "teacher_exact_seq_match_answers": int(te),
            "exact_seq_answer_count": int(answers),
            "answer_token_count": int(count),
            "dataset_item_count": len(self.ds.pairs),
            "dataset_coverage": "whole_training_set_once_per_call",
            "token_coverage": "every_teacher_realized_answer_token",
            "answer_only": True,
            "evaluation_only": True,
            "validation_subset": False,
            "used_for_backward": False,
            "optimizer_weight": 0.0,
            "aggregation": "token_weighted_mean",
            "inference_semantics": "teacher_forced_fixed_sequence_scoring",
            "autoregressive": False,
            "teacher_forced": True,
            "complete_student_trajectory": True,
            "uses_teacher_hidden_as_student_input_after_embedding": False,
            "teacher_states_used_only_as_scoring_targets": True,
            "stage_epoch_synchronized": True,
            "acceptance_semantics": (
                "argmax_on_teacher_forced_predictor_rows_against_vllm_ids"),
            "CE_target": "teacher_realized_answer_token_ids",
            "KL_direction": "uncensored_adapters_disabled_teacher_to_student",
            "teacher_reference_semantics": (
                "frozen_adapters_disabled_zero_run_teacher_trajectory"),
            "teacher_kv_state_source": (
                "frozen_adapters_disabled_zero_run_teacher_owned_layers"),
            "kv_execution_semantics": (
                "full_sequence_cache_construction_equivalent_to_prefill"),
            "vocabulary_head": "frozen",
            "evaluation_backend": self.backend_name,
            "adapter_epoch": epoch,
        }
        if censored:
            censorship = ("privileged_rows_hidden_and_zeroed"
                           if self.cfg.mask.compaction == "flow_mask"
                           else "intact_control_no_privileged_censorship")
            self.log.log(
                kind="student_trajectory_eval",
                trajectory="live_pp_censored_student_full_trajectory",
                censorship_state=censorship,
                comparison_role="b_censored_validation_during_local_training",
                adapters_enabled=True,
                student_kv_state_source=(
                    "current_adapters_enabled_censored_student_owned_layers"),
                **common)
        else:
            self.log.log(
                kind="vllm_teacher_forced_reproduction_eval",
                trajectory="live_pp_uncensored_zero_run_teacher_trajectory",
                censorship_state="uncensored_full_vllm_input_and_output",
                comparison_role="a_epoch_zero_teacher_forced_vllm_reproduction",
                adapters_enabled=False,
                student_kv_state_source=(
                    "frozen_adapters_disabled_zero_run_teacher_owned_layers"),
                trainer_argmax_acceptance=tm / count_d,
                trainer_exact_seq_rate=te / answer_d,
                **common)

    def evaluate_fixed_vllm_sequences(self, epoch: int, *,
                                      include_b: bool = True) -> None:
        """Run tests a and b synchronously over the live PP student.

        The reference is an adapters-disabled, uncensored forward through the
        same stage owners: the frozen zero-run teacher trajectory. Test a is
        recorded from that trajectory only at epoch zero. Test b enables the
        current adapters and applies the deployment censorship mask while
        retaining the zero-run teacher as its comparison target. Both paths
        are teacher-forced over the stored vLLM answer tokens.
        """
        if not self.cohorts or self.ds is None:
            return
        started = time.perf_counter()
        a_total = torch.zeros(8, dtype=torch.float64, device=self.device)
        b_total = torch.zeros(8, dtype=torch.float64, device=self.device)
        for cohort in self.cohorts:
            prepared = self._failure_guard(
                "uncensored_fixed_sequence_inputs",
                lambda c=cohort: self._cohort_inputs(c, censored=False))
            ids, mask, pos = prepared
            adapter_flags = self._adapter_disable_flags()
            teacher_ctx = (self.adapters_off() if self.adapters_off is not None
                           else contextlib.nullcontext())
            teacher_cache = self._failure_guard(
                "zero_teacher_cache_init", self._new_cache)
            with teacher_ctx:
                teacher_hidden = self._run_pipeline(
                    ids, mask, pos, caches=teacher_cache, use_cache=True)
            self._failure_guard(
                "zero_teacher_cache_ownership",
                lambda c=teacher_cache: self._assert_cache_ownership(c))
            self._failure_guard(
                "adapter_state_restoration",
                lambda: (_ for _ in ()).throw(RuntimeError(
                    "adapters-disabled teacher context did not restore state"))
                if self._adapter_disable_flags() != adapter_flags else None)
            if epoch == 0:
                local = self._failure_guard(
                    "zero_teacher_vllm_head",
                    lambda c=cohort, t=teacher_hidden:
                    self._fixed_sequence_metric(c, t, t))
                value = (local if self.stage == self.last_stage else
                         torch.empty(8, dtype=torch.float64,
                                     device=self.device))
                self.dist.broadcast(value, src=self.last_stage,
                                    group=self.group)
                a_total += value

            if include_b:
                prepared = self._failure_guard(
                    "censored_fixed_sequence_inputs",
                    lambda c=cohort: self._cohort_inputs(c, censored=True))
                ids, mask, pos = prepared
                student_cache = self._failure_guard(
                    "censored_student_cache_init", self._new_cache)
                student_hidden = self._run_pipeline(
                    ids, mask, pos, caches=student_cache, use_cache=True)
                self._failure_guard(
                    "censored_student_cache_ownership",
                    lambda c=student_cache: self._assert_cache_ownership(c))
                local = self._failure_guard(
                    "censored_fixed_sequence_frozen_head",
                    lambda c=cohort, s=student_hidden, t=teacher_hidden:
                    self._fixed_sequence_metric(c, s, t))
                value = (local if self.stage == self.last_stage else
                         torch.empty(8, dtype=torch.float64,
                                     device=self.device))
                self.dist.broadcast(value, src=self.last_stage,
                                    group=self.group)
                b_total += value
        self._failure_guard(
            "fixed_sequence_logging",
            lambda: (
                self._log_fixed_sequence(epoch=epoch, censored=False,
                                         totals=a_total)
                if epoch == 0 else None,
                self._log_fixed_sequence(epoch=epoch, censored=True,
                                         totals=b_total)
                if include_b else None,
            ))
        self.timings["fixed_sequence_validation_seconds"] = (
            time.perf_counter() - started)

    def generate_answers(self, tokenizer, prompts: list[str],
                         budgets: list[int], eos: int,
                         generation_batch: int, *,
                         timing_key: str = "recall_generation_seconds"):
        """Left-padded cached greedy generation with HF-compatible padding."""
        started = time.perf_counter()
        answers: list[str] = [""] * len(prompts)
        completion: list[dict] = [{} for _ in prompts]
        def configure_tokenizer():
            previous = (tokenizer.padding_side, tokenizer.pad_token)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return previous, int(tokenizer.pad_token_id)
        (was_padding, was_pad_token), pad = self._failure_guard(
            "recall_tokenizer_setup", configure_tokenizer)
        try:
            for begin in range(0, len(prompts), max(1, generation_batch)):
                chunk_prompts = prompts[begin:begin + max(1, generation_batch)]
                chunk_budgets = budgets[begin:begin + len(chunk_prompts)]

                def prepare():
                    if self.stage != 0:
                        return None
                    enc = tokenizer(
                        chunk_prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
                    ids = enc["input_ids"].to(self.device)
                    mask = enc["attention_mask"].to(self.device)
                    pos = mask.long().cumsum(-1) - 1
                    pos.masked_fill_(mask == 0, 1)
                    return ids, mask, pos

                prepared = self._failure_guard("recall_tokenize", prepare)
                if self.stage == 0:
                    ids, mask, pos = prepared
                else:
                    ids = mask = pos = None
                cache = self._failure_guard(
                    "recall_cache_init", self._new_cache)
                hidden = self._run_pipeline(
                    ids, mask, pos, caches=cache, use_cache=True)
                self._failure_guard(
                    "recall_prefill_cache_ownership",
                    lambda c=cache: self._assert_cache_ownership(c))

                def next_from_hidden():
                    if self.stage != self.last_stage:
                        return None
                    view = self.stack.loss_view(self.stack.n_layers, hidden)
                    return self.stack.lm_head(view[:, -1]).float().argmax(-1)

                next_token = self._failure_guard(
                    "recall_prefill_head", next_from_hidden)
                next_token = self._broadcast_tensor(
                    next_token if self.stage == self.last_stage else None,
                    src=self.last_stage, shape=(len(chunk_prompts),),
                    dtype=torch.long)
                unfinished = torch.ones(
                    len(chunk_prompts), dtype=torch.bool, device=self.device)
                budget_tensor = torch.tensor(
                    chunk_budgets, dtype=torch.long, device=self.device)
                generated = []
                max_budget = max(chunk_budgets)
                for step in range(max_budget):
                    active = unfinished & budget_tensor.gt(step)
                    emitted = torch.where(
                        active, next_token,
                        torch.full_like(next_token, pad))
                    generated.append(emitted)
                    unfinished = (active & emitted.ne(eos)
                                  & budget_tensor.gt(step + 1))
                    if step + 1 >= max_budget or not bool(unfinished.any().item()):
                        break
                    if self.stage == 0:
                        ids = emitted[:, None]
                        mask = torch.cat((mask, active[:, None].long()), dim=1)
                        pos = mask.long().cumsum(-1)[:, -1:] - 1
                    else:
                        ids = mask = pos = None
                    hidden = self._run_pipeline(
                        ids, mask, pos, caches=cache, use_cache=True)
                    self._failure_guard(
                        "recall_decode_cache_ownership",
                        lambda c=cache: self._assert_cache_ownership(c))
                    next_token = self._failure_guard(
                        "recall_decode_head", next_from_hidden)
                    next_token = self._broadcast_tensor(
                        next_token if self.stage == self.last_stage else None,
                        src=self.last_stage, shape=(len(chunk_prompts),),
                        dtype=torch.long)

                def decode_outputs():
                    generated_rows = torch.stack(generated, dim=1).tolist()
                    for row, (token_row, budget) in enumerate(
                            zip(generated_rows, chunk_budgets)):
                        token_row = token_row[:budget]
                        stopped = eos in token_row
                        decoded_ids = token_row[:token_row.index(eos)] \
                            if stopped else token_row
                        generated_tokens = (token_row.index(eos) + 1
                                            if stopped else len(token_row))
                        if self.stage == 0:
                            answers[begin + row] = tokenizer.decode(
                                decoded_ids, skip_special_tokens=True)
                        completion[begin + row] = {
                            "generated_tokens": generated_tokens,
                            "budget_tokens": budget,
                            "stopped": stopped,
                            "hard_cut": len(token_row) >= budget and not stopped,
                            "decoded_token_ids": decoded_ids,
                        }
                self._failure_guard("recall_decode_outputs", decode_outputs)
        finally:
            def restore_tokenizer():
                tokenizer.padding_side = was_padding
                if tokenizer.pad_token != was_pad_token:
                    tokenizer.pad_token = was_pad_token
            self._failure_guard("recall_tokenizer_restore", restore_tokenizer)
        self.timings[timing_key] = (
            self.timings.get(timing_key, 0.0)
            + time.perf_counter() - started)
        return answers, completion

    def evaluate_uncensored_vllm_generation(self, epoch: int) -> None:
        """Run a': adapters-on autoregression from the full RAG prompt."""
        limit = int(self.cfg.eval.vllm_uncensored_generation_limit)
        if limit <= 0 or self.ds is None:
            return
        from .recite import (character_error_rate, normalize_verse,
                             strip_think, teacher_prompt)

        records = self._failure_guard(
            "uncensored_control_preparation",
            lambda: [record for record in self.ds.records
                     if record.get("answer_text")][:limit])
        if not records:
            self._failure_guard(
                "uncensored_control_skipped_log",
                lambda: self.log.log(
                    kind="vllm_uncensored_autoregressive_control_skipped",
                    epoch=epoch,
                    reason="no_records_with_vllm_answer_text")
                if self.is_writer else None)
            return
        prompts, references = self._failure_guard(
            "uncensored_control_prompts",
            lambda: ([teacher_prompt(record) for record in records],
                     [record["answer_text"] for record in records]))
        ref_lengths = self.token_lengths(
            self.tokenizer, references, add_special_tokens=False)
        budgets = [length + int(
            self.cfg.eval.vllm_uncensored_max_extra_tokens)
                   for length in ref_lengths]
        eos = self.stop_token_id(self.tokenizer)
        started = time.perf_counter()
        answers, completion = self.generate_answers(
            self.tokenizer, prompts, budgets, eos,
            self.cfg.eval.generation_batch,
            timing_key="uncensored_vllm_generation_seconds")

        def score():
            if not self.is_writer:
                return None
            exact_text = 0
            exact_tokens = 0
            cer_sum = 0.0
            rows = []
            for record, reference, answer, detail in zip(
                    records, references, answers, completion):
                ref_norm = normalize_verse(reference)
                got_norm = normalize_verse(strip_think(answer))
                same_text = got_norm == ref_norm
                ref_ids = self.tokenizer.encode(
                    reference, add_special_tokens=False)
                same_tokens = detail["decoded_token_ids"] == ref_ids
                exact_text += int(same_text)
                exact_tokens += int(same_tokens)
                cer = character_error_rate(ref_norm, got_norm) \
                    if got_norm else 1.0
                cer_sum += cer
                rows.append({
                    "example_id": record["example_id"],
                    "exact_answer": same_text,
                    "exact_token_sequence": same_tokens,
                    "cer": cer,
                    "generated_tokens": detail["generated_tokens"],
                    "budget_tokens": detail["budget_tokens"],
                    "stopped": detail["stopped"],
                })
            return exact_text, exact_tokens, cer_sum, rows

        scored = self._failure_guard("uncensored_control_scoring", score)
        def log_control():
            if not self.is_writer:
                return
            exact_text, exact_tokens, cer_sum, rows = scored
            self.log.log(
                kind="vllm_uncensored_autoregressive_control",
                epoch=epoch,
                comparison_role="a_prime_uncensored_rag_vllm_reproduction",
                inference_semantics="autoregressive_greedy_rollout",
                autoregressive=True,
                teacher_forced=False,
                censorship_state="uncensored_full_rag_context",
                adapters_enabled=True,
                optimization_active=False,
                evaluation_only=True,
                evaluation_backend=self.backend_name,
                adapter_epoch=epoch,
                stage_epoch_synchronized=True,
                n=len(records),
                exact_answer_rate=exact_text / max(len(records), 1),
                exact_token_sequence_rate=exact_tokens / max(len(records), 1),
                mean_character_error_rate=cer_sum / max(len(records), 1),
                per_example=rows,
            )
        self._failure_guard("uncensored_control_logging", log_control)
        self.timings["uncensored_vllm_generation_seconds"] = (
            time.perf_counter() - started)

    def run_epoch(self, epoch: int, *, baseline, started_at: float):
        """Run one complete synchronous battery and restore exact module modes."""
        from ..train.telemetry import (_epoch_end_telemetry,
                                       _epoch_zero_telemetry)

        self._epoch = int(epoch)
        self.timings = {}
        total_started = time.perf_counter()
        self.dist.barrier(group=self.group)
        self._verify_entry(epoch)
        gpu_entry = self._failure_guard(
            "gpu_ownership_at_entry", self._assert_own_gpu_only)
        adapter_before = self._failure_guard(
            "pre_adapter_fingerprint", self._owned_adapter_digest)
        vocab_before = self._failure_guard(
            "pre_vocabulary_fingerprint", self._frozen_vocab_digest)
        rotation_before = None
        if self.rotator is not None:
            def rotation_entry():
                if self.rotator._staged or self.rotator._inflight:
                    raise RuntimeError(
                        "distributed battery requires a quiescent rotator at "
                        f"the epoch boundary; staged={sorted(self.rotator._staged)} "
                        f"inflight={sorted(self.rotator._inflight)}")
                return (self.rotator.stall_seconds,
                        self.rotator.h2d_bytes, self.rotator.pages)
            rotation_before = self._failure_guard(
                "rotation_entry", rotation_entry)

        def switch_to_eval():
            before = {module: bool(module.training)
                      for module in self.stack.model.modules()}
            self.stack.model.eval()
            return before

        mode_before = self._failure_guard("switch_to_eval", switch_to_eval)
        provenance = {
            "evaluation_backend": self.backend_name,
            "weight_source": self.weight_source,
            "adapter_epoch": epoch,
            "launch_identity": os.environ.get("SELFUPDATE_V4_LAUNCH_ID"),
            "stage_epoch_synchronized": True,
            "complete_student_trajectory": True,
            "foreign_blocks_materialized": False,
            "kv_cache_state_source": (
                "current_adapters_enabled_per_owned_layer_for_rollout"),
            "kv_cache_execution": "prefill_once_then_incremental_decode",
        }
        timing_detail = {}
        evaluation_error = None
        try:
            with torch.inference_mode():
                if epoch == 0 or self.cfg.train.v4_relay_every_cohorts:
                    self.evaluate_fixed_vllm_sequences(
                        epoch,
                        include_b=bool(
                            self.cfg.train.v4_relay_every_cohorts))
                if epoch == 0:
                    baseline = _epoch_zero_telemetry(
                        self.cfg, self.stack, self.tokenizer, self.log,
                        started_at, backend=self, writer=self.is_writer,
                        provenance=provenance, timings=timing_detail)
                else:
                    baseline = _epoch_end_telemetry(
                        self.cfg, self.stack, self.tokenizer, self.log,
                        epoch=epoch - 1, baseline=baseline,
                        started_at=started_at, backend=self,
                        writer=self.is_writer, provenance=provenance,
                        timings=timing_detail)
                self.evaluate_uncensored_vllm_generation(epoch)
        except BaseException as exc:
            evaluation_error = exc
        finally:
            # Direct flag restoration is intentional: Module.train() recurses
            # and would overwrite heterogeneous child states.
            try:
                for module, training in mode_before.items():
                    module.training = training
            except BaseException as exc:
                if evaluation_error is None:
                    evaluation_error = exc
            if self.rotator is not None:
                try:
                    self.rotator.quiesce()
                    stall0, bytes0, pages0 = rotation_before
                    self.timings["rotation_stall_seconds"] = (
                        self.rotator.stall_seconds - stall0)
                    self.timings["rotation_h2d_gb"] = (
                        self.rotator.h2d_bytes - bytes0) / 2**30
                    self.timings["rotation_pages"] = (
                        self.rotator.pages - pages0)
                    self.rotator.stall_seconds = stall0
                    self.rotator.h2d_bytes = bytes0
                    self.rotator.pages = pages0
                except BaseException as exc:
                    if evaluation_error is None:
                        evaluation_error = exc
        self._failure_guard(
            "evaluation_body",
            lambda: (_ for _ in ()).throw(evaluation_error)
            if evaluation_error is not None else None)

        mutation_error = None
        try:
            if self._owned_adapter_digest() != adapter_before:
                raise RuntimeError("trainable adapter mutated during evaluation")
            if self._frozen_vocab_digest() != vocab_before:
                raise RuntimeError("frozen vocabulary mutated during evaluation")
            if any(module.training != training
                   for module, training in mode_before.items()):
                raise RuntimeError("module train/eval mode was not restored exactly")
            if self.rotator is not None:
                if self.rotator._staged or self.rotator._inflight:
                    raise RuntimeError(
                        "rotator was not quiescent after evaluation")
        except BaseException as exc:
            mutation_error = exc
        self._failure_guard(
            "postcondition", lambda: (_ for _ in ()).throw(mutation_error)
            if mutation_error is not None else None)
        gpu_exit = self._failure_guard(
            "gpu_ownership", self._assert_own_gpu_only)
        self.dist.barrier(group=self.group)
        total = time.perf_counter() - total_started
        def log_summary():
            if not self.is_writer:
                return
            verified = gpu_entry is True and gpu_exit is True
            self.log.log(
                kind="distributed_battery",
                epoch=epoch,
                **provenance,
                standard_scoring_seconds=round(
                    self.timings.get("standard_scoring_seconds", 0.0), 3),
                recall_generation_seconds=round(
                    self.timings.get("recall_generation_seconds", 0.0), 3),
                fixed_sequence_validation_seconds=round(
                    self.timings.get("fixed_sequence_validation_seconds", 0.0), 3),
                uncensored_vllm_generation_seconds=round(
                    self.timings.get("uncensored_vllm_generation_seconds", 0.0), 3),
                total_boundary_seconds=round(total, 3),
                offload_seconds=0.0,
                model_load_seconds=0.0,
                adapter_graft_seconds=0.0,
                mode_restored_exactly=True,
                trainable_parameters_unchanged=True,
                frozen_vocabulary_unchanged=True,
                every_layer_executed_exactly_once_per_forward=True,
                cache_state_retained_for_owned_layers_only=True,
                shared_kv_transport=(
                    "nccl_transient_full_length_side_channel"
                    if any(self.stack.is_kv_shared_layer(layer)
                           for layer in range(1, self.stack.n_layers + 1))
                    else "not_present"),
                rotary_weight_residency=(
                    "page_owned_block_per_forward"
                    if self.rotator is not None else "resident"),
                rotation_stall_seconds=round(
                    self.timings.get("rotation_stall_seconds", 0.0), 3),
                rotation_h2d_gb=round(
                    self.timings.get("rotation_h2d_gb", 0.0), 3),
                rotation_pages=int(self.timings.get("rotation_pages", 0)),
                live_adapter_sha256_by_stage=self.adapter_digests,
                no_foreign_gpu_context=(True if verified else None),
                foreign_gpu_context_verification=(
                    "verified_no_foreign_context" if verified else "unverified"),
                **timing_detail,
            )
        self._failure_guard("distributed_battery_log", log_summary)
        return baseline

    def finalize(self) -> None:
        """Keep the evaluation communicator alive until every rank is done."""
        self.dist.barrier(group=self.group)
