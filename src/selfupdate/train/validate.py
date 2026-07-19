"""Dispatch-time validation of the experiment's knob schedule.

Knob-flow law (2026-07-05): a knob that a schedule does not implement must
RAISE, never silently ignore — spec/code divergence is the bug class that
produced the unwired-ce_kind incident. Since the stored tests were retired
(owner decision 2026-07-11), this module and the prose laws in CLAUDE.md ARE
the specification; ``scripts/audit_configs.py`` sweeps every config through
it offline.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..eval.tasks import RECALL_CORPUS_PATHS
from ..eval.teacher_output import EVALUATION_ONLY_OUTPUT_NAMES
from ..teacher.cache import resolved_node_epoch0_root
from .ppn import _strict_splits
from .runtime import uses_pipeline_map

RUN_CLASSES = {
    "method", "teacher_reference", "ablation", "control",
    "legacy_archive", "confounded", "open",
}
NON_METHOD_CLASSES = RUN_CLASSES - {"method"}


def validate_knob_schedule(cfg) -> None:
    sched = cfg.train.schedule
    run_class = cfg.train.run_class
    bad = []
    if getattr(cfg, "layerwise_project_version", "3.4") != "3.4":
        bad.append(
            "layerwise_project_version must be the separate project identity '3.4'")
    if cfg.train.pp_execution not in ("serial", "wavefront", "independent"):
        bad.append(
            "train.pp_execution must be serial, wavefront, or independent")
    if not 0 < cfg.train.partition_safety_margin <= 1:
        bad.append("train.partition_safety_margin must be in (0, 1]")
    pipeline_splits = list(getattr(cfg.model, "pipeline_splits", []) or [])
    if pipeline_splits:
        try:
            _strict_splits(max(pipeline_splits) + 1, pipeline_splits)
        except ValueError as exc:
            bad.append(str(exc))
    if (getattr(cfg.model, "pipeline_split", 0)
            and pipeline_splits):
        bad.append("model.pipeline_split and model.pipeline_splits are mutually exclusive")
    configured_devices = list(getattr(cfg.model, "pipeline_devices", []) or [])
    configured_stages = len(pipeline_splits) + 1
    if getattr(cfg.model, "pipeline_split", 0):
        configured_stages = 2
    if configured_devices and len(configured_devices) != configured_stages:
        bad.append("model.pipeline_devices must contain one id per PP stage")
    if getattr(cfg.model, "pipeline_world_size", 0) and (
            cfg.model.pipeline_world_size != configured_stages):
        bad.append("model.pipeline_world_size does not match configured PP stages")
    if cfg.train.partition_profile_path and not Path(
            cfg.train.partition_profile_path).is_file():
        bad.append(
            f"partition profile does not exist: {cfg.train.partition_profile_path}")
    if cfg.train.auto_partition and not cfg.train.partition_profile_path:
        bad.append("train.auto_partition requires train.partition_profile_path")
    if cfg.train.pp_execution == "wavefront":
        if cfg.train.pipeline_version != 3 or cfg.train.pipeline_revision != "3.2":
            bad.append("PPn wavefront requires the preserved pipeline-v3.2 protocol")
        if cfg.train.history_policy != "causal_bk":
            bad.append("PPn wavefront requires history_policy=causal_bk")
        if cfg.train.update_granularity != "online":
            bad.append("PPn wavefront requires update_granularity=online")
        if cfg.train.stale_gradient_window <= 0:
            bad.append("PPn wavefront requires a positive BxK tile width")
        if cfg.train.grad_accum != 1:
            bad.append("PPn wavefront requires grad_accum=1")
        if cfg.train.schedule != "summed":
            bad.append("PPn wavefront requires schedule=summed")
        if pipeline_splits and not cfg.train.partition_profile_id:
            bad.append(
                "multi-stage PPn wavefront requires a pinned partition_profile_id")
    if cfg.train.hidden_loss in EVALUATION_ONLY_OUTPUT_NAMES:
        raise ValueError(
            f"train.hidden_loss={cfg.train.hidden_loss!r} is forbidden: "
            "CE-eval-loss and KL-eval-loss are evaluation-only measurements "
            "over the whole training-set traversal and are NEVER used for "
            "backward or optimizer updates")
    if cfg.train.pipeline_version not in (1, 2, 3, 4):
        raise ValueError("train.pipeline_version must be 1, 2, 3, or 4")
    if cfg.cache.runtime_policy not in ("durable", "node_epoch0"):
        raise ValueError(
            "cache.runtime_policy must be durable or node_epoch0")
    if cfg.cache.generation_max_tokens < 0:
        raise ValueError("cache.generation_max_tokens must be non-negative")
    if cfg.cache.item_cache_items < 0:
        raise ValueError("cache.item_cache_items must be non-negative")
    if (cfg.cache.runtime_policy == "node_epoch0"
            and not cfg.cache.node_root.startswith("/dev/shm/")):
        raise ValueError(
            "cache.runtime_policy=node_epoch0 requires cache.node_root under "
            "/dev/shm so the cache is host-local shared RAM")
    if cfg.cache.runtime_policy == "node_epoch0":
        # The environment override wins at runtime, so validate the same
        # resolved value rather than allowing it to bypass the config guard.
        resolved_node_epoch0_root(cfg)
    if cfg.train.update_granularity not in (
        "legacy_answer_sum", "answer", "token", "grid", "online",
    ):
        raise ValueError(
            f"unknown train.update_granularity {cfg.train.update_granularity!r}")
    if (cfg.train.pipeline_version != 3
            and cfg.train.lr_epoch_multipliers):
        bad.append(
            "lr_epoch_multipliers is implemented only by pipeline-v3.1")
    if cfg.train.pipeline_version == 4:
        # v4 = blockwise teacher-forced training with frozen teacher KV.
        # Both the block input and the attention context are teacher-fixed,
        # so layers are independent; multi-GPU is layer-sharding via
        # v4_stage_splits/-devices, never the PP device_map.
        if cfg.train.pipeline_revision not in ("", "4.0"):
            bad.append("pipeline-v4 revision must be 4.0")
        if cfg.train.trajectory_source != "teacher_hidden":
            bad.append("pipeline-v4 is teacher-forced per layer: "
                       "trajectory_source must be teacher_hidden")
        if cfg.train.update_granularity != "online":
            bad.append("pipeline_version=4 requires update_granularity=online")
        if sched != "summed":
            bad.append("pipeline-v4 uses the summed layer objective")
        if cfg.train.grad_accum != 1:
            bad.append("pipeline-v4 requires grad_accum=1 (one write per "
                       "block per cohort is the update law)")
        if cfg.train.v4_teacher_source not in ("cache", "online", "store"):
            bad.append("v4_teacher_source must be cache, online, or store")
        if (cfg.train.v4_teacher_source == "cache"
                and not cfg.cache.store_full_teacher_inputs):
            bad.append("pipeline-v4 cache source needs the full-prefix "
                       "teacher cache: cache.store_full_teacher_inputs=true")
        if cfg.train.v4_teacher_source == "online":
            if cfg.train.v4_loop_order != "item_major":
                bad.append("v4_teacher_source=online requires "
                           "v4_loop_order=item_major (one teacher forward per "
                           "cohort; layer_major would redo it per layer)")
        if cfg.train.v4_teacher_source in ("online", "store"):
            if cfg.train.v4_kv_source != "teacher_frozen":
                bad.append("v4_teacher_source=online/store implements "
                           "teacher_frozen only: the recorded KV/store IS "
                           "the frozen teacher; student_refresh would "
                           "rebuild it from weights that keep moving")
            if not cfg.train.lora.enabled:
                bad.append("v4_teacher_source=online/store derives the "
                           "teacher by disabling adapters: requires "
                           "train.lora.enabled")
        if (cfg.train.v4_teacher_source == "store"
                and cfg.train.v4_teacher_residency == "rebuild"):
            bad.append("v4_teacher_source=store is FILL-ONCE (store-fill); "
                       "residency=rebuild would drop the stored entries "
                       "and there is no per-epoch teacher recompute to rebuild from")
        if "deepseek" in (cfg.model.name or "").lower():
            # Plan B8 phase A: the frozen-context adapter (deepseek_ctx.py)
            # serves sliding K=V + compressed entries + teacher-forced
            # indexer routing from a per-(layer, cohort) online record.
            # store lane implemented 2026-07-18: _fill_deepseek_layer
            # records typed-cache artifacts during the store-fill walk
            # (entries stay stage-local; the relay still carries only
            # boundary hiddens) and the step serves FrozenDeepseekCtx.
            if cfg.train.v4_kv_source == "student_refresh":
                bad.append("deepseek-v4 requires v4_kv_source=teacher_frozen:"
                           " compressed entries and indexer selection are "
                           "key-side and must stay teacher-recorded")
            if cfg.train.expert_routing_source != "black_box":
                bad.append("deepseek-v4 MoE supports black_box routing only "
                           "(no MoEController router adapter yet; hash-MoE "
                           "layers are id-routed and identical either way)")
        if cfg.mask.compaction not in ("flow_mask", "intact"):
            bad.append("pipeline-v4 censorship is attention censorship: "
                       "flow_mask (method) or intact (diagnostic control)")
        if cfg.train.v4_kv_source not in ("teacher_frozen", "student_refresh"):
            bad.append("v4_kv_source must be teacher_frozen or student_refresh")
        if (cfg.train.v4_kv_source == "student_refresh"
                and cfg.train.v4_kv_refresh_epochs <= 0):
            bad.append("v4_kv_source=student_refresh requires "
                       "v4_kv_refresh_epochs > 0")
        if (cfg.train.v4_kv_source == "teacher_frozen"
                and cfg.train.v4_kv_refresh_epochs):
            bad.append("v4_kv_refresh_epochs is set but v4_kv_source is "
                       "teacher_frozen (frozen KV never refreshes)")
        if cfg.train.v4_optimizer not in ("immediate_sgd", "adam"):
            bad.append("v4_optimizer must be immediate_sgd or adam")
        if cfg.train.v4_grad_clip < 0:
            bad.append("v4_grad_clip must be >= 0 (0 disables clipping)")
        if cfg.train.v4_weight_residency not in ("resident", "rotate",
                                                 "auto"):
            bad.append("v4_weight_residency must be resident, rotate, or "
                       "auto")
        if cfg.train.v4_weight_residency != "resident":
            if not cfg.train.v4_stage_scoped:
                bad.append("v4_weight_residency rotate/auto is the scaling "
                           "lane and requires v4_stage_scoped")
            if cfg.train.v4_loop_order != "layer_major":
                bad.append("weight rotation requires v4_loop_order="
                           "layer_major: item_major pages every owned "
                           "block every cohort (~4 TB/epoch at 397B)")
        if cfg.train.v4_battery_mode not in ("graft", "subprocess"):
            bad.append("v4_battery_mode must be graft or subprocess")
        if (cfg.train.v4_battery_mode == "subprocess"
                and cfg.train.v4_stage < 0
                and not cfg.train.v4_stage_scoped):
            bad.append("v4_battery_mode=subprocess coordinates a staged "
                       "launch; a single-process RESIDENT run probes "
                       "directly (scoped single-process — rotary PPP1 — "
                       "requires it)")
        if cfg.train.v4_stage_scoped:
            if cfg.train.v4_battery_mode != "subprocess":
                bad.append("v4_stage_scoped cannot graft (stage 0 lacks "
                           "the full model): set v4_battery_mode=subprocess")
            if (cfg.train.v4_stage < 0
                    and cfg.train.v4_weight_residency not in
                    ("rotate", "auto")):
                bad.append("single-process v4_stage_scoped is the rotary "
                           "PPP1 lane (owner demo, 2026-07-18): every "
                           "block CPU-mastered and paged through ONE "
                           "device — requires v4_weight_residency rotate "
                           "(or auto); a resident single process should "
                           "load the full model instead")
            if not cfg.train.lora.enabled:
                bad.append("v4_stage_scoped requires lora (full-FT master "
                           "weights are not stage-assembled)")
            if cfg.train.v4_teacher_source == "online":
                bad.append("v4_stage_scoped cannot use v4_teacher_source="
                           "online: the per-cohort teacher forward walks EVERY "
                           "layer and foreign blocks are meta — use cache "
                           "(or the store-fill relay when it lands)")
        betas = tuple(cfg.train.v4_adam_betas)
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            bad.append("v4_adam_betas must be two values in [0, 1)")
        if cfg.train.v4_adam_eps <= 0:
            bad.append("v4_adam_eps must be > 0")
        if cfg.train.v4_adam_weight_decay < 0:
            bad.append("v4_adam_weight_decay must be >= 0")
        if cfg.train.v4_loop_order not in ("layer_major", "item_major"):
            bad.append("v4_loop_order must be layer_major or item_major")
        if cfg.train.v4_loss_positions not in (
                "answer", "aligned", "thinking_answer"):
            bad.append("v4_loss_positions must be answer, aligned, or "
                       "thinking_answer")
        if cfg.train.v4_teacher_residency not in (
                "auto", "gpu_corpus", "cpu_stream", "rebuild"):
            bad.append("v4_teacher_residency must be auto, gpu_corpus, "
                       "cpu_stream, or rebuild")
        if cfg.train.v4_relay_every_cohorts < 0:
            bad.append("v4_relay_every_cohorts must be >= 0")
        if cfg.model.pipeline_split or cfg.model.pipeline_splits:
            bad.append(
                "pipeline-v4 processes each load the full model on one card; "
                "model.pipeline_split(s) drive the PP device_map loader and "
                "must stay empty — layer ownership is train.v4_stage_splits")
        splits = list(cfg.train.v4_stage_splits or [])
        if any(left >= right for left, right in zip(splits, splits[1:])):
            bad.append(
                f"v4_stage_splits must be strictly increasing: {splits}")
        if splits and splits[0] <= 0:
            bad.append("v4_stage_splits are one-based block cuts and must "
                       "be positive")
        stages = len(splits) + 1
        devices = list(cfg.train.v4_stage_devices or [])
        if devices and len(devices) != stages:
            bad.append("v4_stage_devices must name one physical id per stage")
        # Device uniqueness is HOST-scoped (Multi-Node Conventions): a
        # cross-node stage map (SELFUPDATE_V4_STAGE_HOSTS) may legally
        # reuse a physical id on different hosts (PPP5 2026-07-18:
        # agpuh01 GPU 3 + agpuh02 GPUs 0-3).
        import os as _os
        hosts = (_os.environ.get("SELFUPDATE_V4_STAGE_HOSTS", "").split()
                 or ["local"] * len(devices))
        if len(hosts) != len(devices):
            hosts = ["local"] * len(devices)
        pairs = list(zip(hosts, devices))
        if len(set(pairs)) != len(pairs):
            bad.append("v4_stage_devices must be unique physical ids per "
                       "host (host, device) pairs collide")
        if not -1 <= cfg.train.v4_stage < stages:
            bad.append(
                f"v4_stage {cfg.train.v4_stage} outside -1..{stages - 1}")
        if cfg.train.conn_window > 1:
            bad.append("pipeline-v4 is strictly block-local: conn_window <= 1")
        if cfg.train.window_dedup:
            bad.append("pipeline-v4 has no connected windows to deduplicate")
        if cfg.train.moe_mode != "dense_or_black_box":
            # v4 routing interventions ride the online teacher forward: the
            # adapters-off per-cohort forward that produces the teacher
            # hiddens also records every wrapped router's top-k (and
            # log-probs for router_aligned). The cache stores no routing,
            # and layer_major has no per-cohort teacher forward to hook.
            if cfg.train.v4_teacher_source != "online":
                bad.append("pipeline-v4 MoE routing interventions "
                           "(teacher_forced/router_aligned) record teacher "
                           "routing during the online adapters-off forward; "
                           "set v4_teacher_source=online")
        if (cfg.train.hidden_loss.startswith("delta_")
                or cfg.train.hidden_loss == "multi_delta_nmse"):
            bad.append("pipeline-v4 supports absolute local hidden losses "
                       "only")
        if cfg.train.online_teacher or cfg.train.frozen_teacher_copy:
            bad.append("pipeline-v4 is cache-driven: no online teacher or "
                       "frozen teacher copy")
        if cfg.train.offload_adam:
            bad.append("offload_adam is the v1/v2 summed knob; pipeline-v4 "
                       "Adam is v4_optimizer=adam")
    elif cfg.train.v4_stage != -1 or cfg.train.v4_stage_splits:
        bad.append("v4_stage/v4_stage_splits are set but pipeline_version "
                   "is not 4")
    if cfg.train.pipeline_version == 3:
        if cfg.train.pipeline_revision not in ("", "3.0", "3.1", "3.2"):
            bad.append("pipeline-v3 revision must be 3.0, 3.1, or 3.2")
        bk_probe = cfg.train.history_policy == "causal_bk_probe"
        bk_training = cfg.train.history_policy == "causal_bk"
        bk_execution = bk_probe or bk_training
        if cfg.train.update_granularity != "online":
            bad.append("pipeline_version=3 requires update_granularity=online")
        if sched != "summed":
            bad.append("pipeline-v3 online execution uses the summed forward layer walk")
        if cfg.train.grad_accum != 1:
            bad.append("pipeline-v3 requires grad_accum=1")
        if bk_execution:
            if cfg.train.pipeline_revision not in ("3.1", "3.2"):
                bad.append(
                    "causal_bk execution requires pipeline_revision=3.1 or 3.2")
            if cfg.train.micro_batch < 2:
                bad.append("causal_bk execution requires micro_batch B >= 2")
            if cfg.train.batching not in ("padded", "bucketed"):
                bad.append("causal_bk execution requires padded or bucketed batching")
            shard_users = cfg.train.activation_shard_users
            if shard_users <= 0 or shard_users > cfg.train.micro_batch:
                bad.append(
                    "activation_shard_users must be in [1, micro_batch] "
                    "for causal_bk")
            if cfg.train.prefill_query_chunk <= 0:
                bad.append("causal_bk requires prefill_query_chunk > 0")
            if cfg.train.prefill_parallel_shards < 1:
                bad.append("causal_bk prefill_parallel_shards must be positive")
            if cfg.train.prefill_parallel_shards > 1:
                bad.append(
                    "prefill_parallel_shards > 1 is disabled: concurrent "
                    "CUDA prefill enters Triton autotuners with shared mutable "
                    "state (measured TypeError in the 27B PP4 trial)")
        else:
            if cfg.train.activation_shard_users:
                bad.append(
                    "activation_shard_users is implemented only by causal_bk")
            if cfg.train.prefill_query_chunk != 64:
                bad.append("prefill_query_chunk is implemented only by causal_bk")
            if cfg.train.prefill_parallel_shards != 1:
                bad.append(
                    "prefill_parallel_shards is implemented only by causal_bk")
            if cfg.train.micro_batch != 1:
                bad.append("pipeline-v3.0 requires micro_batch=1")
            if cfg.train.batching != "item":
                bad.append("pipeline-v3.0 requires batching=item")
        if cfg.train.online_optimizer != "immediate_sgd":
            bad.append("pipeline-v3 requires online_optimizer=immediate_sgd")
        if cfg.train.online_write_dispatch not in (
                "after_backward", "grad_ready"):
            bad.append(
                "pipeline-v3 online_write_dispatch must be after_backward "
                "or grad_ready")
        if cfg.train.stale_gradient_window < 0:
            bad.append("pipeline-v3 stale_gradient_window must be >= 0")
        if cfg.train.stale_gradient_window != 1:
            if (cfg.train.trajectory_source != "teacher_hidden"
                    and not bk_training):
                bad.append(
                    "stale_gradient_window != 1 initially requires "
                    "trajectory_source=teacher_hidden unless causal_bk "
                    "executes the student B×K block trajectory")
            if cfg.train.history_policy not in (
                    "causal_frozen_history", "causal_bk_probe", "causal_bk"):
                bad.append(
                    "stale-gradient windows require causal_frozen_history "
                    "or causal_bk_probe")
            if cfg.train.backward_dispatch != "per_block":
                bad.append(
                    "stale-gradient windows use backward_dispatch=per_block")
            if cfg.train.online_write_dispatch != "after_backward":
                bad.append(
                    "stale-gradient windows require after_backward fused writes")
            if (not cfg.train.lora.enabled
                    and cfg.train.history_policy != "causal_bk"):
                bad.append("stale-gradient windows are initially LoRA-only")
            if cfg.train.hidden_loss not in (
                "nmse", "l2mse", "cosine", "huber", "charbonnier",
                    "clipped_nmse", "vocab_cosine_sampled", "lens_kl",
                    "lens_js"):
                bad.append(
                    "stale-gradient windows initially support stateless "
                    "geometric hidden losses only")
        if cfg.train.backward_dispatch not in (
                "per_block", "per_token_disconnected",
                "answer_wavefront_disconnected", "answer_pipeline_lanes",
                "teacher_layer_lanes"):
            bad.append(
                "pipeline-v3 backward_dispatch must be per_block, "
                "per_token_disconnected, answer_wavefront_disconnected, "
                "answer_pipeline_lanes, or teacher_layer_lanes")
        if (cfg.train.backward_dispatch in (
                "per_token_disconnected", "answer_wavefront_disconnected",
                "answer_pipeline_lanes")
                and not cfg.train.lora.enabled):
            bad.append(
                "disconnected backward dispatch is initially LoRA-only; "
                "full-weight training uses per_block to avoid retaining "
                "every block gradient")
        if cfg.train.backward_dispatch == "answer_wavefront_disconnected":
            if cfg.train.history_policy != "causal_frozen_history":
                bad.append(
                    "answer_wavefront_disconnected requires "
                    "causal_frozen_history")
            if cfg.train.trajectory_source != "student_hidden":
                bad.append(
                    "answer_wavefront_disconnected currently implements the "
                    "student-hidden dependency grid; teacher-hidden already "
                    "parallelizes by layer")
        if cfg.train.backward_dispatch == "answer_pipeline_lanes":
            if cfg.train.history_policy != "causal_frozen_history":
                bad.append("answer_pipeline_lanes requires causal_frozen_history")
            if cfg.train.trajectory_source != "student_hidden":
                bad.append("answer_pipeline_lanes requires student_hidden")
            if cfg.train.hidden_loss not in (
                    "nmse", "l2mse", "cosine", "huber", "charbonnier",
                    "clipped_nmse", "vocab_cosine_sampled"):
                bad.append(
                    "answer_pipeline_lanes initially supports stateless "
                    "geometric hidden losses only")
        if cfg.train.backward_dispatch == "teacher_layer_lanes":
            if cfg.train.history_policy != "causal_frozen_history":
                bad.append("teacher_layer_lanes requires causal_frozen_history")
            if cfg.train.trajectory_source != "teacher_hidden":
                bad.append("teacher_layer_lanes requires teacher_hidden")
            if cfg.train.hidden_loss not in (
                    "nmse", "l2mse", "cosine", "huber", "charbonnier",
                    "clipped_nmse", "vocab_cosine_sampled"):
                bad.append(
                    "teacher_layer_lanes initially supports stateless "
                    "geometric hidden losses only")
        if cfg.train.lr_rule == "fixed":
            if cfg.train.lr_epoch_multipliers:
                bad.append(
                    "lr_epoch_multipliers must be empty when lr_rule=fixed")
        elif cfg.train.lr_rule == "epoch_piecewise":
            if not bk_training:
                bad.append(
                    "lr_rule=epoch_piecewise is implemented only by "
                    "pipeline-v3.1 causal_bk training")
            multipliers = cfg.train.lr_epoch_multipliers
            if len(multipliers) != cfg.train.epochs:
                bad.append(
                    "lr_epoch_multipliers must contain exactly one value "
                    "per training epoch")
            elif any(not math.isfinite(value) or value < 0
                     for value in multipliers):
                bad.append(
                    "lr_epoch_multipliers must be finite and non-negative")
        else:
            bad.append(
                "pipeline-v3 lr_rule must be fixed or epoch_piecewise")
        if cfg.train.history_policy not in (
            "recompute_prefix", "causal_frozen_history",
            "causal_static_eager_probe", "causal_static_graph_probe",
            "causal_bk_probe", "causal_bk",
        ):
            bad.append(
                "pipeline-v3 history_policy must be recompute_prefix, "
                "causal_frozen_history, causal_static_eager_probe, "
                "causal_static_graph_probe, causal_bk_probe, or causal_bk")
        if cfg.train.history_policy in (
                "causal_static_eager_probe", "causal_static_graph_probe"):
            if cfg.train.trajectory_source != "teacher_hidden":
                bad.append("causal_static probes require teacher_hidden")
            if cfg.train.stale_gradient_window != 1:
                bad.append("causal_static probes are exact K=1 probes")
            if cfg.train.backward_dispatch != "per_block":
                bad.append("causal_static probes require per_block")
            if cfg.train.online_write_dispatch != "after_backward":
                bad.append("causal_static probes require after_backward")
            if not cfg.train.lora.enabled:
                bad.append("causal_static probes are initially LoRA-only")
        if cfg.train.history_policy == "causal_bk_probe":
            if cfg.train.trajectory_source != "teacher_hidden":
                bad.append("causal_bk_probe requires teacher_hidden")
            if cfg.train.stale_gradient_window <= 0:
                bad.append("causal_bk_probe requires finite K > 0")
            if cfg.train.backward_dispatch != "per_block":
                bad.append("causal_bk_probe requires per_block")
            if cfg.train.online_write_dispatch != "after_backward":
                bad.append("causal_bk_probe requires after_backward")
            if not cfg.train.lora.enabled:
                bad.append("causal_bk_probe is initially LoRA-only")
        if cfg.train.history_policy == "causal_bk":
            if cfg.train.stale_gradient_window <= 0:
                bad.append("causal_bk requires finite K > 0")
            if cfg.train.backward_dispatch != "per_block":
                bad.append("causal_bk requires per_block")
            if cfg.train.online_write_dispatch != "after_backward":
                bad.append("causal_bk requires after_backward")
            if cfg.train.max_steps:
                bad.append("causal_bk currently runs complete epochs; max_steps must be 0")
        if cfg.train.conn_window not in (0, 1):
            bad.append("pipeline-v3 is strictly block-local (conn_window 0/1)")
        if cfg.train.conn_stride != 0:
            bad.append("pipeline-v3 has no connected-window stride")
        if cfg.train.offload_adam:
            bad.append("pipeline-v3 has no Adam state to offload")
        if cfg.train.anchor_hidden_weight:
            bad.append("pipeline-v3 anchor updates are not implemented")
        if cfg.train.scramble_targets:
            bad.append("pipeline-v3 target scrambling is not implemented")
        if cfg.train.window_dedup:
            bad.append("pipeline-v3 has no connected windows to deduplicate")
        if cfg.train.moe_mode != "dense_or_black_box":
            bad.append("pipeline-v3 MoE routing interventions are not implemented")
        if cfg.train.hidden_loss.startswith("delta_") or cfg.train.hidden_loss == "multi_delta_nmse":
            bad.append("pipeline-v3 first pass supports absolute local hidden losses only")
        if cfg.mask.compaction not in ("flow_mask", "pad_random", "intact"):
            bad.append("pipeline-v3 censorship is flow_mask, pad_random, or intact; removal modes are retired")
        if cfg.train.trajectory_source == "teacher_hidden":
            if cfg.train.teacher_hidden_source in ("cpu_cache", "gpu_cache"):
                if not cfg.cache.store_full_teacher_inputs:
                    bad.append("cached teacher_hidden requires cache.store_full_teacher_inputs=true")
                if cfg.train.online_teacher or cfg.train.frozen_teacher_copy:
                    bad.append("cached teacher_hidden must not load an online/frozen teacher")
                if cfg.train.pp_execution != "independent":
                    bad.append("cached teacher_hidden requires pp_execution=independent")
            elif cfg.train.teacher_hidden_source == "online":
                if not (cfg.train.online_teacher or cfg.train.frozen_teacher_copy):
                    bad.append("teacher_hidden online needs online_teacher (LoRA) or frozen_teacher_copy (full weights)")
                if cfg.train.online_teacher and not cfg.train.lora.enabled:
                    bad.append("teacher_hidden online_teacher requires LoRA so adapters-off is the frozen teacher")
                if cfg.train.frozen_teacher_copy and cfg.train.lora.enabled:
                    bad.append("teacher_hidden LoRA must use adapters-off online_teacher, not a redundant frozen copy")
                if cfg.train.online_teacher and cfg.train.frozen_teacher_copy:
                    bad.append("teacher_hidden must select exactly one full-prefix teacher source")
                if cfg.train.pp_execution == "independent":
                    bad.append("independent execution requires the cached source; online teacher still has stage dependencies")
            else:
                bad.append(
                    "teacher_hidden_source must be online, cpu_cache, or gpu_cache")
            if cfg.mask.compaction == "pad_random":
                bad.append("teacher_hidden + pad_random would feed uncensored teacher states at random-fill rows; use flow_mask or intact")
        elif cfg.train.trajectory_source != "student_hidden":
            bad.append(f"unknown pipeline-v3 trajectory_source={cfg.train.trajectory_source!r}")
        elif cfg.train.online_teacher or cfg.train.frozen_teacher_copy:
            bad.append("student_hidden consumes the disk cache directly; disable the unused online/frozen teacher")
        elif cfg.train.pp_execution == "independent":
            bad.append("independent execution is only valid for cached teacher_hidden")
    elif cfg.train.pipeline_version == 2:
        if cfg.train.pipeline_revision:
            bad.append("pipeline_revision is only valid for pipeline-v3")
        if cfg.train.update_granularity == "legacy_answer_sum":
            bad.append("pipeline_version=2 requires update_granularity=answer, token, or grid")
        if sched != "summed":
            bad.append("pipeline-v2 aggregation is implemented for summed schedule only")
        if cfg.train.update_granularity == "answer":
            if cfg.train.micro_batch != 1 or cfg.train.grad_accum != 1:
                bad.append("answer aggregation requires micro_batch=1 and grad_accum=1")
        elif cfg.train.update_granularity == "token":
            if cfg.train.batching not in ("padded", "bucketed"):
                bad.append("token aggregation requires padded or bucketed batching")
            if cfg.train.grad_accum != cfg.train.micro_batch:
                bad.append("token aggregation requires one batch per update (grad_accum=micro_batch)")
        elif cfg.train.update_granularity == "grid":
            if cfg.train.answers_per_update <= 0:
                bad.append("grid aggregation requires answers_per_update > 0")
            if cfg.train.tokens_per_answer_update < 0:
                bad.append("grid aggregation requires tokens_per_answer_update >= 0 (0 means all)")
            if (run_class == "method"
                    and cfg.train.tokens_per_answer_update == 0):
                bad.append("grid method arms require finite tokens_per_answer_update > 0; "
                           "unbounded K is reserved for explicitly typed controls")
            if cfg.train.update_reduction not in ("answer_mean", "token_mean"):
                bad.append("grid aggregation requires update_reduction=answer_mean or token_mean")
            if cfg.train.micro_batch != cfg.train.answers_per_update:
                bad.append("grid aggregation requires micro_batch=answers_per_update")
            if cfg.train.grad_accum != 1:
                bad.append("grid aggregation is exactly one tile per update and requires grad_accum=1")
            if cfg.train.batching not in ("padded", "bucketed"):
                bad.append("grid aggregation requires padded or bucketed batching")
            if cfg.train.online_teacher:
                bad.append("grid aggregation with an online teacher is not implemented")
    elif (cfg.train.pipeline_version != 4
          and cfg.train.update_granularity != "legacy_answer_sum"):
        # pipeline-v4's online granularity is validated in its own branch.
        bad.append("non-legacy update granularity requires its matching pipeline version")
    if (cfg.train.pipeline_version != 3
            and (cfg.train.backward_dispatch != "per_block"
                 or cfg.train.online_write_dispatch != "after_backward"
                 or cfg.train.stale_gradient_window != 1)):
        bad.append(
            "backward/write/window dispatch knobs are implemented only by "
            "pipeline-v3")
    if cfg.train.update_granularity != "grid":
        if cfg.train.answers_per_update:
            bad.append("answers_per_update is set but update_granularity is not grid")
        if cfg.train.tokens_per_answer_update:
            bad.append("tokens_per_answer_update is set but update_granularity is not grid")
        if cfg.train.update_reduction:
            bad.append("update_reduction is set but update_granularity is not grid")
    for knob, value, implemented in (
        ("attention_source", cfg.train.attention_source, "student_attention"),
        ("expert_routing_source", cfg.train.expert_routing_source, "black_box"),
    ):
        if value != implemented:
            bad.append(f"{knob}={value!r} is reserved but not implemented")
    if (cfg.train.pipeline_version not in (3, 4)
            and cfg.train.trajectory_source != "student_hidden"):
        bad.append(
            f"trajectory_source={cfg.train.trajectory_source!r} is implemented only by pipeline-v3/v4")
    if run_class not in RUN_CLASSES:
        raise ValueError(f"unknown train.run_class {run_class!r}")
    if run_class == "teacher_reference":
        raise ValueError(
            "train.run_class='teacher_reference' is eval-only: epoch-zero "
            "recall belongs in evaluation artifacts, never in training")
    if cfg.train.batching not in ("item", "padded", "bucketed"):
        raise ValueError(f"unknown train.batching {cfg.train.batching!r}")
    if cfg.mask.compaction not in (
        "remove", "stub", "stub_gap", "remove_gap", "pad_random", "flow_mask", "intact",
    ):
        raise ValueError(f"unknown mask.compaction {cfg.mask.compaction!r}")
    if (cfg.mask.compaction == "flow_mask"
            and cfg.train.pipeline_version not in (3, 4)):
        bad.append(
            "flow_mask is wired only into pipeline-v3/v4 block execution")
    if cfg.cache.source_compaction and cfg.cache.source_compaction not in (
        "remove", "stub", "stub_gap", "remove_gap", "pad_random",
    ):
        raise ValueError(
            f"unknown cache.source_compaction {cfg.cache.source_compaction!r}")
    if cfg.eval.every_epochs <= 0:
        raise ValueError("eval.every_epochs must be positive")
    if cfg.eval.standard_damage_every_epochs < 0:
        raise ValueError("eval.standard_damage_every_epochs must be >= 0")
    if cfg.train.hidden_loss == "vocab_cosine_sampled":
        if cfg.train.vocab_cosine_samples <= 1:
            bad.append(
                "vocab_cosine_sampled requires vocab_cosine_samples > 1")
    elif cfg.train.vocab_cosine_samples:
        bad.append(
            "vocab_cosine_samples is set but hidden_loss is not "
            "vocab_cosine_sampled")
    jacobian_kind = cfg.train.hidden_loss in (
        "jacobian_nmse", "jacobian_vocab_mse", "jacobian_cosine", "jacobian_lens_kl",
    )
    if jacobian_kind:
        if not cfg.train.jacobian_lens_path:
            raise ValueError(
                f"hidden_loss={cfg.train.hidden_loss!r} requires "
                "train.jacobian_lens_path")
        if not Path(cfg.train.jacobian_lens_path).is_file():
            raise ValueError(
                f"Jacobian lens artifact does not exist: "
                f"{cfg.train.jacobian_lens_path}")
    elif cfg.train.jacobian_lens_path:
        raise ValueError(
            "train.jacobian_lens_path is set but hidden_loss is not a "
            "jacobian_* objective")
    if cfg.eval.standard_damage_every_epochs:
        if cfg.eval.standard_damage_limit <= 0:
            raise ValueError("eval.standard_damage_limit must be positive")
        if cfg.eval.standard_damage_batch_size <= 0:
            raise ValueError("eval.standard_damage_batch_size must be positive")
        if cfg.train.schedule == "sequential":
            raise ValueError(
                "eval.standard_damage_every_epochs is epoch-based and is not "
                "implemented for the sequential per-layer schedule")
    unknown_corpora = set(cfg.eval.recall_corpora) - set(RECALL_CORPUS_PATHS)
    if unknown_corpora:
        raise ValueError(
            f"unknown eval.recall_corpora {sorted(unknown_corpora)}; "
            f"choose from {sorted(RECALL_CORPUS_PATHS)}")
    if cfg.train.moe_mode not in (
        "dense_or_black_box", "teacher_forced", "router_aligned",
    ):
        raise ValueError(f"unknown train.moe_mode {cfg.train.moe_mode!r}")
    if cfg.train.moe_mode != "dense_or_black_box":
        if sched != "summed":
            bad.append("moe_mode (teacher_forced/router_aligned are "
                       "implemented for the summed schedule only)")
        if (not cfg.train.online_teacher
                and cfg.train.pipeline_version != 4):
            # pipeline-v4 forbids online_teacher and instead records
            # routing during its own adapters-off online forward; its
            # branch above enforces v4_teacher_source=online.
            bad.append("moe_mode needs train.online_teacher (routing targets "
                       "are per-step, recorded adapters-off on the same "
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
        if (cfg.train.update_granularity != "grid"
                and cfg.train.history_policy not in (
                    "causal_bk_probe", "causal_bk")
                and cfg.train.grad_accum % cfg.train.micro_batch != 0):
            bad.append("grad_accum must be a multiple of micro_batch for batched training")
    is_method = run_class == "method"
    if cfg.train.conn_window > 1 and sched not in ("summed", "mixed"):
        bad.append("conn_window")
    if cfg.train.hidden_loss == "multi_delta_nmse":
        if cfg.train.conn_window <= 1 or cfg.train.conn_stride != 1:
            bad.append("multi_delta_nmse (needs a faithful connected window)")
        if max(cfg.train.multi_delta_scales, default=0) >= cfg.train.conn_window:
            bad.append("multi_delta_scales (each offset must be < conn_window)")
    if cfg.train.hidden_loss == "component_nmse":
        if sched != "summed" or cfg.train.conn_window != 1:
            bad.append("component_nmse (currently certified only for summed slide-1 local blocks)")
        if not (cfg.train.online_teacher or cfg.train.frozen_teacher_copy):
            bad.append("component_nmse (needs online frozen-teacher component targets)")
    if cfg.train.hidden_loss == "mahalanobis":
        if not cfg.train.mahalanobis_path or not Path(cfg.train.mahalanobis_path).is_file():
            bad.append("mahalanobis_path (needs a frozen precision artifact)")
    if cfg.train.conn_stride not in (0, 1):
        bad.append("conn_stride (only 0 = disjoint and 1 = sliding exist — "
                   "docs/windows.md; any other value would silently fall "
                   "into the disjoint branch and train different credit "
                   "assignment than intended)")
    if cfg.train.anchor_hidden_weight > 0 and sched != "summed":
        bad.append("anchor weights (the anchor step is wired into the "
                   "summed schedule only; other schedules would silently "
                   "train without the anchor)")
    if cfg.train.scramble_targets and sched != "summed":
        bad.append("scramble_targets")
    if sched == "teacher_censored" and uses_pipeline_map(cfg):
        bad.append("pipeline placement (teacher_censored walks stationary "
                   "teacher-stream inputs and crashes cross-device at item 1; "
                   "the combo is unimplemented — fail here, not mid-run)")
    if cfg.train.window_dedup and (cfg.train.conn_window <= 1
                                   or cfg.train.conn_stride != 1):
        bad.append("window_dedup (needs faithful sliding windows: "
                   "conn_window > 1, conn_stride == 1)")
    if (cfg.train.window_dedup
            and cfg.train.moe_mode == "router_aligned"):
        bad.append("window_dedup with router_aligned (router recording keeps "
                   "per-window graphs; known graph-leak path)")
    if cfg.train.offload_adam and sched != "summed":
        bad.append("offload_adam")
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


# Historical name, still used by scripts that predate the split.
_validate_knob_schedule = validate_knob_schedule
