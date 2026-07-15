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
from ..teacher.cache import resolved_node_epoch0_root
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
    if cfg.train.pipeline_version not in (1, 2, 3):
        raise ValueError("train.pipeline_version must be 1, 2, or 3")
    if cfg.cache.runtime_policy not in ("durable", "node_epoch0"):
        raise ValueError(
            "cache.runtime_policy must be durable or node_epoch0")
    if cfg.cache.generation_max_tokens < 0:
        raise ValueError("cache.generation_max_tokens must be non-negative")
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
    if cfg.train.pipeline_version == 3:
        if cfg.train.pipeline_revision not in ("", "3.0", "3.1"):
            bad.append("pipeline-v3 revision must be 3.0 or 3.1")
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
            if cfg.train.pipeline_revision != "3.1":
                bad.append("causal_bk execution requires pipeline_revision=3.1")
            if cfg.train.micro_batch < 2:
                bad.append("causal_bk execution requires micro_batch B >= 2")
            if cfg.train.batching not in ("padded", "bucketed"):
                bad.append("causal_bk execution requires padded or bucketed batching")
            shard_users = cfg.train.activation_shard_users
            if shard_users < 0 or shard_users > cfg.train.micro_batch:
                bad.append(
                    "activation_shard_users must be in [0, micro_batch] "
                    "for causal_bk")
        else:
            if cfg.train.activation_shard_users:
                bad.append(
                    "activation_shard_users is implemented only by causal_bk")
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
            if not cfg.train.lora.enabled:
                bad.append("stale-gradient windows are initially LoRA-only")
            if cfg.train.hidden_loss not in (
                    "nmse", "l2mse", "cosine", "huber", "charbonnier",
                    "clipped_nmse"):
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
                    "clipped_nmse"):
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
                    "clipped_nmse"):
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
            if not cfg.train.lora.enabled:
                bad.append("causal_bk is initially LoRA-only")
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
            if not (cfg.train.online_teacher or cfg.train.frozen_teacher_copy):
                bad.append("teacher_hidden needs online_teacher (LoRA) or frozen_teacher_copy (full weights); aligned disk caches lack h[L-1] prefixes")
            if cfg.train.online_teacher and not cfg.train.lora.enabled:
                bad.append("teacher_hidden online_teacher requires LoRA so adapters-off is the frozen teacher")
            if cfg.train.frozen_teacher_copy and cfg.train.lora.enabled:
                bad.append("teacher_hidden LoRA must use adapters-off online_teacher, not a redundant frozen copy")
            if cfg.train.online_teacher and cfg.train.frozen_teacher_copy:
                bad.append("teacher_hidden must select exactly one full-prefix teacher source")
            if cfg.mask.compaction == "pad_random":
                bad.append("teacher_hidden + pad_random would feed uncensored teacher states at random-fill rows; use flow_mask or intact")
        elif cfg.train.trajectory_source != "student_hidden":
            bad.append(f"unknown pipeline-v3 trajectory_source={cfg.train.trajectory_source!r}")
        elif cfg.train.online_teacher or cfg.train.frozen_teacher_copy:
            bad.append("student_hidden consumes the disk cache directly; disable the unused online/frozen teacher")
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
    elif cfg.train.update_granularity != "legacy_answer_sum":
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
    if (cfg.train.pipeline_version != 3
            and cfg.train.trajectory_source != "student_hidden"):
        bad.append(
            f"trajectory_source={cfg.train.trajectory_source!r} is implemented only by pipeline-v3")
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
    if cfg.mask.compaction == "flow_mask" and cfg.train.pipeline_version != 3:
        bad.append("flow_mask is wired only into pipeline-v3 block execution")
    if cfg.cache.source_compaction and cfg.cache.source_compaction not in (
        "remove", "stub", "stub_gap", "remove_gap", "pad_random",
    ):
        raise ValueError(
            f"unknown cache.source_compaction {cfg.cache.source_compaction!r}")
    if cfg.eval.every_epochs <= 0:
        raise ValueError("eval.every_epochs must be positive")
    if cfg.eval.standard_damage_every_epochs < 0:
        raise ValueError("eval.standard_damage_every_epochs must be >= 0")
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
        bad.append("window_dedup with router_aligned (router capture keeps "
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
