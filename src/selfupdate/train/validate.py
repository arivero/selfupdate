"""Dispatch-time validation for the v4-only trainer.

The objective is structural, not an optional schedule: teacher ``h[L-1]`` is
the input to block L and teacher ``h[L]`` is its target.  Old pipeline and
student-trajectory knobs are rejected before model loading.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..eval.tasks import RECALL_CORPUS_PATHS
from ..eval.teacher_output import EVALUATION_ONLY_OUTPUT_NAMES
from ..teacher.cache import resolved_node_epoch0_root

RUN_CLASSES = {
    "method", "teacher_reference", "ablation", "control",
    "legacy_archive", "confounded", "open",
}
NON_METHOD_CLASSES = RUN_CLASSES - {"method"}


def validate_knob_schedule(cfg) -> None:
    """Reject every configuration outside the teacher-forced v4 contract."""
    train = cfg.train
    bad: list[str] = []

    if train.pipeline_version != 4:
        raise ValueError(
            "this repository is v4-only: train.pipeline_version must be 4; "
            "student-trajectory training pipelines were removed")
    if train.pipeline_revision not in ("", "4.0"):
        bad.append("pipeline_revision must be 4.0")
    if cfg.model.pipeline_split or cfg.model.pipeline_splits:
        bad.append("model.pipeline_split(s) are obsolete; use v4_stage_splits")

    if train.hidden_loss in EVALUATION_ONLY_OUTPUT_NAMES:
        raise ValueError(
            f"train.hidden_loss={train.hidden_loss!r} is evaluation-only and "
            "can never enter backward")
    if ((train.hidden_loss.startswith("delta_")
         and train.hidden_loss != "delta_cosine")
            or train.hidden_loss.startswith("multi_delta_")):
        bad.append(
            "v4 admits only delta_cosine with a detached teacher anchor; "
            "all other delta/connected-trajectory losses are forbidden")

    if train.v4_teacher_source not in ("cache", "online", "store"):
        bad.append("v4_teacher_source must be cache, online, or store")
    if (train.v4_teacher_source == "cache"
            and not cfg.cache.store_full_teacher_inputs):
        bad.append("v4 cache source requires store_full_teacher_inputs=true")
    if train.v4_teacher_source == "online" and train.v4_loop_order != "item_major":
        bad.append("online teacher source requires item_major")
    if train.v4_teacher_source in ("online", "store"):
        if train.v4_kv_source != "teacher_frozen":
            bad.append("online/store teacher source requires teacher_frozen KV")
        if not train.lora.enabled:
            bad.append("online/store teacher source requires LoRA adapters-off")
    if (train.v4_teacher_source == "store"
            and train.v4_teacher_residency == "rebuild"):
        bad.append("fill-once store cannot use rebuild residency")

    if train.v4_kv_source not in ("teacher_frozen", "student_refresh"):
        bad.append("v4_kv_source must be teacher_frozen or student_refresh")
    if train.v4_kv_source == "student_refresh" and train.v4_kv_refresh_epochs <= 0:
        bad.append("student_refresh requires v4_kv_refresh_epochs > 0")
    if train.v4_kv_source == "teacher_frozen" and train.v4_kv_refresh_epochs:
        bad.append("teacher_frozen KV never refreshes")
    if "deepseek" in (cfg.model.name or "").lower():
        if train.v4_kv_source != "teacher_frozen":
            bad.append("DeepSeek compressed context must remain teacher-recorded")
        if train.expert_routing_source != "black_box":
            bad.append("DeepSeek v4 supports black_box routing only")

    if cfg.mask.compaction not in ("flow_mask", "intact"):
        bad.append("v4 requires flow_mask censorship or intact control")
    if cfg.cache.runtime_policy not in ("durable", "node_epoch0"):
        raise ValueError("cache.runtime_policy must be durable or node_epoch0")
    if cfg.cache.runtime_policy == "node_epoch0":
        if not cfg.cache.node_root.startswith("/dev/shm/"):
            raise ValueError("node_epoch0 cache root must be under /dev/shm")
        resolved_node_epoch0_root(cfg)
    if cfg.cache.generation_max_tokens < 0 or cfg.cache.item_cache_items < 0:
        raise ValueError("cache token/item limits must be non-negative")

    if train.v4_optimizer not in ("immediate_sgd", "adam"):
        bad.append("v4_optimizer must be immediate_sgd or adam")
    if train.v4_grad_clip < 0:
        bad.append("v4_grad_clip must be non-negative")
    betas = tuple(train.v4_adam_betas)
    if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
        bad.append("v4_adam_betas must contain two values in [0,1)")
    if train.v4_adam_eps <= 0 or train.v4_adam_weight_decay < 0:
        bad.append("v4 Adam epsilon must be positive and decay non-negative")

    if train.v4_loop_order not in ("layer_major", "item_major"):
        bad.append("v4_loop_order must be layer_major or item_major")
    if train.v4_loss_positions not in ("answer", "aligned", "thinking_answer"):
        bad.append("invalid v4_loss_positions")
    if train.v4_teacher_residency not in (
            "auto", "gpu_corpus", "cpu_stream", "rebuild"):
        bad.append("invalid v4_teacher_residency")
    if train.v4_weight_residency not in ("resident", "rotate", "auto"):
        bad.append("invalid v4_weight_residency")
    if train.v4_relay_every_cohorts < 0:
        bad.append("v4_relay_every_cohorts must be non-negative")
    if train.v4_weight_residency != "resident":
        if not train.v4_stage_scoped:
            bad.append("rotate/auto weight residency requires stage-scoped loading")
        if train.v4_loop_order != "layer_major":
            bad.append("weight rotation requires layer_major")

    if train.v4_battery_mode not in ("graft", "subprocess"):
        bad.append("v4_battery_mode must be graft or subprocess")
    if (train.v4_battery_mode == "subprocess" and train.v4_stage < 0
            and not train.v4_stage_scoped):
        bad.append("resident single-process evaluation must use graft mode")
    if train.v4_stage_scoped:
        if train.v4_battery_mode != "subprocess":
            bad.append("stage-scoped evaluation requires subprocess mode")
        if (train.v4_stage < 0
                and train.v4_weight_residency not in ("rotate", "auto")):
            bad.append("single-process stage-scoped mode is the rotary PPP1 lane")
        if not train.lora.enabled:
            bad.append("stage-scoped loading requires LoRA")
        if train.v4_teacher_source == "online":
            bad.append("stage-scoped loading cannot walk foreign blocks online")

    splits = list(train.v4_stage_splits or [])
    if any(left >= right for left, right in zip(splits, splits[1:])):
        bad.append("v4_stage_splits must be strictly increasing")
    if splits and splits[0] <= 0:
        bad.append("v4_stage_splits are positive one-based block cuts")
    stages = len(splits) + 1
    devices = list(train.v4_stage_devices or [])
    if devices and len(devices) != stages:
        bad.append("v4_stage_devices must name one device per stage")
    hosts = os.environ.get("SELFUPDATE_V4_STAGE_HOSTS", "").split()
    # Repeated physical ids are valid in a cross-node config.  Static config
    # audit has no host assignment, so uniqueness is enforced once the
    # launcher supplies SELFUPDATE_V4_STAGE_HOSTS.
    if hosts:
        if len(hosts) != len(devices):
            bad.append("v4 stage host/device lists must have equal length")
        elif len(set(zip(hosts, devices))) != len(devices):
            bad.append("v4 stage (host,device) assignments must be unique")
    if not -1 <= train.v4_stage < stages:
        bad.append(f"v4_stage outside -1..{stages - 1}")

    if train.run_class not in RUN_CLASSES:
        raise ValueError(f"unknown train.run_class {train.run_class!r}")
    if train.run_class == "teacher_reference":
        raise ValueError("teacher_reference is evaluation-only, never training")
    if train.batching not in ("item", "padded", "bucketed"):
        raise ValueError(f"unknown train.batching {train.batching!r}")
    if cfg.eval.every_epochs <= 0 or cfg.eval.standard_damage_every_epochs < 0:
        raise ValueError("evaluation cadence is invalid")
    if cfg.eval.standard_damage_every_epochs and (
            cfg.eval.standard_damage_limit <= 0
            or cfg.eval.standard_damage_batch_size <= 0):
        raise ValueError("standard-damage limits must be positive")
    unknown_corpora = set(cfg.eval.recall_corpora) - set(RECALL_CORPUS_PATHS)
    if unknown_corpora:
        raise ValueError(f"unknown recall corpora: {sorted(unknown_corpora)}")

    if train.hidden_loss == "vocab_cosine_sampled":
        if train.vocab_cosine_samples <= 1:
            bad.append("vocab_cosine_sampled requires more than one row")
    elif train.vocab_cosine_samples:
        bad.append("vocab_cosine_samples set for a different loss")
    jacobian = train.hidden_loss in (
        "jacobian_nmse", "jacobian_vocab_mse", "jacobian_cosine",
        "jacobian_lens_kl",
    )
    if jacobian and (not train.jacobian_lens_path
                     or not Path(train.jacobian_lens_path).is_file()):
        bad.append("jacobian loss requires an existing jacobian_lens_path")
    if not jacobian and train.jacobian_lens_path:
        bad.append("jacobian_lens_path set for a non-jacobian loss")
    if train.hidden_loss == "mahalanobis" and (
            not train.mahalanobis_path
            or not Path(train.mahalanobis_path).is_file()):
        bad.append("mahalanobis requires an existing precision artifact")

    if train.moe_mode not in (
            "dense_or_black_box", "teacher_forced", "router_aligned"):
        raise ValueError(f"unknown train.moe_mode {train.moe_mode!r}")
    if train.moe_mode != "dense_or_black_box":
        if train.v4_teacher_source != "online":
            bad.append("router interventions require online teacher recording")
        if train.moe_mode == "router_aligned" and train.moe_router_weight <= 0:
            bad.append("router_aligned requires a positive router weight")
    elif train.moe_router_weight:
        bad.append("router weight set without router_aligned mode")

    if bad:
        raise ValueError(
            f"knob(s) {bad} violate the v4 teacher-hidden contract")


_validate_knob_schedule = validate_knob_schedule
