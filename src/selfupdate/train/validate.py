"""Dispatch-time validation of the experiment's knob schedule.

Knob-flow law (2026-07-05): a knob that a schedule does not implement must
RAISE, never silently ignore — spec/code divergence is the bug class that
produced the unwired-ce_kind incident. Since the stored tests were retired
(owner decision 2026-07-11), this module and the prose laws in CLAUDE.md ARE
the specification; ``scripts/audit_configs.py`` sweeps every config through
it offline.
"""

from __future__ import annotations

from pathlib import Path

from ..eval.tasks import RECALL_CORPUS_PATHS
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
    if run_class not in RUN_CLASSES:
        raise ValueError(f"unknown train.run_class {run_class!r}")
    if run_class == "teacher_reference":
        raise ValueError(
            "train.run_class='teacher_reference' is eval-only: epoch-zero "
            "recall belongs in evaluation artifacts, never in training")
    if cfg.train.batching not in ("item", "padded", "bucketed"):
        raise ValueError(f"unknown train.batching {cfg.train.batching!r}")
    if cfg.mask.compaction not in (
        "remove", "stub", "stub_gap", "remove_gap", "pad_random",
    ):
        raise ValueError(f"unknown mask.compaction {cfg.mask.compaction!r}")
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
        if cfg.train.grad_accum % cfg.train.micro_batch != 0:
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
    if (cfg.train.anchor_kl_weight > 0 or cfg.train.anchor_hidden_weight > 0) and sched != "summed":
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
    if cfg.train.readout_window_blocks > 0 and cfg.train.readout_source == "UNSET":
        raise ValueError(
            "readout_source must be set EXPLICITLY to teacher_kl when "
            "readout_window_blocks > 0 — defaults are experiment variables")
    if cfg.train.readout_source not in ("UNSET", "teacher_kl"):
        bad.append("readout_source must be teacher_kl; reference-text training is forbidden")
    if cfg.train.readout_weight > 0 and cfg.train.readout_window_blocks <= 0:
        bad.append("readout_weight without readout_window_blocks > 0 (the "
                   "readout term would silently never run — the L == readout0 "
                   "branch is unreachable; an arm would land in results "
                   "classified as a readout arm having trained none)")
    if cfg.train.readout_window_blocks > 0:
        if sched == "teacher_censored":
            bad.append("readout_window_blocks (teacher_censored is pure by definition)")
        if sched == "sequential":
            bad.append("readout_window_blocks (the sequential schedule has "
                       "no readout path; the window would be silently ignored)")
        if cfg.train.conn_window <= 0:
            # Owner hard stop 2026-07-04, enforced for EVERY run class: "not
            # as a baseline, not as a repro reference, not under any
            # subterfuge". Tail experiments belong to ../selfupdate_kd.
            raise ValueError(
                "tail-only window (readout_window_blocks > 0 with conn_window "
                "0/absent) is banned for every run class; use the sanctioned "
                "sliding conn_window/conn_stride: 1 or route the arm to "
                "../selfupdate_kd")
        if cfg.train.hidden_loss == "zero":
            raise ValueError(
                "readout_window_blocks with hidden_loss='zero' is a tail-only "
                "arm in disguise (no body signal, readout-only gradient) — "
                "banned for every run class")
        if is_method:
            if cfg.train.conn_window <= 0 or cfg.train.conn_stride != 1:
                bad.append("readout_window_blocks without sanctioned sliding conn_window/conn_stride")
            if cfg.train.readout_window_blocks != cfg.train.conn_window:
                bad.append("readout_window_blocks must equal conn_window for method arms")
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
