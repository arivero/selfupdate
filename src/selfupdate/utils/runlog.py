"""Append-only JSONL run metrics and run-directory bootstrap."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml


class RunLog:
    def __init__(self, run_dir: str | Path, defaults: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.defaults = dict(defaults or {})

    def log(self, **kv) -> None:
        kv = {**self.defaults, **kv}
        kv.setdefault("t", round(time.time(), 3))
        self._f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def setup_run_dir(cfg) -> tuple[Path, "RunLog"]:
    """runs/<run_name>/ with config.yaml dumped; the single bootstrap both
    trainers share so run metadata stays consistent across methods. A rerun
    rotates any previous metrics.jsonl aside so analysis never mixes
    attempts under the latest config."""
    run_dir = Path("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    old = run_dir / "metrics.jsonl"
    if old.exists() and old.stat().st_size > 0:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        old.rename(run_dir / f"metrics.prev-{stamp}.jsonl")
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), allow_unicode=True)
    )
    defaults = {}
    if cfg.train.pipeline_version in (2, 3):
        repo_root = Path(__file__).resolve().parents[3]
        examples = Path(cfg.data.examples_path)
        runtime_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--",
             "src/selfupdate", "scripts/train.py"], cwd=repo_root)
        runtime_untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--",
             "src/selfupdate", "scripts/train.py"], cwd=repo_root,
            text=True).splitlines()
        defaults = {
            "run_name": cfg.run_name,
            "layerwise_project_version": getattr(
                cfg, "layerwise_project_version", "3.4"),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root,
                text=True).strip(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runtime_dirty": bool(runtime_diff or runtime_untracked),
            "runtime_diff_sha256": (
                hashlib.sha256(runtime_diff).hexdigest()
                if runtime_diff else None),
            "runtime_untracked": runtime_untracked,
            "dataset_path": cfg.data.examples_path,
            "dataset_sha256": (
                hashlib.sha256(examples.read_bytes()).hexdigest()
                if examples.is_file() else None),
            "model_base_identity": cfg.model.name,
            "student_init_identity": cfg.train.init_from or cfg.model.name,
            "pipeline_version": cfg.train.pipeline_version,
            "pipeline_revision": cfg.train.pipeline_revision,
            "pp_execution": cfg.train.pp_execution,
            "physical_gpu_mapping": list(
                getattr(cfg.model, "pipeline_devices", []) or []),
            "partition_profile_id": cfg.train.partition_profile_id,
            "partition_profile_path": cfg.train.partition_profile_path,
            "update_granularity": cfg.train.update_granularity,
            "answers_per_update": cfg.train.answers_per_update,
            "tokens_per_answer_update": cfg.train.tokens_per_answer_update,
            "update_reduction": cfg.train.update_reduction,
            "grid_layer_order": (
                "forward" if cfg.train.pipeline_version == 2 else None),
            "grid_causal_context": (
                "full_prefix" if cfg.train.pipeline_version == 2 else None),
            "grid_student_trajectory_edge": (
                "h[L-1] -> h[L]"
                if cfg.train.pipeline_version == 2 else None),
            "optimizer_updates_per_tile": (
                1 if cfg.train.pipeline_version == 2 else None),
            "atomic_update_event": (
                "one_answer_token_x_one_block"
                if cfg.train.pipeline_version == 3 else None),
            "final_logit_training": False,
            "trajectory_source": cfg.train.trajectory_source,
            "attention_source": cfg.train.attention_source,
            "expert_routing_source": cfg.train.expert_routing_source,
            "mask_mode": cfg.mask.mode,
            "censorship_compaction": cfg.mask.compaction,
            "cache_runtime_policy": cfg.cache.runtime_policy,
            "loss_kind": cfg.train.hidden_loss,
            "seed": cfg.train.seed,
            "batching": cfg.train.batching,
            "micro_batch": cfg.train.micro_batch,
            "grad_accum": cfg.train.grad_accum,
            "online_optimizer": cfg.train.online_optimizer,
            "backward_dispatch": cfg.train.backward_dispatch,
            "online_write_dispatch": cfg.train.online_write_dispatch,
            "lr_rule": cfg.train.lr_rule,
            "history_policy": cfg.train.history_policy,
            "conn_window": cfg.train.conn_window,
            "conn_stride": cfg.train.conn_stride,
            "pipeline_split": cfg.model.pipeline_split,
            "pipeline_splits": cfg.model.pipeline_splits,
            "pipeline_world_size": getattr(cfg.model, "pipeline_world_size", 0),
            "device_map": cfg.model.device_map,
        }
    return run_dir, RunLog(run_dir, defaults=defaults)


def read_metrics(run_dir: str | Path) -> list[dict]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
