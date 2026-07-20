"""Pipeline-v4 entry point.

Training has one objective in this repository: block ``L`` consumes the
detached teacher state ``h[L-1]``; its trainable student weights produce a
differentiable local output matched to teacher ``h[L]``.  End-to-end student
trajectory states are produced only by the no-grad validation relay in
``online_v4``; they never become training inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ExperimentConfig
from ..utils.runlog import setup_run_dir
from ..utils.seeding import seed_everything
from .moe import dequantize_overrides
from .runtime import TrainingRuntime
from .stop import cooperative_stop_signals
from .validate import validate_knob_schedule


def _require_node_cache(cfg: ExperimentConfig) -> None:
    """Fail before model loading when the host-local cache is unavailable."""
    if cfg.cache.runtime_policy != "node_epoch0":
        return
    from ..teacher.cache import resolve_cache_dir
    from ..teacher.node_epoch0 import ready_manifest, runtime_identity

    cache_root, cache_hash = resolve_cache_dir(cfg)
    if ready_manifest(
            cache_root, cache_hash, compatibility=runtime_identity()) is None:
        raise RuntimeError(
            "node-local epoch-zero teacher cache is not ready at "
            f"{cache_root}; run scripts/build_teacher_cache.py with "
            "--coordinated-node-cache under this node's GPU runtime "
            "(pre-load gate — refused before weight materialization)")


def train_layerwise(cfg: ExperimentConfig) -> Path:
    """Train the v4 teacher-hidden-input objective and publish its checkpoint."""
    validate_knob_schedule(cfg)
    run_dir, log = setup_run_dir(cfg)
    seed_everything(cfg.train.seed)
    _require_node_cache(cfg)

    load_kw = dequantize_overrides(cfg.model.name, cfg.train.moe_mode)
    runtime = TrainingRuntime(cfg).load(load_kw)
    tokenizer, stack = runtime.tokenizer, runtime.stack
    cache = runtime.load_cache()
    log.log(
        kind="teacher_cache_source",
        runtime_policy=cfg.cache.runtime_policy,
        cache_root=str(cache.root),
        cache_hash=cache._index["config_hash"],
        node_epoch0_manifest=runtime.cache_manifest,
    )

    from .online_v4 import _owned_range, certify_locality_v4, train_online_v4

    with cooperative_stop_signals():
        stopped = train_online_v4(
            cfg, stack, tokenizer, log, cache,
            peft_model=runtime.peft_model, run_dir=run_dir)
        locality = certify_locality_v4(
            cfg, stack, tokenizer, cache, run_dir,
            peft_model=runtime.peft_model)
        cert_keys = (
            "items", "gradient_contract", "final_logit_training",
            "local_grad_norm", "cross_block_leak_grad_norm",
            "frozen_vocab_grad_norm", "local_signal_present_in_every_block",
            "passed", "skipped", "owner_note",
        )
        log.log(
            kind="locality_certification",
            **{key: locality[key] for key in cert_keys if key in locality},
        )
        if not locality["passed"] and not locality.get("skipped"):
            raise RuntimeError(
                "pipeline-v4 locality certification failed; checkpoint "
                f"withheld: {locality}")

        runtime.save_checkpoint(run_dir)
        owned = _owned_range(cfg, stack.n_layers)
        (run_dir / "checkpoint" / "v4_stage_manifest.json").write_text(
            json.dumps({
                "v4_stage": cfg.train.v4_stage,
                "v4_stage_splits": list(cfg.train.v4_stage_splits or []),
                "owned_blocks": [owned.start, owned.stop - 1],
                "training_input": "teacher_h[L-1]",
                "training_target": "teacher_h[L]",
            }, indent=2) + "\n")
        log.log(kind="done", graceful_stop=bool(stopped),
                **runtime.memory_summary())
    log.close()
    return run_dir
