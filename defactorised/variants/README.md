# Purpose-specific launchers

These small entry points select a semantic workflow before forwarding the
remaining scalar arguments to the corresponding standalone program in
`defactorised/`. They never import `selfupdate` and never add `src/` to the
module search path. A mode cannot be overridden at the command line.

| Original overloaded program | Purpose-specific entry points | Fixed mode |
|---|---|---|
| `parallel_bench.py` | `parallel_bench_single.py`, `parallel_bench_pipeline_2gpu.py`, `parallel_bench_tensor_2gpu.py` | `single`, `pp2`, `tp2` |
| `build_teacher_cache.py` | `build_teacher_cache_full.py`, `build_teacher_cache_answers.py`, `build_teacher_cache_index.py` | full cache, answers only, imported-response index only |
| `evaluate.py` | `evaluate_base.py`, `evaluate_checkpoint.py`, `evaluate_layer_residuals.py` | epoch-zero/base, checkpoint recall, checkpoint residual diagnostic |
| `standard_destruction_eval.py` | `standard_destruction_base.py`, `standard_destruction_checkpoint.py` | base or checkpoint |
| `destruct_eval.py` | `destruct_base_fast.py`, `destruct_base_full.py`, `destruct_checkpoint_fast.py`, `destruct_checkpoint_full.py` | model source and battery size |
| `logit_lens.py` | `logit_lens_raw.py`, `logit_lens_tuned.py` | raw or tuned lens |
| `train_certify.py` | `train_certify_list.py`, `train_certify_all.py`, `train_certify_selected.py` | list, all, or explicit `--variant` selection |
| `smoke_family.py` | `smoke_family_default.py`, `smoke_family_selected.py`, `smoke_family_all.py` | default model, explicit `--model`, or all families |
| `teacher_ceiling.py` | `teacher_ceiling_no_context.py`, `teacher_ceiling_full_context.py`, `teacher_ceiling_window_context.py`, `teacher_ceiling_chapter_context.py` | context scope |

For example:

```bash
python defactorised/variants/parallel_bench_pipeline_2gpu.py --help
python defactorised/variants/evaluate_checkpoint.py \
  --experiment configs/experiments/example.yaml --checkpoint runs/example/checkpoint
```

Only modes that change the meaning of a command are split. Scalar resource,
batching, path, and sampling knobs remain arguments; duplicating scripts for
every numeric combination would hide rather than clarify the workflow.
