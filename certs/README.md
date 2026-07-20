# Pipeline-v4 numerical checks

This checkout has no stored numerical reference set. Pipeline-v4 comparisons
are minted from the current code, used for one change, and then discarded.

Run the same tiny teacher-hidden experiment once as a single process and once
with independent layer shards, then compare every `(epoch, layer)` loss cell:

```bash
python scripts/compare_v4_shard_numerics.py \
    runs/<single-process-run> runs/<staged-run>
```

The comparison is exact by default; pass `--rtol` only when the experiment
deliberately changes numerical placement. `scripts/v4_battery.py` is the
stage-coordinated token-prediction validation subprocess, not a training
objective or a stored certification fixture.
