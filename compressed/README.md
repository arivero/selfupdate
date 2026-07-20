# Compressed script collection

This directory mirrors every executable Python, shell, and Slurm script below
`defactorised/`.  It is the shorter, faster-to-load companion to that explicit
teaching collection; `defactorised/` remains the readable standalone baseline.

Python scripts store the embedded `selfupdate` package once in
`_selfupdate_bundle.zip`.  `_shared_bundle.py` checks the archive digest and
installs an in-memory importer, so compressed entry points never read the
repository's `src/selfupdate` tree at runtime.  Shell and Slurm scripts remain
ordinary text.  Only simple declarative wrappers had comments compacted;
scheduler, lease, cache, reaper, NCCL, and coordination logic stays visible.

Regenerate or verify exact coverage with a modern Python:

```bash
python compressed/_generate_compressed.py
python compressed/_generate_compressed.py --check
```

`MANIFEST.csv` maps every source Python file to its compressed counterpart and
records hashes and sizes; additional rows explicitly label collection-only
infrastructure. `_generate_compressed.py`, `_shared_bundle.py`, and the
integrity-stamped `_selfupdate_bundle.zip` are not additional source entry
points. `compressed/generate.py` remains the behavior-equivalent counterpart
of `defactorised/generate.py`. Python child-entry paths are redirected to
`compressed/`. The analysis tools deliberately retain `defactorised/` as their
default population so moving the tool does not silently change the scientific
comparison.

See [`MANIFEST.shell.md`](MANIFEST.shell.md) for shell coverage and preservation
decisions, [`MEASUREMENTS.md`](MEASUREMENTS.md) for the Python footprint and
startup measurement, and
[`../compression_analysis/REPORT.md`](../compression_analysis/REPORT.md) for
the reproducible all-script audit.  “Optimized” in that report means smaller
source with static equivalence evidence; it is not a blanket training-speed or
floating-point claim.
