# Defactorised Python entry points

Every tracked `scripts/*.py` has a same-named standalone copy in this
directory. Files that import `selfupdate` embed a compressed snapshot of the
53-module package and install an in-memory importer before their original
imports run. They neither add nor read the repository's `src` directory.
Files with no `selfupdate` imports stay plain source copies, apart from removal
of obsolete `sys.path` bootstraps.

These flat files are mechanical, behavior-preserving deliverables, not the
recommended material for a one-hour code walk-through. The curated
`demos/ppn_partition_readable.py` instead inlines only the production PPn cost
profile and contiguous partition dynamic program. It exposes the important
architecture without the full package payload.

Regenerate and check determinism with the repository's modern Python runtime:

```bash
/tmp/$USER/selfupdate-venv/bin/python defactorised/generate.py
/tmp/$USER/selfupdate-venv/bin/python defactorised/generate.py --check
```

The generator also redirects Python child entry points (`train`, certification
cache construction, and the pipeline-v4 battery) to their `defactorised/`
counterparts. Embedded module `__file__` values retain the original repo-root
depth convention so provenance logging still resolves the checkout root.

`shell_helpers.py` replaces shell heredocs that previously inserted `src` onto
`sys.path`, including the venv's late eval-import guard:

```bash
python defactorised/shell_helpers.py v4-launch-info BASE.yaml EXP.yaml
python defactorised/shell_helpers.py node-cache-index BASE.yaml EXP.yaml
python defactorised/shell_helpers.py venv-import-check
```
