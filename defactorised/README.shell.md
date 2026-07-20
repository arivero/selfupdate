# Shell and Slurm mirrors

This directory mirrors every tracked `scripts/*.sh` and `scripts/*.sbatch`
entry point as of this change. Their executable modes are preserved. Calls to
Python and shell entry points are routed through `defactorised/`, and the two
inline Python configuration/cache probes use the standalone
`shell_helpers.py`, so the launchers do not route back through
`src/selfupdate`.

The shell include in `gpu_scheduler.sh` is intentionally retained: it loads
the mirrored `gpu_lease.sh` beside it and is unrelated to the Python
`src/selfupdate` dependency targeted by this directory.

Tracked TSV queue inputs are not duplicated. Launchers resolve their repository
root as the parent of this directory and deliberately continue to read the
canonical assets under `scripts/` (for example `scripts/queue.tsv` and
`scripts/watchdog_backlog.tsv`). Duplicating those mutable operational queues
would create two competing sources of truth.

`MANIFEST.shell.txt` records the original-to-mirror mapping.
