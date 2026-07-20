# Compression-analysis methodology

`audit_compression.py` compares the complete recursive `.py`, `.sh`, and
`.sbatch` population beneath `defactorised/` with the same relative paths
beneath `compressed/`. An optional two-column mapping CSV supports deliberate
renames or merged launchers without changing the audit code.

The audit separates five questions that are often accidentally conflated:

1. **Literal source size:** UTF-8 bytes, lines, and a reproducible GPT-like
   planning estimate (`ceil(bytes / 4)`) per file.
2. **Total collection size:** allocated filesystem blocks counted once per
   inode, so hardlinks and symlinks are not mistaken for duplicated storage.
3. **Startup:** median isolated `--help` process latency, only when explicitly
   requested and both scripts pass a conservative stdlib-only safety gate.
4. **Steady-state runtime:** not inferred from source compression and not
   benchmarked by this CPU-only audit.
5. **Numerical behavior:** never inferred from static syntax. The audit itself
   launches no model or GPU process; a separate on-demand H100 certification
   covers the self-contained `lora_online` variant and is reported explicitly.

Static equivalence evidence is strongest-first: identical bytes; identical
bytes after removing the marked private bundle/shared-bootstrap regions;
identity after additionally canonicalizing the necessary collection-root
reference (`defactorised/` versus `compressed/`); equal Python AST; or equal
comment/format-insensitive lexical tokens. Anything else is explicitly
`unverified` and requires behavioral review.

Collection totals include every regular file except transient `__pycache__`
and `.pyc` files, so the compressed side pays for its shared ZIP/helper rather
than making that cost disappear from one-to-one script rows. The report also
shows script-source totals separately from this all-file footprint.

Statuses have deliberately narrow meanings:

- `optimized`: smaller literal source and verified static equivalence.
- `preserved`: verified static equivalence without smaller literal source.
- `not-applicable`: missing counterpart or no static equivalence proof.

Run the static audit from any directory:

```bash
python3 compression_analysis/audit_compression.py
```

After reviewing safety classifications, optionally collect guarded startup
measurements:

```bash
python3 compression_analysis/audit_compression.py --benchmark-startup
```

Outputs are `compression_audit.csv` (complete machine-readable evidence) and
`REPORT.md` (collection summary plus per-script table).
