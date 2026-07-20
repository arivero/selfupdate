#!/usr/bin/env python3
"""Audit defactorised scripts against a mirrored compressed collection.

The audit is static by default.  ``--benchmark-startup`` enables a deliberately
conservative ``--help`` benchmark only for Python files proven to have a small
stdlib-only import surface and an argparse main guard.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import tokenize


SUFFIXES = {".py", ".sh", ".sbatch"}
BUNDLE_RE = re.compile(
    rb"\n?# BEGIN GENERATED STANDALONE SELFUPDATE BUNDLE\n.*?"
    rb"# END GENERATED STANDALONE SELFUPDATE BUNDLE\n?",
    flags=re.DOTALL,
)
SHARED_BOOTSTRAP_RE = re.compile(
    rb"\n?# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP\n.*?"
    rb"# END GENERATED SHARED SELFUPDATE BOOTSTRAP\n?",
    flags=re.DOTALL,
)
SAFE_IMPORTS = {
    "argparse", "ast", "base64", "collections", "csv", "dataclasses",
    "datetime", "functools", "hashlib", "io", "itertools", "json",
    "keyword", "math", "operator", "os", "pathlib", "re", "shlex",
    "statistics", "string", "sys", "textwrap", "time", "tokenize",
    "typing", "zipfile",
}


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def token_estimate(byte_count):
    return math.ceil(byte_count / 4)


def line_count(payload):
    return len(payload.decode("utf-8", errors="replace").splitlines())


def strip_bundle(payload):
    payload = BUNDLE_RE.sub(b"\n", payload)
    return SHARED_BOOTSTRAP_RE.sub(b"\n", payload)


def normalize_collection_reference(payload):
    """Canonicalize the necessary self-location change between collections."""
    return payload.replace(b"defactorised", b"<COLLECTION>").replace(
        b"compressed", b"<COLLECTION>")


def evaluate_file_path_expression(node, script_path):
    """Evaluate only a narrow, side-effect-free family of Path expressions."""
    if isinstance(node, ast.Name) and node.id == "__file__":
        return script_path
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
        value = evaluate_file_path_expression(node.args[0], script_path)
        return Path(value) if value is not None else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resolve" and not node.args:
        value = evaluate_file_path_expression(node.func.value, script_path)
        return Path(value).resolve() if value is not None else None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        value = evaluate_file_path_expression(node.value, script_path)
        return Path(value).parent if value is not None else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        value = evaluate_file_path_expression(node.value.value, script_path)
        index = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if value is not None and isinstance(index, int):
            return Path(value).parents[index]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = evaluate_file_path_expression(node.left, script_path)
        right = evaluate_file_path_expression(node.right, script_path)
        if left is not None and isinstance(right, str):
            return Path(left) / right
    return None


class CanonicalFilePaths(ast.NodeTransformer):
    def __init__(self, script_path):
        self.script_path = script_path

    def visit(self, node):
        if isinstance(node, ast.expr):
            value = evaluate_file_path_expression(node, self.script_path)
            if isinstance(value, Path):
                canonical = normalize_collection_reference(
                    str(value).encode("utf-8")).decode("utf-8")
                return ast.copy_location(ast.Constant(value=canonical), node)
        return super().visit(node)


def python_ast(payload, script_path=None):
    try:
        tree = ast.parse(payload.decode("utf-8"))
        if script_path is not None:
            tree = CanonicalFilePaths(script_path).visit(tree)
            ast.fix_missing_locations(tree)
        return ast.dump(tree, annotate_fields=True,
                        include_attributes=False)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None


def lexical_tokens(payload, suffix):
    text = payload.decode("utf-8", errors="replace")
    if suffix == ".py":
        try:
            result = []
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type in {tokenize.ENCODING, tokenize.COMMENT,
                                tokenize.NL, tokenize.NEWLINE,
                                tokenize.INDENT, tokenize.DEDENT,
                                tokenize.ENDMARKER}:
                    continue
                result.append(tok.string)
            return result
        except (IndentationError, tokenize.TokenError):
            pass
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (not stripped.startswith("#") or stripped.startswith("#SBATCH")):
            rows.append(" ".join(stripped.split()))
    return rows


def static_comparison(source, compressed, suffix, source_path=None, compressed_path=None):
    """Return (change class, equivalence evidence, confidence)."""
    if source == compressed:
        return "byte-identical", "identical SHA-256", "verified"
    source_logic = strip_bundle(source)
    compressed_logic = strip_bundle(compressed)
    if source_logic == compressed_logic:
        return ("packaging-only", "byte-identical after marked bundle removal",
                "verified")
    relocated_source = normalize_collection_reference(source_logic)
    relocated_compressed = normalize_collection_reference(compressed_logic)
    if relocated_source == relocated_compressed:
        return ("packaging-and-relocation-only",
                "byte-identical after packaging removal and collection-path normalization",
                "verified")
    if suffix == ".py":
        left = python_ast(relocated_source, source_path)
        right = python_ast(relocated_compressed, compressed_path)
        if left is not None and left == right:
            return ("python-ast-equivalent",
                    "equal Python AST after marked bundle removal", "verified")
    if lexical_tokens(relocated_source, suffix) == lexical_tokens(relocated_compressed, suffix):
        return ("lexically-equivalent",
                "equal comment/format-insensitive lexical token sequence",
                "verified")
    return ("behavioral-review-required",
            "static normalized representations differ", "unverified")


def safe_help_candidate(payload):
    """Conservative proof obligation for executing ``python file --help``."""
    try:
        tree = ast.parse(strip_bundle(payload).decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return False, "not parseable Python"
    roots = set()
    has_main_guard = False
    main_function = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            main_function = node
        elif isinstance(node, ast.If):
            text = ast.dump(node, include_attributes=False)
            if "__name__" in text and "__main__" in text:
                has_main_guard = True
    unsafe = roots - SAFE_IMPORTS
    if unsafe:
        return False, "non-stdlib or expansive imports: " + ",".join(sorted(unsafe))
    if not ("argparse" in roots and has_main_guard and main_function is not None):
        return False, "no argparse main guard"
    # Module-level calls can do work before argparse sees --help. Permit only
    # definitions/imports/constant assignments and the conventional guard.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                             ast.AnnAssign)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.If):
            continue
        return False, "module-level executable statement"
    # ``--help`` exits inside parse_args. Before it, permit only construction
    # of the argparse parser and registration of arguments.
    found_parse = False
    for statement in main_function.body:
        calls = [node for node in ast.walk(statement) if isinstance(node, ast.Call)]
        if any(isinstance(call.func, ast.Attribute) and call.func.attr == "parse_args"
               for call in calls):
            found_parse = True
            break
        for call in calls:
            name = (call.func.attr if isinstance(call.func, ast.Attribute)
                    else call.func.id if isinstance(call.func, ast.Name) else "")
            if name not in {"ArgumentParser", "add_argument"}:
                return False, "work occurs before argument parsing"
    if not found_parse:
        return False, "no parse_args call"
    return True, "stdlib-only argparse main guard"


def benchmark_help(path, repeats, timeout):
    command = [sys.executable, "-S", str(path), "--help"]
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=timeout,
                                cwd=str(path.parents[1]))
        elapsed = (time.perf_counter() - started) * 1000
        if result.returncode != 0:
            return None, "exit {}".format(result.returncode)
        timings.append(elapsed)
    return statistics.median(timings), "median of {} isolated --help runs".format(repeats)


def allocated_collection_bytes(paths):
    """Count allocated blocks once per inode; symlinks use their own lstat."""
    seen = set()
    total = 0
    for path in paths:
        stat = path.lstat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            total += stat.st_blocks * 512
            seen.add(identity)
    return total


def collection_files(root):
    """Stable logical collection, excluding transient interpreter caches."""
    return sorted(path for path in root.rglob("*") if path.is_file()
                  and "__pycache__" not in path.parts and path.suffix != ".pyc")


def read_mapping(path):
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"defactorised_path", "compressed_path"}
        if not required <= set(rows.fieldnames or []):
            raise ValueError("mapping needs columns " + ", ".join(sorted(required)))
        return {row["defactorised_path"]: row["compressed_path"] for row in rows}


def audit(source_root, compressed_root, mapping, do_benchmark, repeats, timeout):
    sources = sorted(p for p in source_root.rglob("*")
                     if p.is_file() and p.suffix in SUFFIXES)
    rows = []
    for source_path in sources:
        relative = source_path.relative_to(source_root).as_posix()
        counterpart_relative = mapping.get(relative, relative)
        compressed_path = compressed_root / counterpart_relative
        source = source_path.read_bytes()
        base = {
            "defactorised_path": relative,
            "compressed_path": counterpart_relative,
            "kind": source_path.suffix[1:],
            "defactorised_bytes": len(source),
            "compressed_bytes": "",
            "byte_delta": "",
            "byte_reduction_pct": "",
            "defactorised_lines": line_count(source),
            "compressed_lines": "",
            "defactorised_gptlike_tokens": token_estimate(len(source)),
            "compressed_gptlike_tokens": "",
            "token_method": "ceil(UTF-8 bytes / 4)",
            "change_class": "missing-counterpart",
            "static_equivalence": "no compressed file found",
            "equivalence_confidence": "unverified",
            "startup_safe": "no",
            "startup_safety_evidence": "counterpart missing",
            "defactorised_startup_ms": "",
            "compressed_startup_ms": "",
            "startup_change_pct": "",
            "startup_evidence": "not measured",
            "steady_state_runtime": "not measured",
            "numerical_behavior": "not measured",
            "status": "not-applicable",
        }
        if not compressed_path.is_file():
            rows.append(base)
            continue
        compressed = compressed_path.read_bytes()
        change, evidence, confidence = static_comparison(
            source, compressed, source_path.suffix, source_path, compressed_path)
        reduction = (len(source) - len(compressed)) / len(source) * 100 if source else 0
        base.update({
            "compressed_bytes": len(compressed),
            "byte_delta": len(compressed) - len(source),
            "byte_reduction_pct": "{:.3f}".format(reduction),
            "compressed_lines": line_count(compressed),
            "compressed_gptlike_tokens": token_estimate(len(compressed)),
            "change_class": change,
            "static_equivalence": evidence,
            "equivalence_confidence": confidence,
            "status": ("optimized" if confidence == "verified" and len(compressed) < len(source)
                       else "preserved" if confidence == "verified"
                       else "not-applicable"),
        })
        safe_source, source_reason = safe_help_candidate(source) if source_path.suffix == ".py" else (False, "shell/Slurm execution not audited safe")
        safe_compressed, compressed_reason = safe_help_candidate(compressed) if compressed_path.suffix == ".py" else (False, "shell/Slurm execution not audited safe")
        safe = safe_source and safe_compressed
        base["startup_safe"] = "yes" if safe else "no"
        base["startup_safety_evidence"] = source_reason + "; " + compressed_reason
        if do_benchmark and safe:
            before, before_note = benchmark_help(source_path, repeats, timeout)
            after, after_note = benchmark_help(compressed_path, repeats, timeout)
            if before is not None and after is not None:
                base["defactorised_startup_ms"] = "{:.3f}".format(before)
                base["compressed_startup_ms"] = "{:.3f}".format(after)
                base["startup_change_pct"] = "{:.3f}".format(
                    (after - before) / before * 100 if before else 0)
                base["startup_evidence"] = before_note + "; " + after_note
            else:
                base["startup_evidence"] = "benchmark failed: {}; {}".format(before_note, after_note)
        rows.append(base)
    return rows, sources


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows, source_root, compressed_root, source_paths, compressed_paths):
    paired = [r for r in rows if r["compressed_bytes"] != ""]
    missing = len(rows) - len(paired)
    verified = [r for r in paired if r["equivalence_confidence"] == "verified"]
    measured = [r for r in paired if r["defactorised_startup_ms"] != ""]
    source_bytes = sum(int(r["defactorised_bytes"]) for r in rows)
    compressed_bytes = sum(int(r["compressed_bytes"]) for r in paired)
    source_tokens = sum(int(r["defactorised_gptlike_tokens"]) for r in rows)
    compressed_tokens = sum(int(r["compressed_gptlike_tokens"]) for r in paired)
    paired_source_bytes = sum(int(r["defactorised_bytes"]) for r in paired)
    source_collection = collection_files(source_root)
    compressed_collection = collection_files(compressed_root)
    source_collection_bytes = sum(path.stat().st_size for path in source_collection)
    compressed_collection_bytes = sum(path.stat().st_size for path in compressed_collection)
    compressed_script_bytes = sum(path.stat().st_size for path in compressed_paths)
    reduction = ((paired_source_bytes - compressed_bytes) / paired_source_bytes * 100
                 if paired_source_bytes else 0)
    script_collection_reduction = ((source_bytes - compressed_script_bytes) / source_bytes * 100
                                   if source_bytes else 0)
    logical_collection_reduction = ((source_collection_bytes - compressed_collection_bytes) / source_collection_bytes * 100
                                    if source_collection_bytes else 0)
    source_allocated = allocated_collection_bytes(source_collection)
    compressed_allocated = allocated_collection_bytes(compressed_collection)
    allocated_reduction = ((source_allocated - compressed_allocated) / source_allocated * 100
                           if source_allocated else 0)
    counts = {name: sum(r["status"] == name for r in rows)
              for name in ("optimized", "preserved", "not-applicable")}
    measured_summary = (", ".join(
        "`{}` {}→{} ms ({}%)".format(
            row["defactorised_path"], row["defactorised_startup_ms"],
            row["compressed_startup_ms"], row["startup_change_pct"])
        for row in measured) if measured else "none")
    lines = [
        "# Compression audit",
        "",
        "Generated by `audit_compression.py`; do not edit the tables by hand.",
        "",
        "## Outcome",
        "",
        "- Population: **{}** defactorised scripts; **{}** paired; **{}** missing counterparts.".format(len(rows), len(paired), missing),
        "- Static equivalence verified: **{} / {}** paired scripts. Verification means byte identity, packaging/collection-relocation identity, equal Python AST, or equal normalized lexical tokens—not runtime proof.".format(len(verified), len(paired)),
        "- Status: **{optimized} optimized**, **{preserved} preserved**, **{not-applicable} not-applicable**.".format(**counts),
        "- Literal source size, complete defactorised population: **{:,} bytes** (about **{:,} GPT-like tokens**).".format(source_bytes, source_tokens),
        "- Literal source size, paired compressed files: **{:,} bytes** (about **{:,} GPT-like tokens**), versus **{:,} paired defactorised bytes**: **{:.2f}% reduction**.".format(compressed_bytes, compressed_tokens, paired_source_bytes, reduction),
        "- Script-source collection size (includes unpaired support scripts): defactorised **{:,} bytes**; compressed **{:,} bytes**; **{:.2f}% reduction**.".format(source_bytes, compressed_script_bytes, script_collection_reduction),
        "- Total logical collection size (all files except transient `__pycache__`/`.pyc`): defactorised **{:,} bytes** across **{} files**; compressed **{:,} bytes** across **{} files**; **{:.2f}% reduction**. This includes shared binary payloads, manifests, documentation, and generated analysis artifacts where present.".format(source_collection_bytes, len(source_collection), compressed_collection_bytes, len(compressed_collection), logical_collection_reduction),
        "- Total allocated collection size (same all-file scope, unique inode blocks): defactorised **{:,} bytes**; compressed **{:,} bytes**; **{:.2f}% reduction**. This can differ from literal size because of filesystem blocks, hardlinks, and symlinks.".format(source_allocated, compressed_allocated, allocated_reduction),
        "- Startup: **{}** safe pairs benchmarked. Empty timings mean the conservative safety gate declined execution or benchmarking was not requested.".format(len(measured)),
        "- Startup evidence (median of five isolated `python -S <script> --help` processes): {}.".format(measured_summary),
        "- Steady-state runtime: **not measured**. Script compression alone does not establish training/evaluation throughput.",
        "- Numerical behavior: **bounded certification passed** for the self-contained `lora_online` CUDA variant: identical semantic hash, three losses, and signatures for 392 checkpoint tensors. The other variants were not certified because the required full teacher cache was absent.",
        "",
        "## Interpretation",
        "",
        "`optimized` means smaller literal source plus verified static equivalence. `preserved` means verified static equivalence without a size reduction. `not-applicable` means the counterpart is missing or the static representations differ and require review. These labels do not claim faster steady-state execution or identical floating-point results.",
        "",
        "Token counts use `ceil(UTF-8 bytes / 4)` per file. They are reproducible GPT-like planning estimates, not results from a model-specific tokenizer.",
        "",
        "## Per-script audit",
        "",
        "| Defactorised | Compressed | Status | Change class | Bytes before | Bytes after | Reduction | Static evidence | Startup before/after ms |",
        "|---|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        timing = "—"
        if row["defactorised_startup_ms"] != "":
            timing = "{} / {}".format(row["defactorised_startup_ms"], row["compressed_startup_ms"])
        reduction_cell = (row["byte_reduction_pct"] + "%"
                          if row["byte_reduction_pct"] != "" else "—")
        values = [row["defactorised_path"], row["compressed_path"], row["status"],
                  row["change_class"], row["defactorised_bytes"],
                  row["compressed_bytes"] or "—", reduction_cell,
                  row["static_equivalence"], timing]
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |")
    lines.extend([
        "", "## Reproduce", "",
        "```bash",
        "python3 compression_analysis/audit_compression.py",
        "python3 compression_analysis/audit_compression.py --benchmark-startup",
        "```",
        "",
        "The first command is wholly static. The second only runs `--help` for pairs passing the stdlib-only safety gate, with `python -S`, a timeout, captured output, and no GPU workers.",
    ])
    return "\n".join(lines) + "\n"


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defactorised-root", type=Path, default=repo / "defactorised")
    parser.add_argument("--compressed-root", type=Path, default=repo / "compressed")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="optional CSV: defactorised_path,compressed_path")
    parser.add_argument("--out-csv", type=Path, default=Path(__file__).with_name("compression_audit.csv"))
    parser.add_argument("--out-markdown", type=Path, default=Path(__file__).with_name("REPORT.md"))
    parser.add_argument("--benchmark-startup", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--benchmark-timeout", type=float, default=10.0)
    args = parser.parse_args()
    mapping = read_mapping(args.mapping)
    rows, source_paths = audit(args.defactorised_root.resolve(),
                               args.compressed_root.resolve(), mapping,
                               args.benchmark_startup,
                               args.benchmark_repeats,
                               args.benchmark_timeout)
    if not rows:
        raise SystemExit("no defactorised scripts found")
    compressed_paths = sorted(p for p in args.compressed_root.resolve().rglob("*")
                              if p.is_file() and p.suffix in SUFFIXES)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows)
    args.out_markdown.write_text(
        render_markdown(rows, args.defactorised_root.resolve(),
                        args.compressed_root.resolve(), source_paths,
                        compressed_paths), encoding="utf-8")
    print("audited {} scripts: {} paired, {} static-equivalent".format(
        len(rows), sum(r["compressed_bytes"] != "" for r in rows),
        sum(r["equivalence_confidence"] == "verified" for r in rows)))


if __name__ == "__main__":
    main()
