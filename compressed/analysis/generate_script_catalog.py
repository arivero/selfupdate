#!/usr/bin/env python3
"""Generate the reproducible inventory of defactorised executable sources.

The inventory is deliberately static analysis: files are decoded and inspected,
never imported or executed.  Run this file from any directory; paths in the
outputs are relative to ``defactorised/``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2] / "defactorised"
CSV_PATH = ROOT / "analysis" / "script_catalog.csv"
MARKDOWN_PATH = ROOT / "analysis" / "SCRIPT_CATALOG.md"
SCRIPT_SUFFIXES = {".py", ".sh", ".sbatch"}
INFRASTRUCTURE = {
    "analysis/code_genealogy.py",
    "analysis/generate_script_catalog.py",
    "generate.py",
    "shell_helpers.py",
    "variants/_forward.py",
}
PURPOSE_OVERRIDES = {
    "evaluate_v31_4b_full_standard.sh":
        "Evaluate the Qwen3.5-4B base and named v3.1 checkpoints on all three 100-item standard-damage tasks.",
    "gpu_speed_backfill.sh":
        "Continuously launch monitored speed checks when configured GPUs are sufficiently idle and have enough free memory.",
    "gpu_util_monitor.sh":
        "Periodically append utilization, memory, and power telemetry for configured GPUs to CSV.",
    "refresh_v31_0p8b_full_damage_reports.sh":
        "Regenerate report-v2 artifacts for the fixed v3.1 0.8B B256K16 cohort under an interprocess lock.",
    "refresh_v31_reports.sh":
        "Regenerate report-v2 artifacts for named runs under an interprocess lock and retain compact diagnostics.",
}
TOKEN_METHOD = "UTF-8 bytes / 4, rounded up (tiktoken unavailable)"
BUNDLE_RE = re.compile(
    rb"\n?# BEGIN GENERATED STANDALONE SELFUPDATE BUNDLE\n.*?"
    rb"# END GENERATED STANDALONE SELFUPDATE BUNDLE\n?",
    flags=re.DOTALL,
)


def one_line(value: str) -> str:
    """Collapse prose to a compact first sentence suitable for a table."""
    value = re.sub(r"\s+", " ", value).strip().strip("# ")
    if not value:
        return value
    match = re.search(r"(?<=[.!?])\s", value)
    if match:
        value = value[: match.start() + 1]
    return value[:237].rstrip() + ("..." if len(value) > 237 else "")


def python_purpose(text: str, relative: str) -> str:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        doc = ast.get_docstring(tree, clean=True)
        if doc:
            paragraph = doc.split("\n\n", 1)[0]
            purpose = one_line(paragraph)
            if purpose:
                return purpose

    # Purpose-specific variants are intentionally tiny and communicate their
    # behavior in the fixed launch(...) call rather than a module docstring.
    launch = re.search(
        r"launch\(\s*[\"']([^\"']+)[\"']\s*,\s*\((.*?)\)\s*,",
        text,
        flags=re.DOTALL,
    )
    if launch:
        fixed = re.findall(r"[\"']([^\"']+)[\"']", launch.group(2))
        if fixed:
            return f"Purpose-specific launcher for {launch.group(1)} with {' '.join(fixed)}."
        name = Path(relative).stem.replace("_", " ")
        return f"Purpose-specific {name} launcher for {launch.group(1)}; it adds no fixed CLI flags."

    # Generated standalone files retain a docstring at byte zero.  This is a
    # fallback for unusual Python entry points without one.
    description = re.search(r"description\s*=\s*[\"']([^\"']+)[\"']", text)
    if description:
        return one_line(description.group(1))
    return "Python entry point; consult its --help output for the accepted workflow."


def shell_purpose(text: str) -> str:
    comments: list[str] = []
    for raw in text.splitlines()[:80]:
        line = raw.strip()
        if line.startswith("#!") or line.startswith("#SBATCH"):
            continue
        if line.startswith("#"):
            value = line[1:].strip()
            if value and not set(value) <= {"-", "=", "_"}:
                comments.append(value)
            continue
        if not line:
            if comments:
                break
            continue
        if comments:
            break
    if comments:
        return one_line(" ".join(comments))
    return "Shell orchestration entry point; consult its usage text and environment variables."


def category(relative: str, suffix: str) -> tuple[str, bool]:
    if relative in INFRASTRUCTURE:
        return "catalog/build infrastructure", False
    if relative.startswith("demos/"):
        return "teaching demo", True
    if relative.startswith("variants/"):
        return "fixed-option variant", True
    if suffix == ".sbatch":
        return "Slurm launcher", True
    if suffix == ".sh":
        return "shell orchestration", True
    return "Python entry point", True


def language(suffix: str) -> str:
    return {".py": "Python", ".sh": "Bash", ".sbatch": "Bash/Slurm"}[suffix]


def estimate_tokens(byte_count: int) -> int:
    """Return the conventional GPT planning estimate of four bytes/token."""
    return math.ceil(byte_count / 4)


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix in SCRIPT_SUFFIXES
    )
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        entrypoint_payload = BUNDLE_RE.sub(b"\n", payload)
        text = payload.decode("utf-8")
        kind, user_facing = category(relative, path.suffix)
        purpose = PURPOSE_OVERRIDES.get(relative)
        if purpose is None:
            purpose = python_purpose(text, relative) if path.suffix == ".py" else shell_purpose(text)
        result.append(
            {
                "path": relative,
                "language": language(path.suffix),
                "category": kind,
                "user_facing": "yes" if user_facing else "no",
                "lines": len(text.splitlines()),
                "bytes": len(payload),
                "standalone_file_tokens": estimate_tokens(len(payload)),
                "entrypoint_logic_tokens": estimate_tokens(len(entrypoint_payload)),
                "purpose": purpose,
            }
        )
    return result


def render_csv(items: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(items[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(items)
    return output.getvalue()


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(items: list[dict[str, object]]) -> str:
    total_bytes = sum(int(item["bytes"]) for item in items)
    total_tokens = sum(int(item["standalone_file_tokens"]) for item in items)
    logic_tokens = sum(int(item["entrypoint_logic_tokens"]) for item in items)
    visible = sum(item["user_facing"] == "yes" for item in items)
    lines = [
        "# Defactorised script catalog",
        "",
        "Generated by `generate_script_catalog.py`; do not edit the table by hand.",
        "The generator performs static inspection only and never imports a script.",
        "",
        "## Coverage and measurement",
        "",
        f"- Cataloged: **{len(items)}** executable-source files "
        f"({visible} user-facing, {len(items) - visible} infrastructure).",
        "- Scope: every `.py`, `.sh`, and `.sbatch` below `defactorised/`, "
        "including flat entry points, fixed-option variants, demos, and this generator.",
        f"- Total literal source size: **{total_bytes:,} bytes**; estimated "
        f"**{total_tokens:,} standalone GPT-like tokens** and **{logic_tokens:,} "
        "entrypoint-logic GPT-like tokens**.",
        f"- Token method: **{TOKEN_METHOD}**. The host Python had no `tiktoken` "
        "installation when this catalog was generated. The estimate is intentionally "
        "dependency-free and reproducible: `ceil(UTF-8 byte count / 4)` per file. "
        "It is a planning estimate, not a model-specific tokenizer result; encoded "
        "standalone bundles and code punctuation can make actual counts differ.",
        "- `Standalone tokens` measures the literal complete file, including its embedded "
        "`selfupdate` payload. `Logic tokens` removes the text from the generated bundle "
        "BEGIN marker through its END marker. For files without that generated block the "
        "two values are identical.",
        "- Lines use Python `splitlines()`: a final newline does not create an extra line.",
        "- `user_facing=no` identifies generation/forwarding infrastructure; it remains "
        "cataloged so coverage is auditable.",
        "",
        "## Inventory",
        "",
        "| Path | Language | Category | User-facing | Lines | Bytes | Standalone tokens | Logic tokens | Purpose |",
        "|---|---|---|:---:|---:|---:|---:|---:|---|",
    ]
    for item in items:
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item[key])
                for key in (
                    "path",
                    "language",
                    "category",
                    "user_facing",
                    "lines",
                    "bytes",
                    "standalone_file_tokens",
                    "entrypoint_logic_tokens",
                    "purpose",
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            print(f"out of date: {path.relative_to(ROOT)}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    args = parser.parse_args()
    items = rows()
    valid = write_or_check(CSV_PATH, render_csv(items), args.check)
    valid &= write_or_check(MARKDOWN_PATH, render_markdown(items), args.check)
    if not args.check:
        print(f"cataloged {len(items)} scripts")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
