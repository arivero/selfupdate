#!/usr/bin/env python3
"""Generate standalone copies of every tracked ``scripts/*.py`` entry point.

The generated files which use ``selfupdate`` carry a compressed snapshot of
the package and import it from memory.  They therefore never add or read the
repository's ``src`` directory.  Entry points which do not import the package
remain ordinary, readable copies with only obsolete path bootstraps removed.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PACKAGE = ROOT / "src" / "selfupdate"
OUT = ROOT / "defactorised"


LOADER = '''# BEGIN GENERATED STANDALONE SELFUPDATE BUNDLE
import base64 as _su_b64
import importlib.abc as _su_abc
import importlib.util as _su_util
import io as _su_io
import os as _su_os
import sys as _su_sys
import zipfile as _su_zipfile

_SU_ARCHIVE = _su_zipfile.ZipFile(_su_io.BytesIO(
    _su_b64.b85decode({payload!r})))
_SU_VIRTUAL_ROOT = _su_os.path.dirname(_su_os.path.abspath(__file__))


class _SelfupdateBundleFinder(_su_abc.MetaPathFinder, _su_abc.Loader):
    def _entry(self, fullname):
        stem = fullname.replace(".", "/")
        package = stem + "/__init__.py"
        if package in _SU_ARCHIVE.namelist():
            return package, True
        module = stem + ".py"
        if module in _SU_ARCHIVE.namelist():
            return module, False
        return None

    def find_spec(self, fullname, path=None, target=None):
        found = self._entry(fullname)
        if found is None:
            return None
        entry, is_package = found
        return _su_util.spec_from_loader(
            fullname, self, origin="<standalone>/" + entry,
            is_package=is_package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        entry, is_package = self._entry(module.__name__)
        # Preserve the original three-parent repo-root convention used by
        # selfupdate.utils.runlog, without placing or reading files here.
        filename = _su_os.path.join(_SU_VIRTUAL_ROOT, entry)
        module.__file__ = filename
        if is_package:
            module.__path__ = [filename.rsplit("/__init__.py", 1)[0]]
        source = _SU_ARCHIVE.read(entry)
        exec(compile(source, filename, "exec"), module.__dict__)


_su_sys.meta_path.insert(0, _SelfupdateBundleFinder())
# END GENERATED STANDALONE SELFUPDATE BUNDLE
'''


def _tracked_scripts() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "scripts/*.py"], cwd=ROOT,
        check=True, text=True, capture_output=True)
    return [ROOT / line for line in result.stdout.splitlines() if line]


def _package_payload() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(PACKAGE.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            # A standalone v4.5 trainer must self-invoke its standalone copy.
            if path.relative_to(PACKAGE).as_posix() == "train/online_v4.py":
                source = source.replace(
                    ' / "scripts" /\n                          "train.py"',
                    ' / "defactorised" /\n                          "train.py"')
            info = zipfile.ZipInfo(
                "selfupdate/" + path.relative_to(PACKAGE).as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, source, compresslevel=9)
    return base64.b85encode(buffer.getvalue()).decode("ascii")


def _imports_selfupdate(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "selfupdate" or
                   alias.name.startswith("selfupdate.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "selfupdate" or (node.module or "").startswith(
                    "selfupdate."):
                return True
    return False


def _bootstrap_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Locate only path mutations which select repo source/script modules."""
    ranges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert" and
                isinstance(func.value, ast.Attribute) and
                isinstance(func.value.value, ast.Name) and
                func.value.value.id == "sys" and func.value.attr == "path"):
            continue
        text = ast.unparse(call)
        if '"src"' in text or "'src'" in text or "/ 'scripts'" in text or \
                '/ "scripts"' in text:
            ranges.append((node.lineno, node.end_lineno or node.lineno))
    return ranges


def _insertion_line(tree: ast.Module) -> int:
    line = 0
    body = tree.body
    index = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        line = body[0].end_lineno or body[0].lineno
        index = 1
    while index < len(body) and isinstance(body[index], ast.ImportFrom) and \
            body[index].module == "__future__":
        line = body[index].end_lineno or body[index].lineno
        index += 1
    return line


def _rewrite_entrypoint(source: str, payload: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for first, last in reversed(_bootstrap_ranges(tree)):
        del lines[first - 1:last]
    source = "".join(lines)
    # Child/generated Python entry points must remain inside the standalone
    # collection. Do not rewrite non-Python queue/data paths under scripts/.
    source = re.sub(
        r"scripts/([A-Za-z0-9_]+\.py)",
        lambda match: ("defactorised/" + match.group(1)
                       if (SCRIPTS / match.group(1)).is_file()
                       else match.group(0)),
        source)
    source = re.sub(
        r'([A-Z_]+) / (["\'])scripts\2 / (["\'])([A-Za-z0-9_]+\.py)\3',
        lambda match: (f'{match.group(1)} / {match.group(2)}defactorised'
                       f'{match.group(2)} / {match.group(3)}{match.group(4)}'
                       f'{match.group(3)}'
                       if (SCRIPTS / match.group(4)).is_file()
                       else match.group(0)),
        source)
    if not _imports_selfupdate(tree):
        return source
    reparsed = ast.parse(source)
    at = _insertion_line(reparsed)
    output = source.splitlines(keepends=True)
    output.insert(at, "\n" + LOADER.format(payload=payload) + "\n")
    return "".join(output)


def _render_ppn_demo() -> str:
    """Inline just the cost-profile partitioner for an inspectable demo."""
    module_source = (PACKAGE / "train" / "ppn.py").read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    wanted = {
        "_strict_splits", "PPnPartition", "BlockCost", "CostProfile",
        "PartitionConstraints", "_segment_memory", "choose_partition",
    }
    pieces = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            pieces.append(ast.get_source_segment(module_source, node))
    entry = (SCRIPTS / "ppn_partition.py").read_text(encoding="utf-8")
    entry_lines = entry.splitlines(keepends=True)
    for first_line, last_line in reversed(_bootstrap_ranges(ast.parse(entry))):
        del entry_lines[first_line - 1:last_line]
    entry = "".join(entry_lines)
    entry = entry.replace(
        "from selfupdate.train.ppn import CostProfile, PartitionConstraints, choose_partition\n",
        "")
    header = '''#!/usr/bin/env python3
"""Readable, single-file PPn cost partitioning demo.

This is the focused architecture walk-through: measured per-block cost and
memory records feed a contiguous dynamic program, which emits explicit stage
boundaries.  It is generated from the production definitions, with no bundle
loader and no import from ``src/selfupdate``.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_VERSION = "3.4"
PIPELINE_VERSION = 3
PIPELINE_REVISION = "3.2"


class PPnError(RuntimeError):
    pass

'''
    # Drop the entry point's shebang/docstring/import preamble, retaining CLI.
    entry_tree = ast.parse(entry)
    first = next(node for node in entry_tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    cli = "\n".join(entry.splitlines()[first.lineno - 1:]) + "\n"
    return header + "\n\n".join(pieces) + "\n\n" + cli


def _render_shell_helpers(payload: str) -> str:
    return '''#!/usr/bin/env python3
"""Standalone replacements for shell heredocs which imported selfupdate."""

from __future__ import annotations

''' + LOADER.format(payload=payload) + '''
import argparse
import hashlib


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("v4-launch-info", "node-cache-index"):
        command = commands.add_parser(name)
        command.add_argument("base")
        command.add_argument("experiment")
    commands.add_parser("venv-import-check")
    args = parser.parse_args()

    if args.command == "venv-import-check":
        import selfupdate
        from selfupdate.eval.standard import STANDARD_TASKS
        digest = hashlib.sha256()
        for entry in sorted(_SU_ARCHIVE.namelist()):
            digest.update(entry.encode("utf-8") + b"\\0")
            digest.update(_SU_ARCHIVE.read(entry))
        print(f"eval.standard ok ({', '.join(list(STANDARD_TASKS)[:3])})")
        print(f"selfupdate bundle {selfupdate.__file__}")
        print(f"bundle sha256 {digest.hexdigest()}")
        return 0

    from selfupdate.config import load_config
    cfg = load_config(args.base, args.experiment)
    if args.command == "v4-launch-info":
        if cfg.train.pipeline_version != 4:
            raise SystemExit(
                "launch_v4_stages.sh requires pipeline_version=4")
        print(len(cfg.train.v4_stage_splits or []) + 1, cfg.run_name)
        return 0

    from selfupdate.teacher.cache import resolve_cache_dir
    from selfupdate.teacher.node_epoch0 import ready_manifest, runtime_identity
    root, cache_hash = resolve_cache_dir(cfg)
    if ready_manifest(root, cache_hash, compatibility=runtime_identity()) is None:
        raise SystemExit(
            f"node-epoch0 index was not atomically published: {root}")
    print(root / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if generated files differ; do not write")
    args = parser.parse_args()
    scripts = _tracked_scripts()
    payload = _package_payload()
    changed = []
    for script in scripts:
        rendered = _rewrite_entrypoint(script.read_text(encoding="utf-8"), payload)
        target = OUT / script.name
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != rendered:
            changed.append(target.relative_to(ROOT).as_posix())
            if not args.check:
                target.write_text(rendered, encoding="utf-8")
                target.chmod(script.stat().st_mode)
    demo = OUT / "demos" / "ppn_partition_readable.py"
    rendered_demo = _render_ppn_demo()
    current_demo = demo.read_text(encoding="utf-8") if demo.exists() else None
    if current_demo != rendered_demo:
        changed.append(demo.relative_to(ROOT).as_posix())
        if not args.check:
            demo.parent.mkdir(parents=True, exist_ok=True)
            demo.write_text(rendered_demo, encoding="utf-8")
            demo.chmod(0o755)
    helper = OUT / "shell_helpers.py"
    rendered_helper = _render_shell_helpers(payload)
    current_helper = helper.read_text(encoding="utf-8") if helper.exists() else None
    if current_helper != rendered_helper:
        changed.append(helper.relative_to(ROOT).as_posix())
        if not args.check:
            helper.write_text(rendered_helper, encoding="utf-8")
            helper.chmod(0o755)
    if changed:
        print("out of date:" if args.check else "generated:", *changed, sep="\n  ")
        return 1 if args.check else 0
    print(f"up to date: {len(scripts)} Python entry points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
