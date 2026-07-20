#!/usr/bin/env python3
"""Generate a deduplicated Python mirror of ``defactorised/``.

Every source Python entry point is represented under ``compressed/``.  The
large ``selfupdate`` archive appears once in ``_selfupdate_bundle.zip``;
``_shared_bundle.py`` is its small integrity-checking importer. Entry points
which previously carried a private copy install that shared importer. Runtime
code never reads ``src/selfupdate``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
from pathlib import Path
import re
import sys
import zipfile


REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "defactorised"
OUT = REPO / "compressed"
MANIFEST = OUT / "MANIFEST.csv"
INFRA_GENERATOR = OUT / "_generate_compressed.py"
ARCHIVE_PATH = OUT / "_selfupdate_bundle.zip"
ARCHIVE_HASH_PATH = OUT / "_selfupdate_bundle.sha256"
BUNDLE_RE = re.compile(
    r"\n?# BEGIN GENERATED STANDALONE SELFUPDATE BUNDLE\n.*?"
    r"# END GENERATED STANDALONE SELFUPDATE BUNDLE\n?",
    flags=re.DOTALL,
)
BOOTSTRAP = '''
# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP
'''


SHARED_TEMPLATE = '''#!/usr/bin/env python3
"""Integrity-checked importer for the shared compressed selfupdate archive."""

import hashlib as _su_hashlib
import importlib.abc as _su_abc
import importlib.util as _su_util
import os as _su_os
import sys as _su_sys
import zipfile as _su_zipfile

_SU_VIRTUAL_ROOT = _su_os.path.dirname(_su_os.path.abspath(__file__))
_SU_ARCHIVE_PATH = _su_os.path.join(_SU_VIRTUAL_ROOT, "_selfupdate_bundle.zip")
_SU_EXPECTED_SHA256 = {archive_sha256!r}
_SU_VALIDATED = False
_SU_ARCHIVE = _su_zipfile.ZipFile(_SU_ARCHIVE_PATH)


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
            fullname, self, origin="<compressed-standalone>/" + entry,
            is_package=is_package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        entry, is_package = self._entry(module.__name__)
        # compressed/selfupdate/... has the same depth as the historical
        # defactorised virtual root, preserving repo-root provenance lookup.
        filename = _su_os.path.join(_SU_VIRTUAL_ROOT, entry)
        module.__file__ = filename
        if is_package:
            module.__path__ = [filename.rsplit("/__init__.py", 1)[0]]
        source = _SU_ARCHIVE.read(entry)
        exec(compile(source, filename, "exec"), module.__dict__)


def install():
    """Validate the binary archive and install the finder once."""
    global _SU_VALIDATED
    if not _SU_VALIDATED:
        digest = _su_hashlib.sha256()
        with open(_SU_ARCHIVE_PATH, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != _SU_EXPECTED_SHA256:
            raise RuntimeError(
                "compressed selfupdate archive integrity failure: " + actual)
        _SU_VALIDATED = True
    if not any(isinstance(item, _SelfupdateBundleFinder)
               for item in _su_sys.meta_path):
        _su_sys.meta_path.insert(0, _SelfupdateBundleFinder())


def archive():
    """Expose the archive for the existing shell-helper integrity check."""
    return _SU_ARCHIVE
'''


README = '''# Compressed script collection

This directory mirrors every executable Python, shell, and Slurm script below
`defactorised/`.  It is the shorter, faster-to-load companion to that explicit
teaching collection; `defactorised/` remains the readable standalone baseline.

Like its source collection, this is a **frozen pre-v4-cleanup teaching
snapshot**. Historical v2/v3 programs remain here intentionally for genealogy;
they are not executable training methods in the cleaned live `src/` tree.

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
'''


def source_paths():
    return sorted(
        path for path in SOURCE.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def extract_archive():
    """Read and decode one private archive without importing its script."""
    for path in source_paths():
        text = path.read_text(encoding="utf-8")
        match = BUNDLE_RE.search(text)
        if match is None:
            continue
        tree = ast.parse(match.group(0))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "b85decode":
                value = ast.literal_eval(node.args[0])
                return base64.b85decode(value)
    raise RuntimeError("no generated standalone bundle found in defactorised Python")


def shared_archive_bytes():
    """Repack deterministically and redirect the embedded v4 child process."""
    original = zipfile.ZipFile(io.BytesIO(extract_archive()))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name in sorted(original.namelist()):
            data = original.read(name)
            if name == "selfupdate/train/online_v4.py":
                data = data.replace(
                    b' / "defactorised" / "v4_battery.py"',
                    b' / "compressed" / "v4_battery.py"',
                )
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, data, compresslevel=9)
    return output.getvalue()


def render_entry(relative, source):
    """Remove a private payload and redirect only collection-local behavior."""
    bundled = BUNDLE_RE.search(source) is not None
    rendered = BUNDLE_RE.sub("\n" + BOOTSTRAP + "\n", source)

    analysis_tools = {
        Path("analysis/generate_script_catalog.py"),
        Path("analysis/code_genealogy.py"),
    }
    # References in help text are updated alongside executable child paths;
    # actual data/config paths outside this collection remain untouched. The
    # analysis mirrors are exceptions because their output prose is part of
    # the reproducible defactorised/ baseline they analyze.
    if relative not in analysis_tools:
        rendered = rendered.replace("defactorised/", "compressed/")
        rendered = rendered.replace('"defactorised"', '"compressed"')
        rendered = rendered.replace("'defactorised'", "'compressed'")

    # These two scripts are scientific analysis tools.  Their historical
    # default population/output is defactorised/, not their containing mirror.
    if relative == Path("analysis/generate_script_catalog.py"):
        rendered = rendered.replace(
            'ROOT = Path(__file__).resolve().parents[1]',
            'ROOT = Path(__file__).resolve().parents[2] / "defactorised"',
        )
    elif relative == Path("analysis/code_genealogy.py"):
        rendered = rendered.replace(
            'default=Path(__file__).resolve().parents[1])',
            'default=Path(__file__).resolve().parents[2] / "defactorised")',
        )
        rendered = rendered.replace(
            'default=Path(__file__).resolve().parent / "artifacts")',
            'default=Path(__file__).resolve().parents[2] / "defactorised" / '
            '"analysis" / "artifacts")',
        )
    return rendered, bundled


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest_text(records):
    output = io.StringIO(newline="")
    fields = [
        "source", "target", "kind", "private_bundle_removed",
        "source_bytes", "target_bytes", "source_sha256", "target_sha256",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def update(path, content, mode, check, changed):
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return
    changed.append(path.relative_to(REPO).as_posix())
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(mode)


def update_bytes(path, content, mode, check, changed):
    current = path.read_bytes() if path.exists() else None
    if current == content:
        return
    changed.append(path.relative_to(REPO).as_posix())
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if generated collection or manifest differs")
    args = parser.parse_args()
    paths = source_paths()
    changed = []
    records = []

    # Preserve this compressor under an infrastructure-only name before
    # restoring compressed/generate.py to the mapped defactorised behavior.
    compressor_source = Path(__file__).read_text(encoding="utf-8")
    update(INFRA_GENERATOR, compressor_source, 0o755, args.check, changed)

    archive_bytes = shared_archive_bytes()
    archive_sha = sha256(archive_bytes)
    shared = SHARED_TEMPLATE.format(archive_sha256=archive_sha)
    update(OUT / "_shared_bundle.py", shared, 0o644, args.check, changed)
    update_bytes(ARCHIVE_PATH, archive_bytes, 0o644, args.check, changed)
    update(ARCHIVE_HASH_PATH, archive_sha + "  _selfupdate_bundle.zip\n",
           0o644, args.check, changed)
    update(OUT / "README.md", README, 0o644, args.check, changed)

    for source_path in paths:
        relative = source_path.relative_to(SOURCE)
        source_bytes = source_path.read_bytes()
        if relative == Path("generate.py"):
            # Preserve the original generator's behavior exactly: it rebuilds
            # defactorised/, while _generate_compressed.py builds this mirror.
            rendered = source_bytes.decode("utf-8")
            target = OUT / relative
            update(target, rendered, source_path.stat().st_mode,
                   args.check, changed)
            target_bytes = rendered.encode("utf-8")
            bundled = False
            kind = "plain behavior-equivalent mirror"
        else:
            rendered, bundled = render_entry(
                relative, source_bytes.decode("utf-8"))
            target = OUT / relative
            update(target, rendered, source_path.stat().st_mode,
                   args.check, changed)
            target_bytes = rendered.encode("utf-8")
            kind = "shared-bundle entry" if bundled else "plain mirror"
        records.append({
            "source": (Path("defactorised") / relative).as_posix(),
            "target": target.relative_to(REPO).as_posix(),
            "kind": kind,
            "private_bundle_removed": str(bundled).lower(),
            "source_bytes": len(source_bytes),
            "target_bytes": len(target_bytes),
            "source_sha256": sha256(source_bytes),
            "target_sha256": sha256(target_bytes),
        })

    infrastructure = (
        (INFRA_GENERATOR, compressor_source.encode("utf-8"),
         "collection infrastructure: compressor"),
        (OUT / "_shared_bundle.py", shared.encode("utf-8"),
         "collection infrastructure: shared importer"),
        (ARCHIVE_PATH, archive_bytes,
         "collection infrastructure: binary package archive"),
        (ARCHIVE_HASH_PATH,
         (archive_sha + "  _selfupdate_bundle.zip\n").encode("utf-8"),
         "collection infrastructure: archive checksum"),
    )
    for target, target_bytes, kind in infrastructure:
        records.append({
            "source": "",
            "target": target.relative_to(REPO).as_posix(),
            "kind": kind,
            "private_bundle_removed": "false",
            "source_bytes": "",
            "target_bytes": len(target_bytes),
            "source_sha256": "",
            "target_sha256": sha256(target_bytes),
        })

    manifest = manifest_text(records)
    update(MANIFEST, manifest, 0o644, args.check, changed)
    if changed:
        print("out of date:" if args.check else "generated:", *changed, sep="\n  ")
        return 1 if args.check else 0
    print("up to date: {} source Python files; one shared archive".format(len(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
