#!/usr/bin/env python3
"""Integrity-checked importer for the shared compressed selfupdate archive."""

import hashlib as _su_hashlib
import importlib.abc as _su_abc
import importlib.util as _su_util
import os as _su_os
import sys as _su_sys
import zipfile as _su_zipfile

_SU_VIRTUAL_ROOT = _su_os.path.dirname(_su_os.path.abspath(__file__))
_SU_ARCHIVE_PATH = _su_os.path.join(_SU_VIRTUAL_ROOT, "_selfupdate_bundle.zip")
_SU_EXPECTED_SHA256 = '3ccfda1323f163ca858302ebd86f9ddce044c769488e4de11fb06bcbf395509e'
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
