"""L40S host-ABI guard for optional compiled Python extensions.

The shared cu126 base venv contains a causal-conv1d wheel linked against
glibc 2.32, newer than the L40S node. Transformers probes availability with
``importlib.util.find_spec`` before choosing its documented torch fallback.
Hide only that optional package when the L40S wrapper explicitly requests it;
ordinary Python environments are unaffected.
"""

from __future__ import annotations

import importlib.util
import os


if os.environ.get("SELFUPDATE_DISABLE_CAUSAL_CONV1D") == "1":
    _find_spec = importlib.util.find_spec

    def _l40s_find_spec(name, package=None):
        if name == "causal_conv1d" or name.startswith("causal_conv1d."):
            return None
        return _find_spec(name, package)

    importlib.util.find_spec = _l40s_find_spec
