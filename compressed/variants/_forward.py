"""Small launcher shared by the purpose-specific defactorised entry points."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable, Sequence


def launch(script: str, fixed: Sequence[str], forbidden: Iterable[str]) -> None:
    """Replace this process with a defactorised script under a fixed mode."""
    argv = sys.argv[1:]
    blocked = tuple(forbidden)
    conflicts = [
        value
        for value in argv
        if any(value == option or value.startswith(option + "=") for option in blocked)
    ]
    if conflicts:
        names = ", ".join(conflicts)
        raise SystemExit(f"this purpose-specific launcher fixes its mode; remove: {names}")

    target = Path(__file__).resolve().parent.parent / script
    if not target.is_file():
        raise SystemExit(f"defactorised target is missing: {target}")
    os.execv(sys.executable, [sys.executable, str(target), *fixed, *argv])
