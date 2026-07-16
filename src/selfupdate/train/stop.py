"""Cooperative stop requests for long-running training campaigns.

SIGTERM and SIGINT request a stop at the next trainer-defined safe boundary.
The first signal never raises inside a CUDA operation; callers remain
responsible for polling :func:`stop_requested`, returning through the normal
checkpoint path, and recording whether the final epoch was partial.
"""

from __future__ import annotations

import contextlib
import signal
import threading


_REQUESTED = threading.Event()
_SIGNUM: int | None = None


def stop_requested() -> bool:
    return _REQUESTED.is_set()


def requested_signal() -> int | None:
    return _SIGNUM


@contextlib.contextmanager
def cooperative_stop_signals():
    """Convert the first SIGTERM/SIGINT into a cooperative stop request."""
    global _SIGNUM
    _REQUESTED.clear()
    _SIGNUM = None
    previous = {}

    def request(signum, _frame):
        global _SIGNUM
        _SIGNUM = int(signum)
        _REQUESTED.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
