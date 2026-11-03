"""Serialize mutations of a session without blocking the event loop."""
from contextlib import contextmanager
from threading import Lock

_locks = {}
_guard = Lock()


@contextmanager
def session_lock(session_id):
    with _guard:
        lock = _locks.setdefault(session_id, Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError("A pipeline is already running for this session")
    try:
        yield
    finally:
        lock.release()
