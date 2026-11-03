"""Context-local paths keep the legacy chart/model writers safe across sessions."""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from uuid import uuid4

from config import UPLOAD_FOLDER

current_session = ContextVar("zerocode_session", default="legacy")
current_run = ContextVar("zerocode_run", default="legacy")


class OutputPath(os.PathLike):
    def __init__(self, child=""):
        self.child = child

    def __fspath__(self):
        base = Path(UPLOAD_FOLDER).resolve()
        session = current_session.get()
        return str((base if session == "legacy" else base / session) / self.child)


output_root = OutputPath()
charts_root = OutputPath("charts")


@contextmanager
def execution_context(session_id, run_id=None):
    # The session_id has already been validated by AgentInput.
    from schemas import AgentInput
    AgentInput(session_id=session_id)
    sid = current_session.set(session_id)
    rid = current_run.set(run_id or str(uuid4()))
    try:
        yield current_run.get()
    finally:
        current_run.reset(rid)
        current_session.reset(sid)


def safe_output_path(filepath: str) -> Path:
    root = Path(UPLOAD_FOLDER).resolve()
    path = Path(filepath).resolve()
    if not path.is_relative_to(root):
        raise ValueError("File must be inside the outputs directory")
    return path
