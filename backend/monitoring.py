import json
from pathlib import Path

import pandas as pd

from database import SessionLocal
from platform_models import AgentLog
from runtime_context import current_run


def serialized_size(value):
    return len(json.dumps(value, default=str).encode("utf-8"))


class AgentMonitor:
    def __init__(self, session_id):
        self.session_id = session_id

    def _log(self, agent_name, event_type, **values):
        with SessionLocal() as db:
            db.add(AgentLog(session_id=self.session_id, run_id=current_run.get(), agent_name=agent_name,
                            event_type=event_type, **values))
            db.commit()

    def on_start(self, agent_name, input_data):
        self._log(agent_name, "start", input_size=serialized_size(input_data))

    def on_complete(self, agent_name, output_data, duration_ms):
        self._log(agent_name, "complete", output_size=serialized_size(output_data), duration_ms=duration_ms)

    def on_error(self, agent_name, error, traceback):
        self._log(agent_name, "error", error=str(error), traceback=traceback)


def get_agent_logs(session_id):
    with SessionLocal() as db:
        rows = db.query(AgentLog).filter_by(session_id=session_id).order_by(AgentLog.timestamp).all()
        return [{column.name: (getattr(row, column.name).isoformat() if column.name == "timestamp"
                               else getattr(row, column.name))
                 for column in AgentLog.__table__.columns} for row in rows]


def generate_agent_heatmap(session_id):
    from config import UPLOAD_FOLDER
    from plotting import plot_lock
    from schemas import AgentInput
    AgentInput(session_id=session_id)
    records = [row for row in get_agent_logs(session_id) if row["event_type"] == "complete"]
    if not records:
        raise ValueError("No completed agent timings for this session")
    frame = pd.DataFrame(records)
    timings = frame.pivot_table(index="run_id", columns="agent_name", values="duration_ms", aggfunc="sum", sort=False)
    timings = timings.reindex(columns=["profiler", "eda", "preprocessing", "ml", "report"])
    path = Path(UPLOAD_FOLDER) / session_id / "agent_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with plot_lock:
        from matplotlib import pyplot as plt
        import seaborn as sns
        figure, axis = plt.subplots(figsize=(10, max(3, 0.6 * len(timings))))
        sns.heatmap(timings, mask=timings.isna(), cmap="YlOrRd", annot=True, fmt=".0f", ax=axis,
                    cbar_kws={"label": "Duration (ms)"})
        axis.set_yticklabels([f"Run {index + 1} ({str(run_id)[:8]})" for index, run_id in enumerate(timings.index)], rotation=0)
        axis.set(xlabel="Pipeline stage", ylabel="Run", title=f"Agent durations: {session_id}")
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
    return str(path)
