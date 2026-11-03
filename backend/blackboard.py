"""A persistent, validated blackboard. Workers own only their result sections."""
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock

from database import SessionLocal
from platform_models import BlackboardRecord
from schemas import AgentInput, BlackboardState, STAGE_OUTPUTS, utc_now


class Blackboard:
    # A lock also covers SQLite tests; SELECT FOR UPDATE serializes PostgreSQL writers.
    _lock = RLock()

    def __init__(self, session_id: str, request: AgentInput | None = None):
        self.session_id = session_id
        with self._lock, SessionLocal() as db:
            if db.get(BlackboardRecord, session_id) is None:
                state = BlackboardState(session_id=session_id, request=request or AgentInput(session_id=session_id))
                db.add(BlackboardRecord(session_id=session_id, state=state.model_dump(mode="json")))
                db.commit()

    def _mutate(self, change):
        with self._lock, SessionLocal() as db:
            row = db.query(BlackboardRecord).filter_by(session_id=self.session_id).with_for_update().one()
            state = BlackboardState.model_validate(row.state)
            change(state)
            row.state = state.model_dump(mode="json")
            row.updated_at = datetime.now(timezone.utc)
            db.commit()

    def update(self, stage: str, data: dict):
        if stage not in STAGE_OUTPUTS:
            raise ValueError(f"Unknown stage: {stage}")
        value = STAGE_OUTPUTS[stage].model_validate(data)
        from guardrails import InputGuard
        InputGuard().validate_agent_output(value.model_dump(mode="json"))

        def write(state):
            setattr(state, f"{stage}_result", value)
            state.timestamps[stage] = utc_now()
        self._mutate(write)

    def transition(self, stage: str, **details):
        def write(state):
            state.current_stage = stage
            state.logs.append({"stage": stage, "timestamp": utc_now(), **details})
        self._mutate(write)

    def configure(self, request: AgentInput):
        if request.session_id != self.session_id:
            raise ValueError("Session mismatch")
        self._mutate(lambda state: setattr(state, "request", request))

    def clear_results(self, stages):
        earliest = min(list(STAGE_OUTPUTS).index(stage) for stage in stages)
        def clear(state):
            for stage in list(STAGE_OUTPUTS)[earliest:]:
                setattr(state, f"{stage}_result", None)
        self._mutate(clear)

    def add_error(self, stage, error):
        self._mutate(lambda state: state.errors.append({"stage": stage, "error": str(error), "timestamp": utc_now()}))

    def add_tokens(self, count, input_tokens=0):
        def write(state):
            state.tokens_used += count
            state.input_tokens += input_tokens
        self._mutate(write)

    def add_trace(self, trace_id):
        def write(state):
            if trace_id not in state.trace_ids:
                state.trace_ids.append(trace_id)
        self._mutate(write)

    def read(self, stage: str):
        if stage not in STAGE_OUTPUTS:
            raise ValueError(f"Unknown stage: {stage}")
        return self.get_full_state()[f"{stage}_result"]

    def get_full_state(self):
        with SessionLocal() as db:
            row = db.get(BlackboardRecord, self.session_id)
            return deepcopy(BlackboardState.model_validate(row.state).model_dump(mode="json"))
