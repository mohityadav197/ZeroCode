import pytest
from pydantic import ValidationError

from schemas import AgentInput, BlackboardState, Profile, STAGE_INPUTS, STAGE_OUTPUTS


def test_all_stage_contracts_and_isolated_state_defaults():
    assert len(STAGE_INPUTS) == len(STAGE_OUTPUTS) == 5
    first = BlackboardState(session_id="a", request=AgentInput(session_id="a"))
    second = BlackboardState(session_id="b", request=AgentInput(session_id="b"))
    first.errors.append({"error": "test"})
    assert second.errors == []
    with pytest.raises(ValidationError):
        AgentInput(session_id="../escape")
    with pytest.raises(ValidationError):
        AgentInput(session_id="a", predecessors={"profiler": {"profile": "not a profile"}})
    with pytest.raises(ValidationError):
        Profile(rows=-1, columns=1, column_types={}, missing={}, unique_counts={},
                constant_columns=[], id_like_columns=[], quality_score=101)
