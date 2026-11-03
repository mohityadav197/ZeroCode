from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from a2a_bridge import router as a2a_router
from api_service import execute_stages, make_request, persist_state
from blackboard import Blackboard
from monitoring import generate_agent_heatmap, get_agent_logs
from observability import trace_link
from schemas import AgentInput
from tool_registry import list_tools

router = APIRouter()
router.include_router(a2a_router)


class PipelineRequest(AgentInput):
    filename: str
    pipeline_mode: Literal["sequential", "parallel", "hierarchical"] | None = None
    messaging_mode: Literal["blackboard", "direct"] | None = None


@router.post("/pipeline/run")
def run_pipeline(payload: PipelineRequest):
    request = make_request(payload.filename, **payload.model_dump(exclude={"filename", "filepath", "pipeline_mode", "messaging_mode"}))
    state = execute_stages(request, ("profiler", "eda", "preprocessing", "ml", "report"),
                           payload.pipeline_mode, payload.messaging_mode)
    persist_state(state)
    return state


class StageRequest(PipelineRequest):
    pass


def run_stage(stage, payload):
    request = make_request(payload.filename, **payload.model_dump(exclude={"filename", "filepath", "pipeline_mode", "messaging_mode"}))
    board = Blackboard(request.session_id)
    prerequisites = {"profiler": (), "eda": ("profiler",), "preprocessing": ("profiler", "eda")}[stage]
    missing = [item for item in prerequisites if not board.read(item)]
    state = execute_stages(request, (*missing, stage), payload.pipeline_mode, payload.messaging_mode)
    persist_state(state)
    return state[f"{stage}_result"]


@router.post("/profile")
def profile(payload: StageRequest):
    return run_stage("profiler", payload)


@router.post("/eda")
def eda(payload: StageRequest):
    return run_stage("eda", payload)


@router.post("/preprocess")
def preprocess(payload: StageRequest):
    return run_stage("preprocessing", payload)


@router.get("/blackboard/{session_id}")
def get_blackboard(session_id: str):
    AgentInput(session_id=session_id)
    return Blackboard(session_id).get_full_state()


@router.get("/tools")
def tools():
    return list_tools()


@router.get("/monitoring/logs/{session_id}")
def logs(session_id: str):
    return get_agent_logs(session_id)


@router.get("/monitoring/heatmap/{session_id}")
def heatmap(session_id: str):
    try:
        path = generate_agent_heatmap(session_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error
    return FileResponse(path, media_type="image/png")


@router.get("/observability/trace/{session_id}")
def session_trace(session_id: str):
    AgentInput(session_id=session_id)
    return trace_link(session_id)
