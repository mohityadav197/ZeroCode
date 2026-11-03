"""Typed A2A bridge shared by HTTP endpoints and in-process orchestration.

The /a2a/{worker} routes use ZeroCode's requested envelope. /a2a additionally
provides the A2A 0.3 message/send JSON-RPC binding and agent discovery.
"""
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field, ValidationError, field_validator

from schemas import Contract, STAGE_INPUTS, utc_now
from session_locks import session_lock

router = APIRouter()


class A2AMessage(Contract):
    sender: str = Field(min_length=1)
    receiver: Literal["profiler", "preprocessing", "eda", "report", "ml"]
    payload: dict[str, Any]
    timestamp: str = Field(default_factory=utc_now)
    session_id: str
    message_type: Literal["task.run"] = "task.run"

    @field_validator("timestamp")
    @classmethod
    def valid_timestamp(cls, value):
        if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("Timestamp must include timezone")
        return value


def worker_for(stage, messaging_mode):
    from agents.adk.profiler_agent import ProfilerAgent
    from agents.adk.preprocessing_agent import PreprocessingAgent
    from agents.crewai.eda_crew import EDACrew
    from agents.crewai.report_crew import ReportCrew
    from agents.ml_worker import MLWorker
    return {"profiler": ProfilerAgent, "preprocessing": PreprocessingAgent, "eda": EDACrew,
            "report": ReportCrew, "ml": MLWorker}[stage](messaging_mode)


class A2ABridge:
    async def send(self, message: A2AMessage, messaging_mode="blackboard"):
        from blackboard import Blackboard
        inputs = STAGE_INPUTS[message.receiver].model_validate({"session_id": message.session_id, **message.payload})
        if inputs.session_id != message.session_id:
            raise ValueError("A2A envelope and payload session IDs must match")
        board = Blackboard(message.session_id)
        if messaging_mode == "blackboard":
            if message.payload.keys() - {"session_id"}:
                raise ValueError("Blackboard messages carry only the session ID; agents pull their inputs")
            if not board.get_full_state()["request"]["filepath"]:
                raise ValueError("Configure the blackboard dataset before dispatch")
        return await worker_for(message.receiver, messaging_mode).run_async(inputs.model_dump())


bridge = A2ABridge()


@router.post("/a2a/{stage}")
async def dispatch(stage: str, message: A2AMessage, messaging_mode: Literal["blackboard", "direct"] = "blackboard"):
    if stage != message.receiver:
        raise HTTPException(400, "Route and receiver must match")
    try:
        with session_lock(message.session_id):
            result = await bridge.send(message, messaging_mode)
        return {"sender": stage, "receiver": message.sender, "payload": result, "timestamp": utc_now(),
                "session_id": message.session_id, "message_type": "task.completed"}
    except (ValueError, ValidationError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    return {"protocolVersion": "0.3.0", "name": "ZeroCode", "description": "AutoML worker bridge",
            "url": str(request.base_url).rstrip("/") + "/a2a", "version": "1.0.0",
            "preferredTransport": "JSONRPC", "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["application/json"], "defaultOutputModes": ["application/json"],
            "skills": [{"id": stage, "name": stage, "description": f"Run the {stage} worker",
                        "tags": ["automl"]} for stage in STAGE_INPUTS]}


@router.post("/a2a")
async def jsonrpc_message(body: dict):
    rpc_id = body.get("id")
    if body.get("jsonrpc") != "2.0" or body.get("method") != "message/send":
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "Supported method: message/send"}}
    try:
        message = body["params"]["message"]
        if message.get("role") != "user" or not message.get("messageId"):
            raise ValueError("A2A user message with messageId required")
        metadata = message.get("metadata", {})
        parts = [part["data"] for part in message["parts"] if part.get("kind") == "data"]
        if len(parts) != 1:
            raise ValueError("Exactly one data part containing agent inputs is required")
        payload = parts[0]
        sid = message.get("contextId") or payload.get("session_id")
        envelope = A2AMessage(sender="a2a-client", receiver=metadata["receiver"], payload=payload, session_id=sid)
        mode = metadata.get("messaging_mode", "blackboard")
        with session_lock(sid):
            result = await bridge.send(envelope, mode)
        task = {"kind": "task", "id": str(uuid4()), "contextId": sid,
                "status": {"state": "completed", "timestamp": utc_now()},
                "artifacts": [{"artifactId": str(uuid4()), "parts": [{"kind": "data", "data": result}]}]}
        return {"jsonrpc": "2.0", "id": rpc_id, "result": task}
    except (KeyError, TypeError, ValueError) as error:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": str(error)}}
    except Exception:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32603, "message": "Worker execution failed; inspect monitoring logs"}}
