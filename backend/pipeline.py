"""Pipeline patterns and orchestration, without taking ownership of worker results."""
import asyncio
from pathlib import Path

from a2a_bridge import A2AMessage, bridge
from blackboard import Blackboard
from config import MESSAGING_MODE, PIPELINE_MODE
from schemas import AgentInput
from session_locks import session_lock

STAGES = ("profiler", "eda", "preprocessing", "ml", "report")


class PipelineOrchestrator:
    def __init__(self, pipeline_mode=PIPELINE_MODE, messaging_mode=MESSAGING_MODE):
        if pipeline_mode not in ("sequential", "parallel", "hierarchical"):
            raise ValueError("Invalid pipeline mode")
        if messaging_mode not in ("blackboard", "direct"):
            raise ValueError("Invalid messaging mode")
        self.pipeline_mode = pipeline_mode
        self.messaging_mode = messaging_mode

    async def run(self, request: AgentInput, stages=STAGES):
        from observability import traced
        from runtime_context import execution_context
        from guardrails import InputGuard
        try:
            InputGuard().validate_user_text(request.prompt)
        except ValueError as error:
            with session_lock(request.session_id):
                with execution_context(request.session_id), traced("pipeline.guardrail"):
                    blocked = Blackboard(request.session_id, request)
                    blocked.transition("blocked", pattern=self.pipeline_mode, messaging_mode=self.messaging_mode)
                    blocked.add_error("guardrail", error)
            raise
        with session_lock(request.session_id):
            with execution_context(request.session_id) as run_id, traced("pipeline.run"):
                board = Blackboard(request.session_id, request)
                previous = board.get_full_state()["request"]
                if previous.get("filepath") and (not request.filepath or Path(request.filepath).resolve() != Path(previous["filepath"]).resolve()):
                    raise ValueError("A session cannot switch datasets; upload a new dataset")
                board.configure(request)
                board.clear_results(stages)
                board.transition("starting", pattern=self.pipeline_mode, messaging_mode=self.messaging_mode, run_id=run_id)
                direct_results = {}
                if self.messaging_mode == "direct":
                    for stage in STAGES:
                        value = board.read(stage)
                        if value is not None:
                            direct_results[stage] = value

                async def trigger(stage):
                    board.transition(stage, pattern=self.pipeline_mode, run_id=run_id,
                                     supervisor="orchestrator" if self.pipeline_mode == "hierarchical" else None)
                    payload = ({"session_id": request.session_id} if self.messaging_mode == "blackboard" else
                               {**request.model_dump(), "predecessors": dict(direct_results)})
                    message = A2AMessage(sender="orchestrator", receiver=stage, payload=payload, session_id=request.session_id)
                    result = await bridge.send(message, self.messaging_mode)
                    if self.messaging_mode == "direct":
                        direct_results[stage] = result
                    return result

                try:
                    remaining = list(stages)
                    if self.pipeline_mode == "parallel" and "profiler" in remaining and "eda" in remaining:
                        # EDA computes its own basic statistics and never waits for profiling.
                        outcomes = await asyncio.gather(trigger("profiler"), trigger("eda"), return_exceptions=True)
                        # Wait for both before surfacing failures, so workers cannot write into a later run.
                        for outcome in outcomes:
                            if isinstance(outcome, BaseException):
                                raise outcome
                        remaining = [stage for stage in remaining if stage not in ("profiler", "eda")]
                    for stage in remaining:
                        # Explicit supervisor -> WorkerNode routing in hierarchical mode.
                        await trigger(stage)
                    board.transition("completed", pattern=self.pipeline_mode, run_id=run_id)
                    return board.get_full_state()
                except Exception as error:
                    board.transition("failed", pattern=self.pipeline_mode, run_id=run_id)
                    board.add_error("orchestrator", error)
                    raise
