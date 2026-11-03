"""Worker template: validate -> callback/span -> execute -> validate -> publish."""
import asyncio
from abc import ABC, abstractmethod

from blackboard import Blackboard
from config import MESSAGING_MODE
from schemas import STAGE_INPUTS, STAGE_OUTPUTS


class WorkerNode(ABC):
    stage: str

    def __init__(self, messaging_mode=MESSAGING_MODE):
        if messaging_mode not in ("blackboard", "direct"):
            raise ValueError("Invalid messaging mode")
        self.messaging_mode = messaging_mode

    def predecessor(self, request, stage):
        if self.messaging_mode == "blackboard":
            return Blackboard(request.session_id).read(stage)
        return request.predecessors.get(stage)

    def run(self, input_data):
        from guardrails import InputGuard
        from monitoring import AgentMonitor
        from observability import traced
        from runtime_context import current_run, execution_context
        from time import perf_counter
        import traceback

        request = STAGE_INPUTS[self.stage].model_validate(input_data)
        board = Blackboard(request.session_id)
        if self.messaging_mode == "blackboard":
            # Results are pulled from the shared board, never forwarded in the envelope.
            request = STAGE_INPUTS[self.stage].model_validate(board.get_full_state()["request"])
        monitor = AgentMonitor(request.session_id)
        start = perf_counter()
        with execution_context(request.session_id, None if current_run.get() == "legacy" else current_run.get()):
            with traced(f"{self.stage}_agent.run", input_shape=request.filepath or "") as span:
                try:
                    monitor.on_start(self.stage, request.model_dump())
                    InputGuard().validate_user_text(request.prompt)
                    result = STAGE_OUTPUTS[self.stage].model_validate(self.execute(request))
                    data = result.model_dump(mode="json")
                    profile = data.get("profile") or (board.read("profiler") or {}).get("profile", {})
                    if profile:
                        span.set_attribute("input_shape", f"{profile['rows']}x{profile['columns']}")
                    InputGuard().validate_agent_output(data)
                    board.update(self.stage, data)
                    monitor.on_complete(self.stage, data, (perf_counter() - start) * 1000)
                    return data
                except Exception as error:
                    board.add_error(self.stage, error)
                    monitor.on_error(self.stage, error, traceback.format_exc())
                    raise

    async def run_async(self, input_data):
        # Propagates contextvars to threads and keeps CPU/SDK work off FastAPI's loop.
        return await asyncio.to_thread(self.run, input_data)

    @abstractmethod
    def execute(self, request):
        raise NotImplementedError
