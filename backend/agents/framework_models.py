"""Real ADK/CrewAI model adapters routed through the same guarded Gemini tool."""
import asyncio
import json
from uuid import uuid4

from config import MODEL_NAME
from crewai import BaseLLM
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from tool_registry import invoke_tool


class RegistryCrewLLM(BaseLLM):
    def __init__(self):
        super().__init__(model=MODEL_NAME, temperature=0.2)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
        return invoke_tool("call_llm", prompt=prompt, system=system)

    def supports_function_calling(self):
        return False

    def get_context_window_size(self):
        return 32000


class RegistryADKModel(BaseLlm):
    model: str = MODEL_NAME

    async def generate_content_async(self, llm_request, stream=False):
        prompt = "\n".join(
            part.text for content in llm_request.contents for part in content.parts or [] if part.text
        )
        instruction = llm_request.config.system_instruction if llm_request.config else ""
        if not isinstance(instruction, str):
            instruction = str(instruction or "")
        result = await asyncio.to_thread(invoke_tool, "call_llm", prompt=prompt, system=instruction)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=result)]))


def query_database(sql: str) -> dict:
    """Query PostgreSQL using the validated database tool."""
    return invoke_tool("query_database", sql=sql)


def parse_csv(filepath: str) -> dict:
    """Validate and describe an uploaded CSV using the file parser tool."""
    return invoke_tool("parse_csv", filepath=filepath)


def run_adk(name, instruction, facts, tools):
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    async def execute():
        service = InMemorySessionService()
        session = await service.create_session(app_name="zerocode", user_id="pipeline", session_id=str(uuid4()))
        agent = Agent(name=name, model=RegistryADKModel(), instruction=instruction, tools=tools)
        runner = Runner(agent=agent, app_name="zerocode", session_service=service)
        text = ""
        async for event in runner.run_async(
            user_id="pipeline", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=json.dumps(facts, default=str))]),
        ):
            if event.is_final_response() and event.content:
                text = "\n".join(part.text for part in event.content.parts or [] if part.text)
        if not text:
            raise ValueError(f"ADK agent {name} returned no final response")
        return text
    return asyncio.run(execute())
