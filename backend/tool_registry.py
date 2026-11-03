"""Single validated, monitored dispatch boundary for every agent tool call."""
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from mcp_tools.contracts import CSVInput, CSVOutput, LLMInput, LLMOutput, QueryInput, QueryOutput
from mcp_tools.mcp_database import query_database
from mcp_tools.mcp_file_parser import parse_csv
from mcp_tools.mcp_groq import call_llm


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable

    @property
    def input_schema(self):
        return self.input_model.model_json_schema()


TOOLS = {
    item.name: item for item in (
        Tool("query_database", "Run a bounded read-only database SELECT", QueryInput, QueryOutput, query_database),
        Tool("call_llm", "Generate guarded text using Gemini", LLMInput, LLMOutput, call_llm),
        Tool("parse_csv", "Validate and summarize an uploaded CSV", CSVInput, CSVOutput, parse_csv),
    )
}


def invoke_tool(name: str, **arguments):
    from guardrails import InputGuard
    from observability import traced
    tool = TOOLS[name]
    inputs = tool.input_model.model_validate(arguments)
    with traced(f"mcp.{name}", input_shape=str(len(str(arguments)))) as span:
        result = tool.handler(**inputs.model_dump())
        if not InputGuard().validate_tool_return(name, result):
            raise ValueError(f"Invalid or unsafe return value from {name}")
        if name == "parse_csv":
            span.set_attribute("input_shape", f"{result['stats']['rows']}x{result['stats']['columns']}")
        return tool.output_model.model_validate(result).model_dump(mode="json", by_alias=True)


def list_tools():
    return [{"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in TOOLS.values()]
