from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryInput(ToolInput):
    sql: str = Field(min_length=1, max_length=20000)


class QueryOutput(ToolInput):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int = Field(ge=0)


class LLMInput(ToolInput):
    prompt: str = Field(min_length=1, max_length=150000)
    system: str = ""


class LLMOutput(RootModel[str]):
    root: str = Field(min_length=1)


class CSVInput(ToolInput):
    filepath: str = Field(min_length=1)


class CSVOutput(ToolInput):
    filepath: str
    column_schema: dict[str, str] = Field(alias="schema")
    preview: list[dict[str, Any]]
    stats: dict[str, Any]
