"""Compatibility facade: existing agents also call the guarded MCP tool boundary."""
import json

from config import MODEL_NAME


class LLMProvider:
    def __init__(self):
        self.model = MODEL_NAME

    def ask(self, prompt: str, system_prompt: str = None) -> str:
        from tool_registry import invoke_tool
        return invoke_tool("call_llm", prompt=prompt, system=system_prompt or "")

    def ask_json(self, prompt: str, system_prompt: str = None) -> dict:
        instruction = (system_prompt or "") + "\nRespond with valid JSON only, without markdown or code fences."
        content = self.ask(prompt, instruction)
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
