"""Explicit offline mode for integration checks; never presented as live LLM timings."""
import json
from contextlib import contextmanager


@contextmanager
def offline_llm():
    from tool_registry import TOOLS
    original = TOOLS["call_llm"].handler

    def respond(prompt, system=""):
        if "JSON" in system:
            return json.dumps({"explanation": "Offline fixture response: inspect computed model statistics.",
                               "recommendations": ["Offline fixture: review missingness and class balance."]})
        return "Final Answer: Offline fixture response. Inspect the computed missing values, class balance and model metrics."

    TOOLS["call_llm"].handler = respond
    try:
        yield
    finally:
        TOOLS["call_llm"].handler = original
