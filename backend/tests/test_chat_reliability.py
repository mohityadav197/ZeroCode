import time

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

import app as app_module
from mcp_tools import mcp_llm


QUESTION = "what are n_estimators in XGBoost"


@pytest.fixture
def llm_transport(monkeypatch):
    """Exercise the real SDK and tool boundary without contacting the provider."""
    requests = []

    def install(handler):
        def create_client(**options):
            assert options["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
            assert options["timeout"] == 30
            assert options["max_retries"] == 0

            def respond(request):
                requests.append(request)
                return handler(request)

            return OpenAI(**options, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

        monkeypatch.setattr(mcp_llm, "GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(mcp_llm, "OpenAI", create_client)
        monkeypatch.setattr(mcp_llm, "RETRY_DELAY_SECONDS", 0)
        return requests

    return install


@pytest.mark.parametrize("failure, cause, request_count", [
    ("timeout", "timed out", 1), ("rate_limit", "rate_limit", 4),
    ("model_not_found", "model_not_found", 1), ("empty", "empty response", 1),
])
def test_chat_provider_failures_are_visible_and_retried_only_when_transient(llm_transport, failure, cause, request_count):
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("provider stalled", request=request)
        if failure == "empty":
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        code = 429 if failure == "rate_limit" else 404
        return httpx.Response(code, json={"error": {"message": failure}})

    requests = llm_transport(respond)
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"message": QUESTION})
    assert response.status_code == 500
    assert response.json()["error"] is True
    assert cause in response.json()["message"].lower()
    assert len(requests) == request_count


def test_chat_question_returns_an_answer_within_ten_seconds(llm_transport):
    answer = "n_estimators is the number of boosting rounds (trees) in XGBoost."
    llm_transport(lambda _: httpx.Response(200, json={"choices": [{"message": {"content": answer}}]}))
    with TestClient(app_module.app) as client:
        start = time.perf_counter()
        response = client.post("/chat", json={"message": QUESTION})
        elapsed = time.perf_counter() - start
    assert response.status_code == 200, response.text
    assert response.json()["response"] == answer
    assert elapsed < 10


def test_chat_route_deadline_bounds_a_stuck_worker(monkeypatch):
    def stalled_chat(*_):
        time.sleep(0.2)
        return "late response"

    monkeypatch.setattr(app_module.orchestrator, "chat", stalled_chat)
    monkeypatch.setattr(app_module, "LLM_TIMEOUT_SECONDS", 0.01)
    with TestClient(app_module.app) as client:
        start = time.perf_counter()
        response = client.post("/chat", json={"message": QUESTION})
        elapsed = time.perf_counter() - start
    assert response.status_code == 500
    assert response.json() == {"error": True, "message": "The LLM did not respond within 0.01 seconds"}
    assert elapsed < 0.15


def test_chat_missing_key_is_a_visible_provider_error(monkeypatch):
    monkeypatch.setattr(mcp_llm, "GEMINI_API_KEY", "")
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"message": QUESTION})
    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["message"]


def test_chat_unexpected_failure_reaches_the_browser(monkeypatch):
    def broken_chat(*_):
        raise KeyError("boom")

    monkeypatch.setattr(app_module.orchestrator, "chat", broken_chat)
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"message": QUESTION}, headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 500
    assert response.json()["error"] is True and "boom" in response.json()["message"]
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
