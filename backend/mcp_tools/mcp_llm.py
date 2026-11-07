import logging
import time

from openai import APIError, OpenAI

from config import GEMINI_API_KEY, MODEL_NAME
from runtime_context import current_session

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 90
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
RETRYABLE_STATUS_CODES = (429, 503)


class LLMRequestError(RuntimeError):
    """Provider failure with its original cause preserved for logs and reports."""


def call_llm(prompt: str, system: str = "") -> str:
    """Ask the configured Gemini model; return guarded text and count actual usage."""
    from blackboard import Blackboard
    from guardrails import InputGuard
    if not GEMINI_API_KEY:
        raise LLMRequestError("GEMINI_API_KEY must be configured for live LLM calls")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    # SDK retries multiply the timeout and leave interactive chat waiting.
    with OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL,
                timeout=LLM_TIMEOUT_SECONDS, max_retries=0) as client:
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, temperature=0.2, max_tokens=1800,
                )
                break
            except (APIError, TimeoutError) as error:
                status = getattr(error, "status_code", None)
                if status not in RETRYABLE_STATUS_CODES:
                    raise LLMRequestError(str(error)) from error
                if attempt > MAX_RETRIES:
                    raise
                logger.warning("Gemini returned %s, retrying (attempt %s/%s)...", status, attempt, MAX_RETRIES)
                time.sleep(RETRY_DELAY_SECONDS)
    if response.usage and current_session.get() != "legacy":
        Blackboard(current_session.get()).add_tokens(response.usage.total_tokens, response.usage.prompt_tokens or 0)
    content = response.choices[0].message.content if response.choices else ""
    if not content or not content.strip():
        raise LLMRequestError("Gemini returned an empty response")
    InputGuard().validate_llm_response(content)
    return content
