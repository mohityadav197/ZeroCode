# ============================================
# ZeroCode — llm_provider.py
# Groq LLM connection and ask methods.
# ============================================

import json
import time

from groq import Groq

from config import GROQ_API_KEY, MODEL_NAME

REQUEST_TIMEOUT = 30.0
RETRY_WAIT_SECONDS = 2


class LLMProvider:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY, timeout=REQUEST_TIMEOUT)
        self.model = MODEL_NAME
        print("LLMProvider initialized successfully")

    def _complete(self, messages, **kwargs):
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )
        except Exception:
            time.sleep(RETRY_WAIT_SECONDS)
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )

    def ask(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._complete(messages)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: LLM request failed - {e}"

    def ask_json(self, prompt: str, system_prompt: str = None) -> dict:
        json_instruction = "Respond with valid JSON only. Do not include any explanation, markdown, or code fences."
        combined_system_prompt = (
            f"{system_prompt}\n{json_instruction}" if system_prompt else json_instruction
        )

        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._complete(messages, response_format={"type": "json_object"})
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
            return {}


if __name__ == "__main__":
    llm = LLMProvider()
    result = llm.ask("Say hello from ZeroCode in one sentence")
    print(result)
