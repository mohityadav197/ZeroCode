"""Local guardrails: pattern detection is an academic demo layer, not a security model."""
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class GuardrailViolation(ValueError):
    pass


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InputGuard:
    injection_patterns = (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\b",
        r"\bsystem\s*:", r"\bjailbreak\b", r"\bdisregard\s+(?:all\s+)?instructions\b",
        r"<\|(?:im_start|system)\|>", r"\breveal\s+(?:the\s+)?(?:system prompt|api key|secrets?)\b",
    )
    toxic_pattern = re.compile(r"\b(?:damn|stupid|idiot|shit|fuck|fucking|crap)\b", re.I)

    def check_prompt_injection(self, text: str) -> bool:
        """True means suspicious instruction text was detected."""
        return any(re.search(pattern, str(text), re.I) for pattern in self.injection_patterns)

    def check_toxic_language(self, text: str) -> bool:
        """True means the caller should warn the user."""
        return bool(self.toxic_pattern.search(text))

    def validate_user_text(self, text: str) -> ValidationResult:
        if self.check_prompt_injection(text):
            raise GuardrailViolation("Prompt injection blocked: remove instructions that override the system")
        warnings = ["Harsh language detected; please keep the conversation constructive."] if self.check_toxic_language(text) else []
        return ValidationResult(valid=True, warnings=warnings)

    def validate_csv_schema(self, df) -> ValidationResult:
        errors = []
        if not isinstance(df, pd.DataFrame) or df.empty:
            return ValidationResult(valid=False, errors=["CSV must contain at least one row"])
        if df.select_dtypes(include="number").empty:
            errors.append("CSV must contain numeric columns")
        null_columns = df.columns[df.isna().all()].tolist()
        if null_columns:
            errors.append(f"All-null columns are not allowed: {null_columns}")
        if any(not str(column).strip() or str(column).startswith("Unnamed:") for column in df.columns):
            errors.append("CSV headers must be nonempty")
        for text in list(map(str, df.columns)) + df.select_dtypes(include=["object", "str", "string"]).stack().astype(str).tolist():
            if self.check_prompt_injection(text):
                errors.append("Prompt injection detected in CSV text")
                break
        return ValidationResult(valid=not errors, errors=errors)

    def validate_llm_response(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise GuardrailViolation("Empty LLM response")
        if self.check_prompt_injection(text) or self.check_toxic_language(text):
            raise GuardrailViolation("Unsafe LLM response blocked")

    def validate_agent_output(self, value: Any):
        if isinstance(value, str) and self.check_prompt_injection(value):
            raise GuardrailViolation("Unsafe agent output blocked")
        if isinstance(value, dict):
            for item in value.values():
                self.validate_agent_output(item)
        elif isinstance(value, list):
            for item in value:
                self.validate_agent_output(item)

    def validate_tool_return(self, tool_name: str, result) -> bool:
        from tool_registry import TOOLS
        try:
            TOOLS[tool_name].output_model.model_validate(result)
            if tool_name == "call_llm":
                self.validate_llm_response(result)
            else:
                self.validate_agent_output(result)
            return True
        except (KeyError, ValidationError, GuardrailViolation):
            return False
