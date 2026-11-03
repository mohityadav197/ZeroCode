"""Validate the original public Python entry points as well as the new workers."""
from functools import wraps

from guardrails import InputGuard
from schemas import EDAOutput, MLOutput, ReportOutput


def validate_frame(frame):
    validation = InputGuard().validate_csv_schema(frame)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))


def analysis_contract(function):
    @wraps(function)
    def wrapped(self, df, *args, **kwargs):
        validate_frame(df)
        result = EDAOutput.model_validate(function(self, df, *args, **kwargs)).model_dump(mode="json")
        InputGuard().validate_agent_output(result)
        return result
    return wrapped


def ml_contract(function):
    @wraps(function)
    def wrapped(self, df, *args, **kwargs):
        validate_frame(df)
        result = MLOutput.model_validate(function(self, df, *args, **kwargs)).model_dump(mode="json")
        InputGuard().validate_agent_output(result)
        return result
    return wrapped


def report_contract(function):
    @wraps(function)
    def wrapped(self, analysis_result, ml_result):
        EDAOutput.model_validate(analysis_result)
        MLOutput.model_validate(ml_result)
        result = ReportOutput.model_validate(function(self, analysis_result, ml_result)).model_dump(mode="json")
        InputGuard().validate_agent_output(result)
        return result
    return wrapped
