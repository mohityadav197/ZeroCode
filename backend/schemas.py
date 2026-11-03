"""Serializable contracts shared by the HTTP, A2A and agent boundaries."""
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentInput(Contract):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w-]+$")
    filepath: str | None = None
    target_col: str | None = None
    problem_type: Literal["binary_classification", "multiclass_classification", "regression"] | None = None
    prompt: str = "Analyze this dataset"
    extra_models: list[str] | None = None
    selected_metrics: list[str] | None = None
    user_preprocessing_choices: dict[str, Any] | None = None
    predecessors: dict[str, Any] = Field(default_factory=dict)

    @field_validator("predecessors")
    @classmethod
    def validate_predecessors(cls, values):
        for stage, output in values.items():
            if stage not in STAGE_OUTPUTS:
                raise ValueError(f"Unknown predecessor: {stage}")
            STAGE_OUTPUTS[stage].model_validate(output)
        return values


class ProfilerInput(AgentInput):
    pass


class EDAInput(AgentInput):
    pass


class PreprocessingInput(AgentInput):
    pass


class MLInput(AgentInput):
    pass


class ReportInput(AgentInput):
    pass


class Profile(Contract):
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    column_types: dict[str, str]
    missing: dict[str, float]
    unique_counts: dict[str, int]
    constant_columns: list[str]
    id_like_columns: list[str]
    quality_score: int = Field(ge=0, le=100)


class ProfilerOutput(Contract):
    profile: Profile
    domain_info: dict[str, Any]
    target_suggestion: dict[str, Any]
    problem_type: dict[str, Any]
    summary: str = ""
    status: Literal["completed"] = "completed"


class EDAOutput(Contract):
    # EDA can profile independently when run alongside the profiler.
    profile: Profile
    domain_info: dict[str, Any]
    target_suggestion: dict[str, Any]
    problem_type: dict[str, Any]
    charts: dict[str, Any]
    eda_code_path: str
    outliers: dict[str, Any]
    target_balance: dict[str, Any]
    insights: list[str]
    model_recommendations: dict[str, Any] | None = None
    status: Literal["completed"] = "completed"


class PreprocessingOutput(Contract):
    artifact_path: str
    preprocessed_data_path: str
    preprocessing_code_path: str
    preprocessing_log: list[str]
    feature_names: list[str]
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    summary: str = ""
    status: Literal["completed"] = "completed"


class MLOutput(BaseModel):
    # Preserve optional metrics and exported artifacts from the existing ML agent.
    model_config = ConfigDict(extra="allow")
    best_model_name: str | None
    leaderboard: list[dict[str, Any]]
    preprocessing_log: list[str]
    best_params: dict[str, Any]
    status: Literal["completed", "failed"]


class ReportOutput(Contract):
    report_path: str
    verification: dict[str, Any]
    recommendations: list[str]
    status: Literal["completed"] = "completed"


class BlackboardState(Contract):
    session_id: str
    request: AgentInput
    profiler_result: ProfilerOutput | None = None
    eda_result: EDAOutput | None = None
    preprocessing_result: PreprocessingOutput | None = None
    ml_result: MLOutput | None = None
    report_result: ReportOutput | None = None
    current_stage: str = "uploaded"
    errors: list[dict[str, Any]] = Field(default_factory=list)
    timestamps: dict[str, str] = Field(default_factory=dict)
    logs: list[dict[str, Any]] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    input_tokens: int = 0


STAGE_INPUTS = dict(zip(
    ("profiler", "eda", "preprocessing", "ml", "report"),
    (ProfilerInput, EDAInput, PreprocessingInput, MLInput, ReportInput),
))
STAGE_OUTPUTS = dict(zip(STAGE_INPUTS, (ProfilerOutput, EDAOutput, PreprocessingOutput, MLOutput, ReportOutput)))
