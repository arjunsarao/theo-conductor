from __future__ import annotations

from enum import Enum
from pydantic import AliasChoices, BaseModel, Field
from typing import Any, Dict, List, Set, Protocol
from dataclasses import dataclass, field


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Step(BaseModel):
    step_id: str
    model_id: int | str = Field(validation_alias=AliasChoices("model_id", "model_idx"))
    instruction: str
    access_list: List[str] = Field(default_factory=list)
    artifact_inputs: List[str] = Field(default_factory=list)
    needs_tools: bool = False
    depends_on: Set[str] = Field(default_factory=set)


class Task(BaseModel):
    task_type: str
    difficulty: Difficulty
    question: str
    workflow: List[Step]

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        workflow = [Step(**step) for step in data["workflow"]]
        return cls(
            task_type=data["task_type"],
            difficulty=Difficulty(data["difficulty"]),
            question=data["question"],
            workflow=workflow,
        )


class StepOutput(BaseModel):
    step_id: str
    model_id: int | str
    text: str
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None


class RunResult(BaseModel):
    task: Task
    outputs: Dict[str, StepOutput]
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    observed_wall_time_ms: float | None = None
    observed_peak_concurrency: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    raw: Any | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None
    finish_reason: str | None = None


class ModelClient(Protocol):
    async def generate(
        self,
        instruction: str,
        question: str,
        context: Dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class ModelSpec:
    client: ModelClient

    provider: str = ""
    display_name: str | None = None
    model_idx: int | str | None = None
    name: str | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    output_budget_observed_tokens: int | None = None
    supports_tools: bool = False
    supports_json: bool = False
    cost_per_1m_input_tokens: float | None = None  # $USD
    cost_per_1m_output_tokens: float | None = None  # $USD
    tags: Set[str] = field(default_factory=set)
    role: str | None = None
    best_for: str | None = None
    useful_for: str | None = None
    routing_note: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("context_length", "max_output_tokens", "output_budget_observed_tokens"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive")

        for field_name in ("cost_per_1m_input_tokens", "cost_per_1m_output_tokens"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        for field_name in ("role", "best_for", "useful_for", "routing_note"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")
