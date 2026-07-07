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


class RunResult(BaseModel):
    task: Task
    outputs: Dict[str, StepOutput]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    raw: Any | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None


class ModelClient(Protocol):
    async def generate(
        self,
        instruction: str,
        question: str,
        context: Dict[str, str],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse: ...


@dataclass(frozen=True)
class ModelSpec:
    client: ModelClient

    provider: str = ""
    display_name: str | None = None
    model_idx: int | str | None = None
    name: str | None = None
    context_length: int | None = None
    supports_tools: bool = False
    supports_json: bool = False
    cost_per_1m_input_tokens: float | None = None  # $USD
    cost_per_1m_output_tokens: float | None = None  # $USD
    tags: Set[str] = field(default_factory=set)
