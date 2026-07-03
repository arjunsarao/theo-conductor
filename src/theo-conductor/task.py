from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import List


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Step:
    step_id: str
    model_id: str
    instruction: str
    access_list: list[str] = field(default_factory=list)
    depends_on: set[str] = field(default_factory=set)



@dataclass
class Task:
    task_type: str
    difficulty: Difficulty
    question: str
    needs_tools: bool
    workflow: List[Step]

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        workflow = [Step(**step) for step in data["workflow"]]
        return cls(
            task_type=data["task_type"],
            difficulty=Difficulty(data["difficulty"]),
            question=data["question"],
            needs_tools=data["needs_tools"],
            workflow=workflow,
        )
