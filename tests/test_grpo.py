import pytest

from theo_conductor.grpo import (
    ConductorParseError,
    compute_reward,
    parse_conductor_json,
)
from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelSpec


VALID_COMPLETION = """
```json
{
  "task_type": "physics",
  "difficulty": "medium",
  "question": "What is 2 + 2?",
  "workflow": [
    {
      "step_id": "final",
      "model_id": "solver",
      "instruction": "Return the answer.",
      "access_list": ["question"]
    }
  ],
  "final_answer": "A"
}
```
"""


def test_parse_conductor_json_accepts_fenced_json():
    task = parse_conductor_json(VALID_COMPLETION)

    assert task.task_type == "physics"
    assert task.workflow[0].step_id == "final"


def test_parse_conductor_json_rejects_invalid_workflow():
    with pytest.raises(ConductorParseError):
        parse_conductor_json('{"workflow": []}')


def test_compute_reward_scores_malformed_invalid_valid_and_correct():
    invalid_workflow = """
    {
      "task_type": "physics",
      "difficulty": "medium",
      "question": "What is 2 + 2?",
      "workflow": [
        {
          "step_id": "solve",
          "model_id": "solver",
          "instruction": "Solve.",
          "access_list": ["question"]
        }
      ]
    }
    """
    valid_without_answer = VALID_COMPLETION.replace('"final_answer": "A"', '"final_answer": "B"')

    assert compute_reward(
        ["not json", invalid_workflow, valid_without_answer, VALID_COMPLETION],
        ground_truth=["A", "A", "A", "A"],
        answer_type=["multipleChoice"] * 4,
    ) == [0.0, 0.2, 0.5, 1.0]


def test_compute_reward_validates_model_ids_when_registry_is_supplied():
    registry = ModelRegistry([ModelSpec(model_idx="other", client=FakeModelClient("other"))])

    assert compute_reward([VALID_COMPLETION], ground_truth=["A"], model_registry=registry) == [0.2]
