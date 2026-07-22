import pytest

from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import Difficulty, ModelSpec, Step, Task
from theo_conductor.validate import validate_task


def make_task(workflow: list[Step] | None = None, **overrides) -> Task:
    data = {
        "task_type": "physics",
        "difficulty": Difficulty.MEDIUM,
        "question": "What is 2 + 2?",
        "workflow": workflow
        if workflow is not None
        else [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve the problem.",
                access_list=["question"],
            ),
            Step(
                step_id="final",
                model_id="synthesizer",
                instruction="Write the final answer.",
                access_list=["question", "solve"],
            ),
        ],
    }
    data.update(overrides)
    return Task(**data)


@pytest.fixture
def model_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelSpec(
                model_idx="solver",
                client=FakeModelClient("solver"),
            ),
            ModelSpec(
                model_idx="synthesizer",
                client=FakeModelClient("synthesizer"),
            ),
            ModelSpec(
                model_idx="tool_solver",
                client=FakeModelClient("tool_solver"),
                supports_tools=True,
            ),
        ]
    )


def test_validate_task_accepts_valid_workflow(model_registry):
    validate_task(make_task(), model_registry)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"task_type": "   "}, "task_type must be a non-empty string"),
        ({"question": "   "}, "question must be a non-empty string"),
        ({"workflow": []}, "workflow must contain at least one step"),
    ],
)
def test_validate_task_rejects_invalid_top_level_fields(overrides, match):
    task = make_task(**overrides)

    with pytest.raises(ValueError, match=match):
        validate_task(task)


def test_validate_task_rejects_blank_step_id():
    task = make_task(
        [
            Step(
                step_id=" ",
                model_id="solver",
                instruction="Solve.",
                access_list=["question"],
            )
        ]
    )

    with pytest.raises(ValueError, match=r"workflow\[0\]\.step_id must be non-empty"):
        validate_task(task)


def test_validate_task_rejects_duplicate_step_id():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve once.",
                access_list=["question"],
            ),
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve twice.",
                access_list=["question"],
            ),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate step_id found: 'solve'"):
        validate_task(task)


def test_validate_task_rejects_blank_instruction():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction=" ",
                access_list=["question"],
            )
        ]
    )

    with pytest.raises(ValueError, match="Step 'solve' instruction must be non-empty"):
        validate_task(task)


def test_validate_task_rejects_self_dependency():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve.",
                access_list=["question", "solve"],
            )
        ]
    )

    with pytest.raises(ValueError, match="Step 'solve' cannot depend on itself"):
        validate_task(task)


def test_validate_task_rejects_unknown_or_future_access_key():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve.",
                access_list=["question", "final"],
            ),
            Step(
                step_id="final",
                model_id="synthesizer",
                instruction="Write final.",
                access_list=["question", "solve"],
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Step 'solve' accesses unknown or future key 'final'",
    ):
        validate_task(task)


def test_validate_task_accepts_any_name_for_last_step():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve.",
                access_list=["question"],
            )
        ]
    )

    validate_task(task)


def test_validate_task_does_not_require_question_in_final_access_list():
    task = make_task(
        [
            Step(
                step_id="answer",
                model_id="solver",
                instruction="Write final.",
                access_list=[],
            )
        ]
    )

    validate_task(task)


def test_validate_task_allows_final_to_select_only_needed_previous_steps():
    task = make_task(
        [
            Step(
                step_id="solve",
                model_id="solver",
                instruction="Solve.",
                access_list=["question"],
            ),
            Step(
                step_id="final",
                model_id="synthesizer",
                instruction="Write final.",
                access_list=["question"],
            ),
        ]
    )

    validate_task(task)


def test_validate_task_checks_model_ids_when_registry_is_provided(model_registry):
    task = make_task(
        [
            Step(
                step_id="final",
                model_id="unknown",
                instruction="Write final.",
                access_list=["question"],
            )
        ]
    )

    with pytest.raises(ValueError, match="Model 'unknown' not found"):
        validate_task(task, model_registry)


def test_validate_task_rejects_tools_when_model_does_not_support_tools(model_registry):
    task = make_task(
        [
            Step(
                step_id="final",
                model_id="solver",
                instruction="Write final.",
                access_list=["question"],
                needs_tools=True,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Step 'final' needs tools, but model 'solver' does not support tools",
    ):
        validate_task(task, model_registry)


def test_validate_task_accepts_tools_when_model_supports_tools(model_registry):
    task = make_task(
        [
            Step(
                step_id="final",
                model_id="tool_solver",
                instruction="Write final.",
                access_list=["question"],
                needs_tools=True,
            )
        ]
    )

    validate_task(task, model_registry)


def test_validate_task_rejects_duplicate_artifact_inputs(model_registry):
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="final",
                model_id="solver",
                instruction="Answer.",
                access_list=["question"],
                artifact_inputs=["results", "results"],
            )
        ],
    )

    with pytest.raises(ValueError, match="duplicate artifact inputs"):
        validate_task(task, model_registry)
