import asyncio
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from theo_conductor.schema import Task, Step, Difficulty, ModelSpec
from theo_conductor.runner import Runner
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.models.fake import FakeModelClient

@pytest.fixture
def fake_registry():
    return ModelRegistry([
        ModelSpec(
            model_idx=0,
            name="fake-fast",
            client=FakeModelClient("fake-fast"),
        ),
        ModelSpec(
            model_idx=1,
            name="fake-slow",
            client=FakeModelClient("fake-slow", delay_s=0.1),
        ),
        ModelSpec(
            model_idx=2,
            name="fake-fail",
            client=FakeModelClient("fake-fail", fail=True),
        ),
    ])


def test_runner_passes_only_allowed_context(fake_registry):
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="a",
                model_idx=0,
                instruction="Do A.",
                access_list=[],
            ),
            Step(
                step_id="b",
                model_idx=0,
                instruction="Do B.",
                access_list=[],
            ),
            Step(
                step_id="c",
                model_idx=0,
                instruction="Do C.",
                access_list=["a"],
            ),
            Step(
                step_id="final",
                model_idx=0,
                instruction="Write final.",
                access_list=["question", "a", "b", "c"],
            ),
        ],
    )

    runner = Runner(fake_registry)
    result = asyncio.run(runner.run(task))

    c_output = result.outputs["c"].text
    final_output = result.outputs["final"].text

    assert "a" in c_output
    assert "b" not in c_output
    assert "a" in final_output
    assert "b" in final_output
    assert "c" in final_output


def test_unknown_model_idx_raises(fake_registry):
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="final",
                model_idx=999,
                instruction="Write final.",
                access_list=["question"],
            ),
        ],
    )

    runner = Runner(fake_registry)

    with pytest.raises(ValueError, match="Model '999' not found"):
        asyncio.run(runner.run(task))


def test_runner_emits_step_events(fake_registry):
    events = []
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="final",
                model_idx=0,
                instruction="Answer.",
                access_list=["question"],
            )
        ],
    )

    asyncio.run(Runner(fake_registry, event_handler=lambda *event: events.append(event)).run(task))

    assert [event[0] for event in events] == ["started", "completed"]
    assert events[1][2].step_id == "final"


def test_runner_applies_worker_decoding_settings(fake_registry):
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="final",
                model_idx=0,
                instruction="Answer.",
                access_list=["question"],
            )
        ],
    )

    asyncio.run(Runner(fake_registry, max_worker_tokens=4096, worker_temperature=0.2).run(task))

    assert fake_registry.get(0).client.calls[0]["max_tokens"] == 4096
    assert fake_registry.get(0).client.calls[0]["temperature"] == 0.2


def test_runner_adds_final_answer_protocol_when_instruction_omits_it(fake_registry):
    task = Task(
        task_type="test",
        difficulty=Difficulty.EASY,
        question="Question?",
        workflow=[
            Step(
                step_id="answer",
                model_idx=0,
                instruction="Answer clearly.",
                access_list=["question"],
            )
        ],
    )

    asyncio.run(Runner(fake_registry).run(task))

    instruction = fake_registry.get(0).client.calls[0]["instruction"]
    assert instruction.endswith("End with a separate line exactly formatted as FINAL: <answer>.")
