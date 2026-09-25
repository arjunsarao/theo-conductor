import asyncio
import json

import pytest

from theo_conductor.models.registry import ModelRegistry
from theo_conductor.runner import Runner
from theo_conductor.schema import Difficulty, ModelResponse, ModelSpec, Step, Task, ToolCall
from theo_conductor.tools import (
    ClarificationRequest, RequestClarificationTool, ToolRegistry,
)


class ToolCallingClient:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return ModelResponse(
                text="",
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                latency_ms=3.0,
                finish_reason="tool_calls",
                tool_calls=(ToolCall(
                    call_id="call-1",
                    name="request_clarification",
                    arguments={
                        "question": "Closed or driven?",
                        "reason": "It changes the equation.",
                        "blocking": True,
                        "options": [
                            {"id": "closed", "description": "Conservative."},
                            {"id": "driven", "description": "Include forcing."},
                        ],
                        "recommended_option": "closed",
                        "impact_if_unanswered": "high",
                        "needed_by": "before_derivation",
                    },
                ),),
                conversation=[{"role": "assistant", "tool_calls": [{"id": "call-1"}]}],
            )
        return ModelResponse(
            text="FINAL: closed",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            latency_ms=5.0,
        )

def test_runner_executes_tool_continues_worker_and_logs_result():
    async def answer(request, context):
        assert context.step.step_id == "final"
        return {
            "answered": True,
            "selected_option": "closed",
            "answer": "Use a closed system.",
        }

    client = ToolCallingClient()
    registry = ModelRegistry([
        ModelSpec(model_idx="worker", client=client, supports_tools=True),
    ])
    tools = ToolRegistry([RequestClarificationTool(answer)])
    task = Task(
        task_type="physics",
        difficulty=Difficulty.MEDIUM,
        question="Derive the equation.",
        workflow=[Step(
            step_id="final",
            model_id="worker",
            instruction="Derive it.",
            access_list=["question"],
        )],
    )

    output = asyncio.run(Runner(registry, tool_registry=tools).run(task)).outputs["final"]

    assert output.text == "FINAL: closed"
    assert output.usage == {
        "prompt_tokens": 6,
        "completion_tokens": 3,
        "total_tokens": 9,
    }
    assert output.latency_ms == 8.0
    assert output.tool_calls[0].name == "request_clarification"
    assert output.tool_calls[0].arguments["blocking"] is True
    assert output.tool_calls[0].result["selected_option"] == "closed"
    assert output.tool_calls[0].is_error is False
    assert client.calls[0]["tools"][0]["function"]["name"] == "request_clarification"
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["answered"] is True

def test_runner_asks_gpt6_astra_by_default_and_tracks_its_usage():
    class AstraClient:
        async def generate(self, **kwargs):
            return ModelResponse(
                text=json.dumps({
                    "selected_option": "closed",
                    "answer": "Use a closed system.",
                    "reasoning": "This is the conventional interpretation.",
                }),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                latency_ms=7.0,
            )

    worker = ToolCallingClient()
    registry = ModelRegistry([
        ModelSpec(model_idx="worker", client=worker, supports_tools=True),
        ModelSpec(
            model_idx="gpt-6-astra",
            name="gpt-6-astra",
            client=AstraClient(),
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=50.0,
        ),
    ])
    task = Task(
        task_type="physics",
        difficulty=Difficulty.MEDIUM,
        question="Derive the equation.",
        workflow=[Step(step_id="final", model_id="worker", instruction="Derive it.")],
    )

    output = asyncio.run(Runner(registry).run(task)).outputs["final"]

    result = json.loads(worker.calls[1]["messages"][-1]["content"])
    assert result["source"] == "conductor"
    assert result["selected_option"] == "closed"
    assert output.tool_calls[0].model_id == "gpt-6-astra"
    assert output.tool_calls[0].usage["total_tokens"] == 15
    assert output.tool_calls[0].usage["estimated_cost_usd"] == pytest.approx(0.00035)


def test_runner_can_opt_in_to_asking_the_user(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "driven")
    worker = ToolCallingClient()
    registry = ModelRegistry([
        ModelSpec(model_idx="worker", client=worker, supports_tools=True),
    ])
    task = Task(
        task_type="physics",
        difficulty=Difficulty.MEDIUM,
        question="Derive the equation.",
        workflow=[Step(step_id="final", model_id="worker", instruction="Derive it.")],
    )

    asyncio.run(Runner(
        registry,
        ask_user_for_clarification=True,
    ).run(task))

    result = json.loads(worker.calls[1]["messages"][-1]["content"])
    assert result["selected_option"] == "driven"


def test_clarification_rejects_unknown_recommended_option():
    with pytest.raises(ValueError, match="recommended_option must match"):
        ClarificationRequest.model_validate({
            "question": "Which?",
            "reason": "Needed.",
            "blocking": True,
            "options": [{"id": "a", "description": "Option A"}],
            "recommended_option": "missing",
            "impact_if_unanswered": "high",
            "needed_by": "now",
        })


def test_tool_registry_logs_validation_errors_without_invoking_handler():
    invoked = False

    async def answer(request, context):
        nonlocal invoked
        invoked = True
        return {}

    task = Task(
        task_type="test", difficulty=Difficulty.EASY, question="Q?",
        workflow=[Step(step_id="s", model_id="m", instruction="I")],
    )
    registry = ToolRegistry([RequestClarificationTool(answer)])
    from theo_conductor.tools import ToolExecutionContext
    record = asyncio.run(registry.execute(
        ToolCall(call_id="bad", name="request_clarification", arguments={}),
        ToolExecutionContext(task=task, step=task.workflow[0], outputs={}),
    ))

    assert record.is_error is True
