import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from theo_conductor.models.openai_compat import OpenAICompatibleClient, build_message


def _completion(text: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )


def test_openai_compatible_client_forwards_constrained_response_format():
    client = OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="planner")
    create = AsyncMock(return_value=_completion())
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "workflow", "strict": True, "schema": {"type": "object"}},
    }

    asyncio.run(
        client.generate(
            instruction="Plan.",
            question="Question?",
            context={},
            response_format=response_format,
        )
    )

    assert create.await_args.kwargs["response_format"] is response_format


def test_openai_compatible_client_does_not_constrain_worker_responses():
    client = OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="worker")
    create = AsyncMock(return_value=_completion("answer"))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    asyncio.run(client.generate(instruction="Answer.", question="Question?", context={}))

    assert "response_format" not in create.await_args.kwargs


def test_openai_compatible_client_configures_transport_timeouts_and_retries():
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        model="judge",
        timeout_seconds=600,
        connect_timeout_seconds=30,
        max_retries=0,
    )

    assert client.client.timeout.read == 600
    assert client.client.timeout.connect == 30
    assert client.client.max_retries == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds must be positive"),
        ({"connect_timeout_seconds": 0}, "connect_timeout_seconds must be positive"),
        ({"max_retries": -1}, "max_retries must be non-negative"),
    ],
)
def test_openai_compatible_client_rejects_invalid_transport_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OpenAICompatibleClient(base_url="http://localhost:8000/v1", model="judge", **kwargs)


def test_build_message_labels_artifacts_separately_from_step_outputs():
    messages = build_message(
        instruction="Analyze.",
        question="Question?",
        context={"solver": "answer", "artifacts": '[{"artifact_id": "results"}]'},
    )

    content = messages[1]["content"]
    assert "<step_output id=solver>answer</step_output>" in content
    assert '<artifacts>[{"artifact_id": "results"}]</artifacts>' in content
