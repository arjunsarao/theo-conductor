import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from theo_conductor.models.openai_compat import OpenAICompatibleClient


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
