import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import theo_conductor.models.openai_compat as openai_compat
from theo_conductor.models.openai_compat import OpenAICompatibleClient, build_message


def _completion(text: str = "{}", *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
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

    response = asyncio.run(
        client.generate(
            instruction="Plan.",
            question="Question?",
            context={},
            response_format=response_format,
        )
    )

    assert create.await_args.kwargs["response_format"] is response_format
    assert response.finish_reason == "stop"


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


def test_openrouter_batch_submits_inline_and_restores_result_order(monkeypatch):
    captured = {}

    class Response:
        def __init__(self, body, status_code=200):
            self.body = body
            self.status_code = status_code

        def json(self):
            return self.body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class BatchHttpClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json):
            captured["post_url"] = url
            captured["payload"] = json
            return Response({"id": "batch-1", "status": "validating"}, status_code=202)

        async def get(self, url):
            captured["get_url"] = url
            return Response({
                "id": "batch-1",
                "status": "completed",
                "results": [
                    {
                        "custom_id": "worker-1",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [{
                                    "message": {"content": "second"},
                                    "finish_reason": "stop",
                                }],
                                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                            },
                        },
                        "error": None,
                    },
                    {
                        "custom_id": "worker-0",
                        "response": None,
                        "error": {"code": "bad_request", "message": "bad item"},
                    },
                ],
            })

    monkeypatch.setattr(openai_compat.httpx, "AsyncClient", BatchHttpClient)
    monkeypatch.setattr(openai_compat.asyncio, "sleep", AsyncMock())
    client = OpenAICompatibleClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="secret",
        model="openai/gpt-test",
        native_batch=True,
    )
    request = {
        "instruction": "Answer.",
        "question": "Question?",
        "context": {},
        "max_tokens": 8,
        "temperature": 0.0,
    }

    results = asyncio.run(client.generate_batch([request, request]))

    assert client.batch_backend == "openrouter"
    assert captured["post_url"] == "https://openrouter.ai/api/beta/batches"
    assert captured["get_url"] == "https://openrouter.ai/api/beta/batches/batch-1"
    assert list(captured["payload"]) == ["endpoint", "model", "requests"]
    assert captured["payload"]["endpoint"] == "/v1/chat/completions"
    assert captured["payload"]["model"] == "openai/gpt-test"
    assert [item["custom_id"] for item in captured["payload"]["requests"]] == [
        "worker-0",
        "worker-1",
    ]
    assert all(
        item["body"]["model"] == "openai/gpt-test"
        for item in captured["payload"]["requests"]
    )
    assert isinstance(results[0], RuntimeError)
    assert results[1].text == "second"
    assert results[1].usage["completion_tokens"] == 1


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
