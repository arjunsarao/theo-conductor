import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

import httpx
from openai import AsyncOpenAI
from theo_conductor.schema import ModelResponse


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 600.0,
        connect_timeout_seconds: float = 5.0,
        max_retries: int = 2,
        native_batch: bool = False,
        batch_poll_interval_seconds: float = 10.0,
        batch_backend: str | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if batch_poll_interval_seconds <= 0:
            raise ValueError("batch_poll_interval_seconds must be positive")
        if batch_backend not in {None, "openai", "openrouter"}:
            raise ValueError("batch_backend must be 'openai' or 'openrouter'")
        self.model = model
        self.supports_batch = native_batch
        self.batch_poll_interval_seconds = batch_poll_interval_seconds
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_backend = batch_backend or (
            "openrouter" if "openrouter.ai" in self.base_url else "openai"
        )
        self.timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=max_retries,
        )

    async def generate(
        self,
        instruction: str,
        question: str,
        context: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        request = self.build_request(
            instruction=instruction,
            question=question,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        start = time.perf_counter()
        completion = await self.client.chat.completions.create(
            **request,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return model_response_from_completion(completion, latency_ms=latency_ms)

    def build_request(
        self,
        *,
        instruction: str,
        question: str,
        context: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": build_message(instruction=instruction, question=question, context=context),
            "max_tokens": max_tokens or 2048,
            "temperature": temperature if temperature is not None else 0.2,
        }
        if response_format is not None:
            request["response_format"] = response_format
        return request

    async def generate_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> list[ModelResponse | Exception]:
        """Execute requests through the OpenAI-compatible asynchronous Batch API.

        This is deliberately opt-in via ``native_batch`` because many servers
        implement chat completions but not files/batches. Results retain input
        order even though provider output JSONL may be unordered.
        """
        if not self.supports_batch:
            raise RuntimeError("native batch support is not enabled for this client")
        if not requests:
            return []
        if self.batch_backend == "openrouter":
            return await self._generate_openrouter_batch(requests)
        return await self._generate_openai_batch(requests)

    async def _generate_openrouter_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> list[ModelResponse | Exception]:
        """Execute an inline OpenRouter Batch Beta job and poll its results."""
        items = []
        for index, request in enumerate(requests):
            body = self.build_request(**request)
            items.append({"custom_id": f"worker-{index}", "body": body})
        payload = {
            # OpenRouter requires this serialization order for stream parsing.
            "endpoint": "/v1/chat/completions",
            "model": self.model,
            "requests": items,
        }
        api_root = self.base_url.removesuffix("/v1")
        batches_url = f"{api_root}/beta/batches"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started_at = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            response = await client.post(batches_url, json=payload)
            response.raise_for_status()
            batch = response.json()
            batch_id = str(batch.get("id") or "")
            if not batch_id:
                raise RuntimeError("OpenRouter batch submission returned no batch id")
            terminal = {"completed", "failed", "expired", "cancelled"}
            while batch.get("status") not in terminal:
                await asyncio.sleep(self.batch_poll_interval_seconds)
                response = await client.get(f"{batches_url}/{batch_id}")
                response.raise_for_status()
                batch = response.json()
        if batch.get("status") != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch_id} ended with status {batch.get('status')}: "
                f"{batch.get('error')}"
            )
        latency_ms = (time.perf_counter() - started_at) * 1000
        return batch_results_from_items(
            batch.get("results") or [],
            request_count=len(requests),
            latency_ms=latency_ms,
        )

    async def _generate_openai_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> list[ModelResponse | Exception]:
        """Execute an OpenAI Files/Batch job and download its JSONL results."""

        lines = []
        for index, request in enumerate(requests):
            lines.append(json.dumps({
                "custom_id": f"worker-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self.build_request(**request),
            }, ensure_ascii=False))
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        started_at = time.perf_counter()
        uploaded = await self.client.files.create(
            file=("worker-batch.jsonl", payload),
            purpose="batch",
        )
        batch = await self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        terminal = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal:
            await asyncio.sleep(self.batch_poll_interval_seconds)
            batch = await self.client.batches.retrieve(batch.id)
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(f"batch {batch.id} ended with status {batch.status}")

        response_file = await self.client.files.content(batch.output_file_id)
        text = response_file.text
        latency_ms = (time.perf_counter() - started_at) * 1000
        items = []
        for line in text.splitlines():
            if not line.strip():
                continue
            items.append(json.loads(line))
        return batch_results_from_items(
            items,
            request_count=len(requests),
            latency_ms=latency_ms,
        )


def model_response_from_completion(completion: Any, *, latency_ms: float) -> ModelResponse:

    choice = completion.choices[0]
    usage = completion.usage.model_dump() if completion.usage is not None else None
    return ModelResponse(
        text=choice.message.content or "",
        raw=completion,
        usage=usage,
        latency_ms=latency_ms,
        finish_reason=getattr(choice, "finish_reason", None),
    )


def model_response_from_body(body: dict[str, Any], *, latency_ms: float) -> ModelResponse:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("batch completion contained no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    return ModelResponse(
        text=message.get("content") or "",
        raw=body,
        usage=body.get("usage"),
        latency_ms=latency_ms,
        finish_reason=choice.get("finish_reason"),
    )


def batch_results_from_items(
    items: Sequence[dict[str, Any]],
    *,
    request_count: int,
    latency_ms: float,
) -> list[ModelResponse | Exception]:
    """Restore provider batch results to request order with item-level errors."""
    results: list[ModelResponse | Exception | None] = [None] * request_count
    for item in items:
        custom_id = str(item.get("custom_id") or "")
        try:
            index = int(custom_id.removeprefix("worker-"))
        except ValueError:
            continue
        if index < 0 or index >= request_count:
            continue
        error = item.get("error")
        response = item.get("response") or {}
        if error or int(response.get("status_code", 0)) != 200:
            results[index] = RuntimeError(
                f"batch item failed: {error or response.get('body')}"
            )
            continue
        try:
            results[index] = model_response_from_body(
                response.get("body") or {},
                latency_ms=latency_ms,
            )
        except Exception as exc:
            results[index] = exc
    return [
        result if result is not None else RuntimeError(f"batch result missing for worker-{index}")
        for index, result in enumerate(results)
    ]


def build_message(*, instruction: str, question: str, context: dict[str, Any]) -> list[dict[str, str]]:
    context_blocks = []

    for step_id, output in context.items():
        if step_id == "artifacts":
            context_blocks.append(f"<artifacts>{output}</artifacts>")
        else:
            context_blocks.append(f"<step_output id={step_id}>{output}</step_output>")

    context_text = "\n".join(context_blocks)
    user_content = f"""
Original question:
{question}

Available context:
{context_text if context_text else "(none)"}

Your instruction:
{instruction}
"""

    return [
        {
            "role": "system",
            "content": "You are a worker model in a multi-model reasoning workflow. Follow the instruction exactly.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
