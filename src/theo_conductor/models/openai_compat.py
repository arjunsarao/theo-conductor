import time
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
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds),
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
        messages = build_message(instruction=instruction, question=question, context=context)
        start = time.perf_counter()

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 2048,
            "temperature": temperature if temperature is not None else 0.2,
        }
        if response_format is not None:
            request["response_format"] = response_format

        completion = await self.client.chat.completions.create(
            **request,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        text = completion.choices[0].message.content or ""
        usage = None
        if completion.usage is not None:
            usage = completion.usage.model_dump()

        return ModelResponse(text=text, raw=completion, usage=usage, latency_ms=latency_ms)


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
