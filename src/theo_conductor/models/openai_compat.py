import time
from typing import Any

from openai import AsyncOpenAI
from theo_conductor.schema import ModelResponse


class OpenAICompatibleClient:
    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(
        self,
        instruction: str,
        question: str,
        context: dict[str, str],
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


def build_message(*, instruction: str, question: str, context: dict[str, str]) -> list[dict[str, str]]:
    context_blocks = []

    for step_id, output in context.items():
        context_blocks.append(f"<step_output id={step_id}>{output}</step_output>")

    context_text = "\n".join(context_blocks)
    user_content = f"""
Original question:
{question}

Available previous step outputs:
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
