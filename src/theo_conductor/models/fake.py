import asyncio
from typing import Any

from theo_conductor.schema import ModelResponse


class FakeModelClient:
    def __init__(
        self,
        name: str,
        *,
        delay_s: float = 0.0,
        fail: bool = False,
    ):
        self.name = name
        self.delay_s = delay_s
        self.fail = fail
        self.calls = []

    async def generate(
        self,
        *,
        instruction: str,
        question: str,
        context: dict[str, str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "instruction": instruction,
                "question": question,
                "context": context,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )

        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        if self.fail:
            raise RuntimeError(f"{self.name} failed intentionally")

        context_keys = ",".join(sorted(context.keys()))

        return ModelResponse(
            text=(f"[model={self.name}] " f"instruction={instruction!r}; " f"context_keys={context_keys}"),
            usage={"fake_tokens": 123},
            latency_ms=self.delay_s * 1000,
        )
