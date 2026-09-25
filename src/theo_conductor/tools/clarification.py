"""Structured clarification tool and interactive CLI handler."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseTool, ToolExecutionContext, ToolOutcome


class ClarificationOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ClarificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    blocking: bool
    options: list[ClarificationOption]
    recommended_option: str | None
    impact_if_unanswered: Literal["low", "medium", "high"]
    needed_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def recommended_option_exists(self) -> "ClarificationRequest":
        option_ids = {option.id for option in self.options}
        if self.recommended_option is not None and self.recommended_option not in option_ids:
            raise ValueError("recommended_option must match an option id")
        return self


ClarificationHandler = Callable[
    [ClarificationRequest, ToolExecutionContext],
    Awaitable[Any],
]


class RequestClarificationTool(BaseTool[ClarificationRequest]):
    name = "request_clarification"
    description = (
        "Ask the configured clarification authority to resolve an ambiguity that cannot be "
        "resolved from the original question or available context."
    )
    arguments_model = ClarificationRequest

    def __init__(self, handler: ClarificationHandler) -> None:
        self.handler = handler

    async def run(
        self,
        arguments: ClarificationRequest,
        context: ToolExecutionContext,
    ) -> Any:
        return await self.handler(arguments, context)



class ConductorClarificationHandler:
    """Ask GPT-6 Astra to resolve a worker clarification request."""

    def __init__(self, model_registry: Any, model_id: str = "gpt-6-astra") -> None:
        self.model_registry = model_registry
        self.model_id = model_id

    def _model(self) -> tuple[str, Any]:
        try:
            return self.model_id, self.model_registry.get(self.model_id)
        except ValueError:
            for key, spec in self.model_registry._models.items():
                if spec.name == self.model_id or spec.display_name == "GPT-6 Astra":
                    return str(key), spec
        raise ValueError(
            "request_clarification requires GPT-6 Astra in the model registry; "
            "add model_id 'gpt-6-astra' or enable user clarification"
        )

    @staticmethod
    def _usage_with_cost(usage: dict[str, Any] | None, spec: Any) -> dict[str, Any] | None:
        if not isinstance(usage, dict):
            return usage
        result = dict(usage)
        prompt = result.get("prompt_tokens", result.get("input_tokens"))
        completion = result.get("completion_tokens", result.get("output_tokens"))
        if prompt is not None and completion is not None and spec.cost_per_1m_input_tokens is not None and spec.cost_per_1m_output_tokens is not None:
            result["estimated_cost_usd"] = (float(prompt) * spec.cost_per_1m_input_tokens + float(completion) * spec.cost_per_1m_output_tokens) / 1_000_000
        return result

    async def __call__(
        self,
        request: ClarificationRequest,
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        model_id, spec = self._model()
        options = [option.model_dump() for option in request.options]
        instruction = (
            "Resolve this clarification request for another worker. Choose the most "
            "defensible interpretation of the original problem. Return only JSON.\n\n"
            + json.dumps({
                "worker_step": context.step.step_id,
                "worker_instruction": context.step.instruction,
                "clarification": request.model_dump(),
            }, ensure_ascii=False)
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "clarification_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_option": {"type": ["string", "null"]},
                        "answer": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["selected_option", "answer", "reasoning"],
                    "additionalProperties": False,
                },
            },
        }
        response = await spec.client.generate(
            instruction=instruction,
            question=context.task.question,
            context={},
            max_tokens=1024,
            temperature=0.0,
            response_format=response_format,
        )
        answer = json.loads(response.text)
        selected = answer.get("selected_option")
        option_ids = {option["id"] for option in options}
        if selected is not None and selected not in option_ids:
            raise ValueError("GPT-6 Astra selected an unknown clarification option")
        return ToolOutcome(
            value={"answered": True, "source": "conductor", **answer},
            model_id=model_id,
            usage=self._usage_with_cost(response.usage, spec),
            latency_ms=response.latency_ms,
        )


class InteractiveClarificationHandler:
    """Serialize terminal prompts so concurrent workers cannot interleave."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        request: ClarificationRequest,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._prompt, request, context)

    @staticmethod
    def _prompt(
        request: ClarificationRequest,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        print(f"\nClarification requested by step {context.step.step_id!r}:")
        print(request.question)
        print(f"Reason: {request.reason}")
        for option in request.options:
            marker = " (recommended)" if option.id == request.recommended_option else ""
            print(f"  {option.id}: {option.description}{marker}")
        prompt = "Answer"
        if request.recommended_option:
            prompt += f" [{request.recommended_option}]"
        try:
            answer = input(f"{prompt}: ").strip()
        except EOFError:
            answer = ""
        if not answer and request.recommended_option:
            answer = request.recommended_option
        option_ids = {option.id for option in request.options}
        return {
            "answered": bool(answer),
            "selected_option": answer if answer in option_ids else None,
            "answer": answer or None,
        }
