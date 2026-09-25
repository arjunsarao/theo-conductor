"""Extensible, provider-neutral worker tool runtime."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from ..schema import Step, Task, ToolCall, ToolCallRecord


ArgumentsT = TypeVar("ArgumentsT", bound=BaseModel)


@dataclass(frozen=True)
class ToolExecutionContext:
    """Run-scoped capabilities available to a tool implementation.

    ``services`` is an explicit extension point for runtimes, sandboxes,
    schedulers, artifact stores, and other capabilities future tools require.
    """

    task: Task
    step: Step
    outputs: Mapping[str, Any]
    services: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutcome:
    value: Any
    is_error: bool = False
    model_id: str | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None


class BaseTool(ABC, Generic[ArgumentsT]):
    """A validated async tool exposed to worker models."""

    name: str
    description: str
    arguments_model: type[ArgumentsT]

    def declaration(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }

    @abstractmethod
    async def run(
        self,
        arguments: ArgumentsT,
        context: ToolExecutionContext,
    ) -> Any | ToolOutcome:
        """Execute validated arguments and return a JSON-serializable value."""


class ToolRegistry:
    def __init__(self, tools: list[BaseTool[Any]] | None = None) -> None:
        self._tools: dict[str, BaseTool[Any]] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool[Any]) -> None:
        if not tool.name or not tool.name.strip():
            raise ValueError("Tool names must be non-empty")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def declarations(self) -> list[dict[str, Any]]:
        return [tool.declaration() for tool in self._tools.values()]

    def __bool__(self) -> bool:
        return bool(self._tools)

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolCallRecord:
        started_at = time.perf_counter()
        is_error = False
        model_id = None
        usage = None
        model_latency_ms = None
        try:
            tool = self._tools.get(call.name)
            if tool is None:
                raise ValueError(f"Unknown tool: {call.name}")
            arguments = tool.arguments_model.model_validate(call.arguments)
            outcome = await tool.run(arguments, context)
            if isinstance(outcome, ToolOutcome):
                result, is_error = outcome.value, outcome.is_error
                model_id = outcome.model_id
                usage = outcome.usage
                model_latency_ms = outcome.latency_ms
            else:
                result = outcome
            json.dumps(result)
        except (ValidationError, ValueError, TypeError) as exc:
            result = {"error": str(exc)}
            is_error = True
        except Exception as exc:
            result = {"error": f"Tool execution failed: {type(exc).__name__}"}
            is_error = True
        return ToolCallRecord(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            is_error=is_error,
            duration_ms=model_latency_ms or (time.perf_counter() - started_at) * 1000,
            model_id=model_id,
            usage=usage,
        )
