import asyncio
from collections.abc import Callable, Awaitable
from dataclasses import replace
import json
import time
from typing import Any

from .artifact import ArtifactStore
from .scheduler import topological_sort, RESERVED_CONTEXT_KEYS
from .schema import ModelSpec, Task, RunResult, StepOutput, Step, ToolCallRecord
from .tools import ToolExecutionContext, ToolRegistry, default_tool_registry
from .models.registry import ModelRegistry
from .validate import validate_task
from .worker_prompt import build_worker_prompt


def _usage_with_estimated_cost(
    response_usage: dict[str, Any] | None,
    spec: ModelSpec,
) -> dict[str, Any] | None:
    """Copy token usage and add the configured list-price estimate when possible."""
    if not isinstance(response_usage, dict):
        return response_usage
    usage = dict(response_usage)
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if (
        prompt_tokens is not None
        and completion_tokens is not None
        and spec.cost_per_1m_input_tokens is not None
        and spec.cost_per_1m_output_tokens is not None
    ):
        try:
            usage["estimated_cost_usd"] = (
                float(prompt_tokens) * spec.cost_per_1m_input_tokens
                + float(completion_tokens) * spec.cost_per_1m_output_tokens
            ) / 1_000_000
        except (TypeError, ValueError):
            pass
    return usage

def _merge_usage(
    total: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not current:
        return total
    merged = dict(total or {})
    for key, value in current.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = merged.get(key, 0) + value
        else:
            merged[key] = value
    return merged


class WorkflowStepError(RuntimeError):
    def __init__(self, step_id: str, reason: str):
        self.step_id = step_id
        self.reason = reason
        super().__init__(f"Step {step_id!r}: {reason}")


def step_failure(output: StepOutput) -> str | None:
    if output.finish_reason in {"length", "max_tokens"}:
        return "capped"
    if output.finish_reason in {"error", "content_filter"}:
        return output.finish_reason
    if not output.text.strip():
        return "empty_response"
    return None


class Runner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        tool_registry: ToolRegistry | None = None,
        event_handler: Callable[[str, Step, StepOutput | None], None] | None = None,
        artifact_store: ArtifactStore | None = None,
        max_worker_tokens: int | None = None,
        use_model_output_limits: bool = False,
        worker_temperature: float = 0.2,
        max_tool_rounds: int = 8,
        tool_services: dict[str, Any] | None = None,
        ask_user_for_clarification: bool = False,
        call_checkpoint: Callable[..., Awaitable[None]] | None = None,
        tool_checkpoint: Callable[..., Awaitable[None]] | None = None,
        call_started: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if max_worker_tokens is not None and max_worker_tokens <= 0:
            raise ValueError("max_worker_tokens must be positive")
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")
        self.model_registry = model_registry
        self.tool_registry = (
            tool_registry if tool_registry is not None
            else default_tool_registry(model_registry, ask_user=ask_user_for_clarification)
        )
        self.artifact_store = artifact_store
        self.event_handler = event_handler
        self.max_worker_tokens = max_worker_tokens
        self.use_model_output_limits = use_model_output_limits
        self.worker_temperature = worker_temperature
        self.max_tool_rounds = max_tool_rounds
        self.tool_services = dict(tool_services or {})
        self.call_checkpoint = call_checkpoint
        self.tool_checkpoint = tool_checkpoint
        self.call_started = call_started

    async def run(
        self, task: Task, *,
        initial_outputs: dict[str, StepOutput] | None = None,
        checkpoint: Callable[[Step, StepOutput | None, Exception | None], Awaitable[None]] | None = None,
        fail_on_incomplete: bool = False,
    ) -> RunResult:
        started_at = time.perf_counter()
        validate_task(task, self.model_registry)
        layers = topological_sort(task)
        outputs: dict[str, StepOutput] = {}
        # A cached step is reusable only when its dependencies are reusable too.
        for layer in layers:
            for step in layer:
                cached = (initial_outputs or {}).get(step.step_id)
                dependencies = set(step.access_list) - RESERVED_CONTEXT_KEYS
                if cached is not None and cached.model_id == step.model_id and not step_failure(cached) and dependencies <= outputs.keys():
                    outputs[step.step_id] = cached
        active_calls = 0
        peak_concurrency = 0

        async def tracked_run_step(step: Step) -> StepOutput:
            nonlocal active_calls, peak_concurrency
            active_calls += 1
            peak_concurrency = max(peak_concurrency, active_calls)
            step_started = time.perf_counter()
            try:
                output = await self.run_step(step, task, outputs)
                reason = step_failure(output) if fail_on_incomplete else None
                error = WorkflowStepError(step.step_id, reason) if reason else None
                if checkpoint:
                    await checkpoint(step, output, error)
                if error:
                    raise error
                outputs[step.step_id] = output
                return output
            except Exception as exc:
                exc.elapsed_ms = (time.perf_counter() - step_started) * 1000
                if checkpoint and not isinstance(exc, WorkflowStepError):
                    await checkpoint(step, None, exc)
                raise
            finally:
                active_calls -= 1

        for layer in layers:
            pending = [step for step in layer if step.step_id not in outputs]
            layer_results = await asyncio.gather(*[tracked_run_step(step) for step in pending], return_exceptions=True)
            failures = [result for result in layer_results if isinstance(result, BaseException)]
            if failures:
                raise failures[0]

        artifacts = (
            [artifact.to_dict() for artifact in self.artifact_store.list()]
            if self.artifact_store is not None
            else []
        )
        return RunResult(
            task=task,
            outputs=outputs,
            artifacts=artifacts,
            observed_wall_time_ms=(time.perf_counter() - started_at) * 1000,
            observed_peak_concurrency=peak_concurrency,
        )

    async def run_step(self, step: Step, task: Task, outputs: dict[str, StepOutput]) -> StepOutput:
        if self.event_handler:
            self.event_handler("started", step, None)
        spec, request = self.prepare_step(step, task, outputs)
        if step.needs_tools and not self.tool_registry:
            raise ValueError(f"Step {step.step_id!r} needs tools, but no tools are registered")
        if self.tool_registry and spec.supports_tools:
            return await self._run_step_with_tools(step, task, outputs, spec, request)
        response = await self._generate(step, spec, request)
        return self.complete_step(step, spec, response)

    async def _generate(self, step: Step, spec: ModelSpec, request: dict[str, Any]):
        started = time.perf_counter()
        if self.call_started:
            await self.call_started(step, spec, request)
        try:
            response = await spec.client.generate(**request)
        except Exception as exc:
            if self.call_checkpoint:
                await self.call_checkpoint(step, spec, None, exc, (time.perf_counter()-started)*1000)
            raise
        if self.call_checkpoint:
            await self.call_checkpoint(step, spec, response, None, (time.perf_counter()-started)*1000)
        return response

    async def _run_step_with_tools(
        self,
        step: Step,
        task: Task,
        outputs: dict[str, StepOutput],
        spec: ModelSpec,
        request: dict[str, Any],
    ) -> StepOutput:
        records: list[ToolCallRecord] = []
        messages: list[dict[str, Any]] | None = None
        aggregate_usage: dict[str, Any] | None = None
        aggregate_latency_ms = 0.0
        tool_rounds = 0
        services = dict(self.tool_services)
        if self.artifact_store is not None:
            services.setdefault("artifact_store", self.artifact_store)
        context = ToolExecutionContext(
            task=task,
            step=step,
            outputs=outputs,
            services=services,
        )

        while True:
            call_request = {**request, "tools": self.tool_registry.declarations()}
            if messages is not None:
                call_request["messages"] = messages
            response = await self._generate(step, spec, call_request)
            aggregate_usage = _merge_usage(aggregate_usage, response.usage)
            aggregate_latency_ms += response.latency_ms or 0.0

            if not response.tool_calls:
                final_response = replace(
                    response,
                    usage=aggregate_usage,
                    latency_ms=aggregate_latency_ms,
                )
                return self.complete_step(step, spec, final_response, tool_calls=records)

            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                raise RuntimeError(
                    f"Step {step.step_id!r} exceeded {self.max_tool_rounds} tool rounds"
                )
            if response.conversation is None:
                raise RuntimeError("Tool-calling model response omitted conversation state")

            messages = list(response.conversation)
            for call in response.tool_calls:
                record = await self.tool_registry.execute(
                    call,
                    context,
                )
                records.append(record)
                if self.tool_checkpoint:
                    await self.tool_checkpoint(step, record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(record.result, ensure_ascii=False),
                })

    def prepare_step(
        self,
        step: Step,
        task: Task,
        outputs: dict[str, StepOutput],
    ) -> tuple[ModelSpec, dict[str, Any]]:
        """Build one worker request without sending it.

        Separating request preparation from execution lets the benchmark pool
        ready steps from many workflows into provider-native batch jobs while
        preserving the exact prompt contract used by ordinary Runner calls.
        """
        spec = self.model_registry.get(step.model_id)
        if self.max_worker_tokens is not None:
            max_tokens = self.max_worker_tokens
        elif self.use_model_output_limits:
            max_tokens = spec.max_output_tokens or 16_384
        else:
            # Preserve the established full-context behavior outside the HLE
            # benchmark unless a caller explicitly selects model output limits.
            max_tokens = spec.context_length or 16_384

        context = {key: outputs[key] for key in step.access_list if key in outputs}
        if step.artifact_inputs:
            if self.artifact_store is None:
                raise ValueError(
                    f"Step {step.step_id!r} declares artifact inputs, but Runner has no artifact store"
                )
            artifacts = [self.artifact_store.get(artifact_id) for artifact_id in step.artifact_inputs]
            context["artifacts"] = json.dumps(
                [artifact.to_dict() for artifact in artifacts],
                ensure_ascii=False,
                indent=2,
            )

        instruction = step.instruction
        if step.step_id == task.workflow[-1].step_id and "final:" not in instruction.lower():
            instruction = f"{instruction.rstrip()}\n\nEnd with a separate line exactly formatted as FINAL: <answer>."

        request: dict[str, Any] = {
            "instruction": instruction,
            "question": task.question,
            "context": context,
            "max_tokens": max_tokens,
            "temperature": self.worker_temperature,
        }
        if task.image:
            request["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "You are a worker model in a multi-model reasoning workflow. "
                        "Follow the instruction exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": build_worker_prompt(
                                instruction=instruction,
                                question=task.question,
                                context=context,
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": task.image}},
                    ],
                },
            ]
        return spec, request

    def complete_step(
        self,
        step: Step,
        spec: ModelSpec,
        response: Any,
        tool_calls: list[ToolCallRecord] | None = None,
    ) -> StepOutput:
        """Convert a client response into the established worker output."""
        output = StepOutput(
            step_id=step.step_id,
            model_id=step.model_id,
            text=response.text,
            usage=_usage_with_estimated_cost(response.usage, spec),
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            tool_calls=tool_calls or [],
        )
        if self.event_handler:
            self.event_handler("completed", step, output)
        return output
