import asyncio
from collections.abc import Callable
import json
import time
from typing import Any

from .artifact import ArtifactStore
from .scheduler import topological_sort
from .schema import ModelSpec, Task, RunResult, StepOutput, Step
from .models.registry import ModelRegistry
from .validate import validate_task


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


class Runner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        tool_registry=None,
        event_handler: Callable[[str, Step, StepOutput | None], None] | None = None,
        artifact_store: ArtifactStore | None = None,
        max_worker_tokens: int | None = None,
        use_model_output_limits: bool = False,
        worker_temperature: float = 0.2,
    ) -> None:
        if max_worker_tokens is not None and max_worker_tokens <= 0:
            raise ValueError("max_worker_tokens must be positive")
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.artifact_store = artifact_store
        self.event_handler = event_handler
        self.max_worker_tokens = max_worker_tokens
        self.use_model_output_limits = use_model_output_limits
        self.worker_temperature = worker_temperature

    async def run(self, task: Task) -> RunResult:
        started_at = time.perf_counter()
        validate_task(task, self.model_registry)
        layers = topological_sort(task)
        outputs: dict[str, StepOutput] = {}
        active_calls = 0
        peak_concurrency = 0

        async def tracked_run_step(step: Step) -> StepOutput:
            nonlocal active_calls, peak_concurrency
            active_calls += 1
            peak_concurrency = max(peak_concurrency, active_calls)
            try:
                return await self.run_step(step, task, outputs)
            finally:
                active_calls -= 1

        for layer in layers:
            layer_results = await asyncio.gather(*[tracked_run_step(step) for step in layer])

            for step, result in zip(layer, layer_results):
                outputs[step.step_id] = result

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
        response = await spec.client.generate(**request)
        return self.complete_step(step, spec, response)

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

        return spec, {
            "instruction": instruction,
            "question": task.question,
            "context": context,
            "max_tokens": max_tokens,
            "temperature": self.worker_temperature,
        }

    def complete_step(self, step: Step, spec: ModelSpec, response: Any) -> StepOutput:
        """Convert a client response into the established worker output."""
        output = StepOutput(
            step_id=step.step_id,
            model_id=step.model_id,
            text=response.text,
            usage=_usage_with_estimated_cost(response.usage, spec),
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
        )
        if self.event_handler:
            self.event_handler("completed", step, output)
        return output
