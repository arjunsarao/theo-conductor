import asyncio
from collections.abc import Callable

from .scheduler import topological_sort
from .schema import Task, RunResult, StepOutput, Step
from .models.registry import ModelRegistry
from .validate import validate_task


class Runner:
    def __init__(
        self,
        model_registry: ModelRegistry,
        tool_registry=None,
        event_handler: Callable[[str, Step, StepOutput | None], None] | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.event_handler = event_handler

    async def run(self, task: Task) -> RunResult:
        validate_task(task, self.model_registry)
        layers = topological_sort(task)
        outputs: dict[str, StepOutput] = {}

        for layer in layers:
            layer_results = await asyncio.gather(*[self.run_step(step, task, outputs) for step in layer])

            for step, result in zip(layer, layer_results):
                outputs[step.step_id] = result

        return RunResult(task=task, outputs=outputs)

    async def run_step(self, step: Step, task: Task, outputs: dict[str, StepOutput]) -> StepOutput:
        if self.event_handler:
            self.event_handler("started", step, None)
        spec = self.model_registry.get(step.model_id)

        context = {key: outputs[key] for key in step.access_list if key in outputs}

        response = await spec.client.generate(
            instruction=step.instruction,
            question=task.question,
            context=context,
            max_tokens=getattr(step, "max_tokens", None),
            temperature=getattr(step, "temperature", None),
        )

        output = StepOutput(
            step_id=step.step_id,
            model_id=step.model_id,
            text=response.text,
            usage=response.usage,
            latency_ms=response.latency_ms,
        )
        if self.event_handler:
            self.event_handler("completed", step, output)
        return output
