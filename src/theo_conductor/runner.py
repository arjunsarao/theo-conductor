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
        max_worker_tokens: int = 4096,
        worker_temperature: float = 0.2,
    ) -> None:
        if max_worker_tokens <= 0:
            raise ValueError("max_worker_tokens must be positive")
        self.model_registry = model_registry
        self.tool_registry = tool_registry
        self.event_handler = event_handler
        self.max_worker_tokens = max_worker_tokens
        self.worker_temperature = worker_temperature

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

        instruction = step.instruction
        if step.step_id == task.workflow[-1].step_id and "final:" not in instruction.lower():
            instruction = f"{instruction.rstrip()}\n\nEnd with a separate line exactly formatted as FINAL: <answer>."

        response = await spec.client.generate(
            instruction=instruction,
            question=task.question,
            context=context,
            max_tokens=self.max_worker_tokens,
            temperature=self.worker_temperature,
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
