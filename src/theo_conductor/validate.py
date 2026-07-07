"""
Given a json thats supposed to specify a workflow, validate it. A valid workflow must have the following properties:
1. The workflow must be a list of steps, where each step is a dictionary with the following keys:
    - step_id: a unique string identifier for the step
    - model_id: a string identifier for the model to be used in the step, it must correspond to a model in the model registry
    - instruction: a string instruction for the model to follow
    - access_list: a list of strings specifying the inputs that the model can access. The access_list must include the question and any previous steps that the model needs to access. The access_list must not include any future steps.
    - needs_tools: a boolean indicating whether the model needs to use any tools to complete the step. If needs_tools is true, then the model must have access to the tools specified in the access_list.
2. The workflow must have a final step that synthesizes the answer from the previous steps. The final step must have access to all previous steps and the question. The final step must not have access to any future steps.
3. The workflow must not have any circular dependencies. A step cannot depend on itself or any future steps.
4. The workflow must have a valid task_type and difficulty. The task_type must be a string and the difficulty must be one of "easy", "medium", or "hard".
5. The workflow must have a valid question. The question must be a string that describes the problem to be solved.
"""

from .schema import Task
from .scheduler import RESERVED_CONTEXT_KEYS, topological_sort
from .models.registry import ModelRegistry


def validate_task(task: Task, model_registry: ModelRegistry | None = None) -> None:
    if not task.task_type.strip():
        raise ValueError("task_type must be a non-empty string.")

    if not task.question.strip():
        raise ValueError("question must be a non-empty string.")

    if not task.workflow:
        raise ValueError("workflow must contain at least one step.")

    seen: set[str] = set()

    for index, step in enumerate(task.workflow):
        if not step.step_id.strip():
            raise ValueError(f"workflow[{index}].step_id must be non-empty.")

        if step.step_id in seen:
            raise ValueError(f"Duplicate step_id found: {step.step_id!r}.")

        if not step.instruction.strip():
            raise ValueError(f"Step {step.step_id!r} instruction must be non-empty.")

        allowed = seen | RESERVED_CONTEXT_KEYS

        for access_key in step.access_list:
            if access_key == step.step_id:
                raise ValueError(f"Step {step.step_id!r} cannot depend on itself.")

            if access_key not in allowed:
                raise ValueError(f"Step {step.step_id!r} accesses unknown or future key {access_key!r}.")

        if model_registry is not None:
            spec = model_registry.get(step.model_id)

            if step.needs_tools and not spec.supports_tools:
                raise ValueError(
                    f"Step {step.step_id!r} needs tools, but model {step.model_id!r} " "does not support tools."
                )

        seen.add(step.step_id)

    final = task.workflow[-1]

    if final.step_id != "final":
        raise ValueError("Final workflow step must have step_id 'final'.")

    required_final_access = {"question"} | {step.step_id for step in task.workflow[:-1]}

    missing = required_final_access - set(final.access_list)
    if missing:
        raise ValueError(f"Final step is missing required access keys: {sorted(missing)}.")

    topological_sort(task)
