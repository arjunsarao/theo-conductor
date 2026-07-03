"""
Schedules steps in a task based on their dependencies DAG.
"""
from .schema import Task, Step
from collections import deque

RESERVED_CONTEXT_KEYS = {"question", "system_prompt", "tool_docs"}


def compute_dependencies(step: Step) -> list[str]:
    return [
        access_key
        for access_key in step.access_list
        if access_key not in RESERVED_CONTEXT_KEYS
    ]

def topological_sort(task: Task) -> list[list[Step]]:
    dependencies_by_step = {
        step.step_id: compute_dependencies(step)
        for step in task.workflow
    }

    steps_by_id = {step.step_id: step for step in task.workflow}

    if len(steps_by_id) != len(task.workflow):
        raise ValueError("Duplicate step_id found in workflow.")

    in_degree = {step.step_id: 0 for step in task.workflow}
    children: dict[str, list[str]] = {step.step_id: [] for step in task.workflow}

    for step in task.workflow:
        for dep in dependencies_by_step[step.step_id]:
            if dep not in steps_by_id:
                raise ValueError(
                    f"Step {step.step_id} depends on unknown step {dep}."
                )

            in_degree[step.step_id] += 1
            children[dep].append(step.step_id)

    queue = deque(
        step_id
        for step_id, degree in in_degree.items()
        if degree == 0
    )

    execution_layers: list[list[Step]] = []
    processed_count = 0

    while queue:
        layer_size = len(queue)
        current_layer: list[Step] = []

        for _ in range(layer_size):
            current_step_id = queue.popleft()
            current_layer.append(steps_by_id[current_step_id])
            processed_count += 1

            for child_id in children[current_step_id]:
                in_degree[child_id] -= 1

                if in_degree[child_id] == 0:
                    queue.append(child_id)

        execution_layers.append(current_layer)

    if processed_count != len(task.workflow):
        raise ValueError("The workflow has a cycle.")

    return execution_layers
