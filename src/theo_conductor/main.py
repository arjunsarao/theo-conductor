import json
from .schema import Task
from .scheduler import topological_sort
from .traces import print_step_order
from .models.registry import ModelRegistry
from .runner import Runner

REGISTRY = ModelRegistry.from_config_dir("configs")


async def async_main() -> None:
    response = json.load(open("../../examples/parallel.json"))
    x = Task.from_dict(response)
    sorted_steps = topological_sort(x)

    r = Runner(model_registry=REGISTRY)
    print_step_order(sorted_steps)
    x = r.run(x)


if __name__ == "__main__":
    async_main()
