import json
from .schema import Task, ModelSpec
from .scheduler import topological_sort
from .traces import print_step_order
from .models.registry import ModelRegistry
from .models.openai_compat import OpenAICompatibleClient
from .runner import Runner

# Maybe move this to a yaml file or something later, but for now this is fine.
REGISTRY = ModelRegistry(
    [
        ModelSpec(
            provider="vllm",
            display_name="Qwen 3.5 9B Base",
            context_length=131_072,
            supports_tools=False,
            supports_json=True,
            tags={"local", "cheap", "general"},
            client=OpenAICompatibleClient(
                base_url="http://localhost:8001/v1",
                model="Qwen/Qwen3.5-9B-Base",
            ),
        ),
        ModelSpec(
            provider="vllm",
            display_name="Qwen 3 8B Megascience LoRA",
            context_length=131_072,
            supports_tools=False,
            supports_json=True,
            tags={"local", "science", "finetuned"},
            client=OpenAICompatibleClient(
                base_url="http://localhost:8002/v1",
                model="qwen-megascience",
            ),
        ),
        ModelSpec(
            provider="vllm",
            display_name="Nemotron 30B A3B",
            context_length=128_000,
            supports_tools=False,
            supports_json=True,
            tags={"local", "reasoning"},
            client=OpenAICompatibleClient(
                base_url="http://localhost:8003/v1",
                model="nvidia-nemotron",
            ),
        ),
    ]
)


async def async_main() -> None:
    response = json.load(open("../../examples/parallel.json"))
    x = Task.from_dict(response)
    sorted_steps = topological_sort(x)

    r = Runner(model_registry=REGISTRY)
    print_step_order(sorted_steps)
    x = r.run(x)


if __name__ == "__main__":
    async_main()
