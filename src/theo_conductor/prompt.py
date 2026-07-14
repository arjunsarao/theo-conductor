import json
from pathlib import Path
from string import Template

from .models.registry import ModelRegistry


def build_prompt(model_list: list[str], tool_list: list[str], example_list: list[str], query: str) -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "conductor-prompt.txt"
    with prompt_path.open() as f:
        prompt_template = f.read()

    template = Template(prompt_template)

    return template.substitute(
        models="\n".join(model_list),
        tools="\n".join(tool_list),
        examples="\n".join(example_list),
        query=query,
    )


def build_worker_model_lines(model_registry: ModelRegistry) -> list[str]:
    lines: list[str] = []

    for model_id, spec in model_registry._models.items():
        details = []
        if spec.display_name:
            details.append(f"name={spec.display_name}")
        if spec.provider:
            details.append(f"provider={spec.provider}")
        if spec.tags:
            details.append(f"tags={','.join(sorted(spec.tags))}")
        if spec.supports_json:
            details.append("supports_json=true")
        if spec.supports_tools:
            details.append("supports_tools=true")

        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- model_id={json.dumps(model_id)}{suffix}")

    return lines


def build_default_examples(model_registry: ModelRegistry) -> list[str]:
    model_ids = list(model_registry._models)
    if not model_ids:
        return []

    example = {
        "task_type": "general",
        "difficulty": "medium",
        "workflow": [
            {
                "step_id": "final",
                "model_id": model_ids[0],
                "instruction": "Solve the problem. End with a separate line exactly formatted as FINAL: <answer>.",
                "access_list": ["question"],
            }
        ],
    }
    return [json.dumps(example, indent=2)]


def build_conductor_prompt(
    question: str,
    model_registry: ModelRegistry,
    *,
    tools: list[str] | None = None,
    examples: list[str] | None = None,
) -> str:
    return build_prompt(
        build_worker_model_lines(model_registry),
        tools or ["No external tools are available."],
        examples or build_default_examples(model_registry),
        question,
    )
