import json
from pathlib import Path
from string import Template

from .models.registry import ModelRegistry


def build_conductor_json_schema(model_registry: ModelRegistry | None = None) -> dict:
    """Return the JSON Schema used to constrain conductor decoding.

    Keeping this schema separate from ``Task`` is intentional: the original
    question is supplied by the caller and must not be echoed by the planner.
    """

    model_ids = model_registry.model_ids() if model_registry is not None else []
    if model_registry is not None and not model_ids:
        raise ValueError("Cannot build a conductor workflow without configured models")

    model_id_schema = (
        {"enum": model_ids}
        if model_ids
        else {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    )

    step_schema = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1},
            "model_id": model_id_schema,
            "instruction": {"type": "string", "minLength": 1},
            "access_list": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["step_id", "model_id", "instruction", "access_list"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "minLength": 1},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "workflow": {
                "type": "array",
                "items": step_schema,
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": ["task_type", "difficulty", "workflow"],
        "additionalProperties": False,
    }


def build_conductor_response_format(model_registry: ModelRegistry) -> dict:
    """Wrap the conductor schema for OpenAI-compatible structured outputs."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "conductor_workflow",
            "strict": True,
            "schema": build_conductor_json_schema(model_registry),
        },
    }


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
