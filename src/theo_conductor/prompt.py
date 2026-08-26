import json
from pathlib import Path
from string import Template

from .models.registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_DIR = PROJECT_ROOT / "examples"
TASK_TYPES = [
    "physics",
    "chemistry",
    "biology",
    "mathematics",
    "computer_science",
    "engineering",
    "earth_and_space_science",
    "medicine_and_health",
    "social_science",
    "humanities",
    "interdisciplinary",
    "other",
]


def build_conductor_json_schema(model_registry: ModelRegistry | None = None) -> dict:
    """Return the JSON Schema used to constrain conductor decoding.

    Keeping this schema separate from ``Task`` is intentional: the original
    question is supplied by the caller and must not be echoed by the planner.
    """

    model_ids = model_registry.model_ids() if model_registry is not None else []
    if model_registry is not None and not model_ids:
        raise ValueError("Cannot build a conductor workflow without configured models")
    if any(not isinstance(model_id, str) for model_id in model_ids):
        raise ValueError("Conductor worker model IDs must be strings")

    model_id_schema = {"type": "string", **({"enum": model_ids} if model_ids else {})}

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
            "task_type": {"type": "string", "enum": TASK_TYPES},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "workflow": {
                "type": "array",
                "items": step_schema,
                "minItems": 1,
                "maxItems": 7,
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
    prompt_path = PROJECT_ROOT / "conductor-prompt.txt"
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
        if not isinstance(model_id, str):
            raise ValueError("Conductor worker model IDs must be strings")
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
        model_lines = [f"- model_id={json.dumps(model_id)}{suffix}"]
        for label in ("role", "best_for", "useful_for", "routing_note"):
            value = getattr(spec, label)
            if value:
                model_lines.append(f"  - {label}: {value}")
        lines.append("\n".join(model_lines))

    return lines


def build_default_examples(
    model_registry: ModelRegistry,
    *,
    examples_dir: str | Path = DEFAULT_EXAMPLES_DIR,
) -> list[str]:
    """Load every JSON workflow example in stable filename order."""

    # Retain the registry argument for the public API used by training and
    # inference. Examples are authored independently of a particular worker
    # pool, while the available IDs are described separately in the prompt.
    del model_registry

    directory = Path(examples_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON conductor examples found in {directory}")

    examples: list[str] = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as example_file:
                example = json.load(example_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON conductor example {path}: {exc}") from exc

        if not isinstance(example, dict):
            raise ValueError(f"Conductor example {path} must contain a JSON object")
        title = path.stem.replace("_", " ").title()
        rendered_example = json.dumps(example, indent=2)
        examples.append(f"### {title}\n\n```json\n{rendered_example}\n```")

    return examples


def build_conductor_prompt(
    question: str,
    model_registry: ModelRegistry,
    *,
    tools: list[str] | None = None,
    examples: list[str] | None = None,
) -> str:
    return build_prompt(
        build_worker_model_lines(model_registry),
        tools or ["No external tools or code-execution environments are available to any worker."],
        examples or build_default_examples(model_registry),
        question,
    )
