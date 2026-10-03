from pathlib import Path

from .models.registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "conductor-prompt-refined.md"
QUESTION_MARKER = "<USER QUESTION>"
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


def build_conductor_json_schema(model_registry: ModelRegistry | None = None, *, model_ids: list[str] | None = None) -> dict:
    """Return the JSON Schema used to constrain conductor decoding.

    Keeping this schema separate from ``Task`` is intentional: the original
    question is supplied by the caller and must not be echoed by the planner.
    """

    model_ids = model_registry.model_ids() if model_registry is not None else (model_ids or [])
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


def load_prompt_template(path: Path) -> tuple[str, list[str]]:
    import re

    template = path.read_text(encoding="utf-8")
    if template.count(QUESTION_MARKER) != 1:
        raise ValueError(f"{path} must contain exactly one {QUESTION_MARKER!r} marker")
    ids = list(dict.fromkeys(re.findall(r'^- model_id="([^"]+)"', template, re.MULTILINE)))
    if not ids:
        raise ValueError(f"Could not find worker model_id entries in {path}")
    return template, ids


def workflow_response_format(model_ids: list[str]) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "conductor_workflow", "strict": True,
        "schema": build_conductor_json_schema(model_ids=model_ids),
    }}


def validate_plan(plan: object, model_ids: set[str]) -> dict:
    """Validate generated compact plans before marking generation successful."""
    if not isinstance(plan, dict):
        raise ValueError("completion is not a JSON object")
    if plan.get("task_type") not in TASK_TYPES:
        raise ValueError("task_type is missing or invalid")
    if plan.get("difficulty") not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty is missing or invalid")
    workflow = plan.get("workflow")
    if not isinstance(workflow, list) or not 1 <= len(workflow) <= 7:
        raise ValueError("workflow must contain between 1 and 7 steps")
    seen = {"question"}
    for index, step in enumerate(workflow):
        if not isinstance(step, dict):
            raise ValueError(f"workflow step {index} is not an object")
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip() or step_id in seen:
            raise ValueError(f"workflow step {index} has an empty, reserved, or duplicate step_id")
        if step.get("model_id") not in model_ids:
            raise ValueError(f"workflow step {step_id!r} has an unavailable model_id")
        if not isinstance(step.get("instruction"), str) or not step["instruction"].strip():
            raise ValueError(f"workflow step {step_id!r} instruction must be non-empty")
        access = step.get("access_list")
        if not isinstance(access, list) or any(not isinstance(key, str) or key not in seen for key in access):
            raise ValueError(f"workflow step {step_id!r} has a forward or unknown dependency")
        seen.add(step_id)
    return plan


def load_conductor_prompt() -> str:
    """Read the complete, authoritative conductor prompt."""
    template = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    if template.count(QUESTION_MARKER) != 1:
        raise ValueError(
            f"{DEFAULT_PROMPT_PATH} must contain exactly one {QUESTION_MARKER!r} marker"
        )
    return template


def build_prompt(question: str, *, prompt_template: str | None = None) -> str:
    template = load_conductor_prompt() if prompt_template is None else prompt_template
    return template.replace(QUESTION_MARKER, question)


def build_conductor_prompt(question: str, model_registry: ModelRegistry) -> str:
    # The registry constrains decoding and execution; prompt content lives in
    # the prompt file, including its worker catalogue and embedded examples.
    return build_prompt(question)
