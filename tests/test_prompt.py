import json
import re

import pytest

from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prompt import (
    DEFAULT_PROMPT_PATH,
    QUESTION_MARKER,
    TASK_TYPES,
    build_conductor_json_schema,
    build_conductor_prompt,
    load_conductor_prompt,
)
from theo_conductor.schema import ModelSpec


def _registry() -> ModelRegistry:
    return ModelRegistry([ModelSpec(model_idx="solver", client=FakeModelClient("solver"))])


def test_prompt_preserves_fulltext_and_substitutes_only_question():
    question = "Find $x$ in {x: 2}. <USER QUESTION>"
    template = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")

    assert build_conductor_prompt(question, _registry()) == template.replace(QUESTION_MARKER, question)


@pytest.mark.parametrize("template", ["No marker", "<USER QUESTION> <USER QUESTION>"])
def test_prompt_rejects_missing_or_duplicate_question_marker(tmp_path, monkeypatch, template):
    path = tmp_path / "prompt.md"
    path.write_text(template, encoding="utf-8")
    monkeypatch.setattr("theo_conductor.prompt.DEFAULT_PROMPT_PATH", path)

    with pytest.raises(ValueError, match="must contain exactly one"):
        load_conductor_prompt()


def test_embedded_examples_match_frontier_catalogue_and_exact_schema():
    registry = ModelRegistry.from_yaml_file("configs/worker_pool_frontier.yaml")
    template = load_conductor_prompt()
    catalogue = set(re.findall(r'^- model_id="([^"]+)"', template, re.MULTILINE))
    assert catalogue == set(registry.model_ids())
    examples_section = template.split("## Examples", 1)[1]
    examples = re.findall(r"```json\n(.*?)\n```", examples_section, re.DOTALL)
    assert len(examples) == 7

    for text in examples:
        example = json.loads(text)
        assert set(example) == {"task_type", "difficulty", "workflow"}
        assert example["task_type"] in TASK_TYPES
        assert example["difficulty"] in {"easy", "medium", "hard"}
        assert 1 <= len(example["workflow"]) <= 7
        earlier_steps = {"question"}
        for step in example["workflow"]:
            assert set(step) == {"step_id", "model_id", "instruction", "access_list"}
            assert isinstance(step["step_id"], str) and step["step_id"]
            assert step["model_id"] in catalogue
            assert isinstance(step["instruction"], str) and step["instruction"]
            assert set(step["access_list"]) <= earlier_steps
            assert step["step_id"] not in earlier_steps
            earlier_steps.add(step["step_id"])


def test_conductor_schema_matches_prompt_contract():
    schema = build_conductor_json_schema(_registry())

    assert schema["properties"]["task_type"] == {"type": "string", "enum": TASK_TYPES}
    assert schema["properties"]["workflow"]["maxItems"] == 7
    assert schema["properties"]["workflow"]["items"]["properties"]["model_id"] == {
        "type": "string",
        "enum": ["solver"],
    }
