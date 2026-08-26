import json

import pytest

from theo_conductor.models.fake import FakeModelClient
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prompt import (
    DEFAULT_EXAMPLES_DIR,
    TASK_TYPES,
    build_conductor_json_schema,
    build_conductor_prompt,
    build_default_examples,
)
from theo_conductor.schema import ModelSpec


def _registry() -> ModelRegistry:
    return ModelRegistry([ModelSpec(model_idx="solver", client=FakeModelClient("solver"))])


def test_default_examples_load_every_json_file_in_filename_order(tmp_path):
    (tmp_path / "second.json").write_text(json.dumps({"example": 2}), encoding="utf-8")
    (tmp_path / "first.json").write_text(json.dumps({"example": 1}), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    examples = build_default_examples(_registry(), examples_dir=tmp_path)

    assert examples == [
        '### First\n\n```json\n{\n  "example": 1\n}\n```',
        '### Second\n\n```json\n{\n  "example": 2\n}\n```',
    ]


def test_default_prompt_contains_all_repository_examples():
    examples = build_default_examples(_registry())
    prompt = build_conductor_prompt("New question", _registry())

    assert len(examples) == 6
    assert all(example in prompt for example in examples)
    assert "### Best Of N\n\n```json" in prompt
    assert sum(example.count("```json") for example in examples) == 6
    assert sum(example.count("\n```") for example in examples) == 12


def test_default_examples_reject_invalid_json(tmp_path):
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON conductor example"):
        build_default_examples(_registry(), examples_dir=tmp_path)


def test_default_examples_reject_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No JSON conductor examples"):
        build_default_examples(_registry(), examples_dir=tmp_path)


def test_repository_examples_match_frontier_catalogue_and_exact_schema():
    registry = ModelRegistry.from_yaml_file("configs/worker_pool_frontier.yaml")
    valid_model_ids = set(registry.model_ids())

    for path in sorted(DEFAULT_EXAMPLES_DIR.glob("*.json")):
        example = json.loads(path.read_text(encoding="utf-8"))
        assert set(example) == {"task_type", "difficulty", "workflow"}, path
        assert example["task_type"] in TASK_TYPES, path
        assert example["difficulty"] in {"easy", "medium", "hard"}, path
        assert 1 <= len(example["workflow"]) <= 7, path

        earlier_steps = {"question"}
        for step in example["workflow"]:
            assert set(step) == {"step_id", "model_id", "instruction", "access_list"}, path
            assert isinstance(step["step_id"], str) and step["step_id"], path
            assert isinstance(step["model_id"], str), path
            assert step["model_id"] in valid_model_ids, path
            assert isinstance(step["instruction"], str) and step["instruction"], path
            assert set(step["access_list"]) <= earlier_steps, path
            assert step["step_id"] not in earlier_steps, path
            earlier_steps.add(step["step_id"])


def test_repository_examples_do_not_request_unavailable_tools():
    tool_requests = ("use python", "execute code", "browse", "search the web", "inspect files")

    for path in sorted(DEFAULT_EXAMPLES_DIR.glob("*.json")):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(request in text for request in tool_requests), path


def test_hle_example_does_not_precommit_curvature_or_mass_conventions():
    text = (DEFAULT_EXAMPLES_DIR / "hle.json").read_text(encoding="utf-8")

    assert "positive Einstein curvature" in text
    assert "AdS" not in text
    assert "m_n^2 =" not in text
    assert "zero mode should exist" not in text


def test_conductor_schema_matches_prompt_contract():
    schema = build_conductor_json_schema(_registry())

    assert schema["properties"]["task_type"] == {"type": "string", "enum": TASK_TYPES}
    assert schema["properties"]["workflow"]["maxItems"] == 7
    assert schema["properties"]["workflow"]["items"]["properties"]["model_id"] == {
        "type": "string",
        "enum": ["solver"],
    }


def test_worker_catalogue_includes_routing_metadata():
    registry = ModelRegistry(
        [
            ModelSpec(
                model_idx="solver",
                display_name="Solver",
                role="primary reasoner",
                best_for="hard derivations",
                useful_for="independent checks",
                routing_note="prefer for delicate work",
                client=FakeModelClient("solver"),
            )
        ]
    )

    prompt = build_conductor_prompt("Question", registry, examples=["Example"])

    assert "  - role: primary reasoner" in prompt
    assert "  - best_for: hard derivations" in prompt
    assert "  - useful_for: independent checks" in prompt
    assert "  - routing_note: prefer for delicate work" in prompt
