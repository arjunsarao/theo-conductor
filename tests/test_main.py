import asyncio
import json

from theo_conductor.main import create_task, load_task, main, parse_args
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelResponse, ModelSpec


class ConductorClient:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ModelResponse(
            text=json.dumps(
                {
                    "task_type": "general",
                    "difficulty": "easy",
                    "workflow": [
                        {
                            "step_id": "final",
                            "model_id": "planner",
                            "instruction": "Answer clearly.",
                            "access_list": ["question"],
                        }
                    ],
                }
            )
        )


def test_parse_args_accepts_task_and_config_dir(tmp_path):
    args = parse_args(["workflow.json", "--config-dir", str(tmp_path), "--no-color"])

    assert args.task.name == "workflow.json"
    assert args.config_dir == tmp_path
    assert args.no_color is True


def test_parse_args_accepts_question_and_conductor_model():
    args = parse_args(["--question", "What is 2 + 2?", "--conductor-model", "planner"])

    assert args.question == "What is 2 + 2?"
    assert args.conductor_model == "planner"


def test_load_task(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(
        json.dumps(
            {
                "task_type": "test",
                "difficulty": "easy",
                "question": "What is 1 + 1?",
                "workflow": [
                    {
                        "step_id": "final",
                        "model_id": "fake",
                        "instruction": "Answer the question.",
                        "access_list": ["question"],
                    }
                ],
            }
        )
    )

    task = load_task(path)

    assert task.question == "What is 1 + 1?"
    assert task.workflow[0].step_id == "final"


def test_create_task_generates_and_parses_workflow():
    client = ConductorClient()
    registry = ModelRegistry([ModelSpec(model_idx="planner", client=client)])

    task = asyncio.run(create_task("What is 2 + 2?", registry, "planner"))

    assert task.question == "What is 2 + 2?"
    assert task.workflow[0].model_id == "planner"
    response_format = client.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["workflow"]["minItems"] == 1
    assert schema["properties"]["workflow"]["maxItems"] == 5
    model_id_schema = schema["properties"]["workflow"]["items"]["properties"]["model_id"]
    assert model_id_schema == {"enum": ["planner"]}


def test_main_reports_missing_task(capsys):
    assert main(["does-not-exist.json"]) == 1
    assert "error:" in capsys.readouterr().err
