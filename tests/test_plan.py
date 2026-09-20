import asyncio
import json

from theo_conductor.models.registry import ModelRegistry
from theo_conductor.plan import (
    generate_plans,
    load_plan_records,
    merge_shards,
    run_cli,
    summarize_records,
)
from theo_conductor.schema import ModelResponse, ModelSpec
from theo_conductor.trace_analysis import TraceDataset


class PlanningClient:
    def __init__(self, *, invalid=False):
        self.calls = 0
        self.invalid = invalid
        self.questions = []

    async def generate(self, **kwargs):
        self.calls += 1
        self.questions.append(kwargs["question"])
        text = "not json" if self.invalid else json.dumps(
            {
                "task_type": "general",
                "difficulty": "hard",
                "workflow": [
                    {
                        "step_id": "final",
                        "model_id": "worker",
                        "instruction": "Solve and return FINAL: <answer>.",
                        "access_list": ["question"],
                    }
                ],
            }
        )
        return ModelResponse(text=text, usage={"total_tokens": 12}, latency_ms=4.0)


def registry():
    return ModelRegistry([ModelSpec(model_idx="worker", client=PlanningClient())])


def rows(count=2):
    return [
        {
            "id": f"hle-{index}",
            "question": f"Question {index}",
            "answer": str(index),
            "answer_type": "exactMatch",
            "reference_answer": f"Reason {index}",
        }
        for index in range(count)
    ]


def test_generate_plans_is_valid_and_resumable(tmp_path):
    client = PlanningClient()
    output = tmp_path / "plans.jsonl"

    first = asyncio.run(
        generate_plans(
            rows=rows(),
            registry=registry(),
            client=client,
            conductor_model="conductor",
            output_path=output,
            concurrency=2,
            attempts=1,
        )
    )
    second = asyncio.run(
        generate_plans(
            rows=rows(),
            registry=registry(),
            client=client,
            conductor_model="conductor",
            output_path=output,
            concurrency=2,
            attempts=1,
        )
    )

    records = load_plan_records(output)
    assert first["generated"] == 2
    assert second["generated"] == 0
    assert second["resumed"] == 2
    assert client.calls == 2
    assert client.questions == ["", ""]
    assert [record["dataset_id"] for record in records] == ["hle-0", "hle-1"]
    assert records[0]["plan"]["question"] == "Question 0"
    assert records[0]["worker_outputs"] == {}
    assert TraceDataset.load(output).summary()["parsed_plans"] == 2


def test_generate_plans_persists_item_failure(tmp_path):
    output = tmp_path / "plans.jsonl"
    summary = asyncio.run(
        generate_plans(
            rows=rows(1),
            registry=registry(),
            client=PlanningClient(invalid=True),
            conductor_model="conductor",
            output_path=output,
            attempts=1,
        )
    )

    record = load_plan_records(output)[0]
    assert summary["invalid"] == 1
    assert record["plan"] is None
    assert record["error_type"] == "ConductorParseError"


def test_merge_and_inspection_cli(tmp_path):
    for shard_index in range(2):
        output = tmp_path / f"plans-shard-{shard_index:05d}.jsonl"
        asyncio.run(
            generate_plans(
                rows=rows(4),
                registry=registry(),
                client=PlanningClient(),
                conductor_model="conductor",
                output_path=output,
                shard_index=shard_index,
                shard_count=2,
                attempts=1,
            )
        )

    merged = merge_shards(tmp_path, expected_shards=2)

    assert merged["records"] == 4
    assert merged["valid"] == 4
    assert (tmp_path / "plans.jsonl").is_file()
    assert run_cli(["summary", str(tmp_path)])["records"] == 4
    shown = run_cli(["show", str(tmp_path), "--id", "hle-2"])
    assert shown["question"] == "Question 2"
    listed = run_cli(["list", str(tmp_path), "--valid-only", "--limit", "2"])
    assert len(listed) == 2
    assert summarize_records(load_plan_records(tmp_path / "plans.jsonl"))["models"] == {"worker": 4}
