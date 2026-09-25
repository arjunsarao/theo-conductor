import asyncio
import json

import pytest

import theo_conductor.workflow_benchmark as workflow_benchmark

from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelResponse, ModelSpec
from theo_conductor.workflow_benchmark import (
    _usage_totals,
    execution_config_hash,
    load_results,
    run_workflow_benchmark,
    select_text_only_plans,
    summarize_workflows,
)


class WorkflowClient:
    def __init__(self):
        self.calls = 0

    async def generate(self, *, question, **kwargs):
        self.calls += 1
        answer = question.rsplit(" ", 1)[-1]
        return ModelResponse(
            text=f"Reasoning for {question}\nFINAL: {answer}",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=3.0,
        )


def plan_records(count=10):
    return [
        {
            "dataset_id": f"hle-{index}",
            "dataset_index": index,
            "question": f"Return {index}",
            "gold_answer": str(index),
            "reference_answer": str(index),
            "answer_type": "exactMatch",
            "subject": "Math",
            "plan": {
                "task_type": "math",
                "difficulty": "easy",
                "question": f"Return {index}",
                "workflow": [
                    {
                        "step_id": "final",
                        "model_id": "worker",
                        "instruction": "Answer the question.",
                        "access_list": ["question"],
                    }
                ],
            },
            "error": None,
        }
        for index in range(count)
    ]


def test_runs_ten_workflows_and_resumes_without_duplicate_calls(tmp_path):
    client = WorkflowClient()
    registry = ModelRegistry(
        [
            ModelSpec(
                model_idx="worker",
                display_name="Worker",
                client=client,
                cost_per_1m_input_tokens=1.0,
                cost_per_1m_output_tokens=2.0,
            )
        ]
    )
    results_path = tmp_path / "results.jsonl"
    config_hash = execution_config_hash(
        config_bytes=b"models: []",
        max_worker_tokens=100,
        worker_temperature=0.0,
    )

    first = asyncio.run(
        run_workflow_benchmark(
            registry=registry,
            plan_records=plan_records(),
            results_path=results_path,
            execution_config_sha256=config_hash,
            concurrency=3,
            max_worker_tokens=100,
            worker_temperature=0.0,
        )
    )
    resumed = asyncio.run(
        run_workflow_benchmark(
            registry=registry,
            plan_records=plan_records(),
            results_path=results_path,
            execution_config_sha256=config_hash,
            concurrency=3,
            max_worker_tokens=100,
            worker_temperature=0.0,
        )
    )

    assert len(first) == len(resumed) == 10
    assert client.calls == 10
    assert len(results_path.read_text().splitlines()) == 10
    assert {record["extracted_answer"] for record in first} == {str(i) for i in range(10)}
    assert all(record["workflow_steps"] == 1 for record in first)
    assert all(record["total_tokens"] == 15 for record in first)
    assert all(record["estimated_cost_usd"] == 0.00002 for record in first)
    assert summarize_workflows(first, bootstrap_samples=10)["completed"] == 10


def test_judges_each_workflow_before_its_result_is_checkpointed(tmp_path):
    class JudgeClient:
        async def generate(self, **kwargs):
            # The immediate path grades exactly one finished workflow at a time.
            assert "Return 0" in kwargs["question"]
            return ModelResponse(
                text='[{"id":"batch-0-item-0","correct":true,"reason":"Matches gold."}]'
            )

    registry = ModelRegistry([ModelSpec(model_idx="worker", client=WorkflowClient())])
    path = tmp_path / "results.jsonl"
    plans = plan_records(1)
    # This is the compact format emitted by the stored frontier planner runs.
    plans[0]["plan"].pop("question")
    records = asyncio.run(run_workflow_benchmark(
        registry=registry,
        plan_records=plans,
        results_path=path,
        execution_config_sha256="immediate-judge",
        judge_client=JudgeClient(),
        judge_model="~deepseek/deepseek-flash-latest",
    ))

    persisted = json.loads(path.read_text())
    assert records[0]["judge_correct"] is True
    assert persisted["judge_correct"] is True
    assert persisted["correct"] is True


def test_worker_batching_pools_models_across_dependency_layers(tmp_path):
    class BatchClient:
        supports_batch = True

        def __init__(self):
            self.batch_sizes = []
            self.single_calls = 0

        async def generate(self, **kwargs):
            self.single_calls += 1
            raise AssertionError("native-batched client should not receive single calls")

        async def generate_batch(self, requests):
            self.batch_sizes.append(len(requests))
            return [
                ModelResponse(
                    text=(
                        "analysis"
                        if not request["context"]
                        else f"FINAL: {request['question'].rsplit(' ', 1)[-1]}"
                    ),
                    usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                )
                for request in requests
            ]

    client = BatchClient()
    registry = ModelRegistry([ModelSpec(model_idx="worker", client=client)])
    records = plan_records(2)
    for record in records:
        record["plan"]["workflow"] = [
            {
                "step_id": "analysis",
                "model_id": "worker",
                "instruction": "Analyze.",
                "access_list": ["question"],
            },
            {
                "step_id": "final",
                "model_id": "worker",
                "instruction": "Answer.",
                "access_list": ["question", "analysis"],
            },
        ]

    results = asyncio.run(run_workflow_benchmark(
        registry=registry,
        plan_records=records,
        results_path=tmp_path / "results.jsonl",
        execution_config_sha256="batched",
        batch_workers=True,
    ))

    assert client.batch_sizes == [2, 2]
    assert client.single_calls == 0
    assert [record["extracted_answer"] for record in results] == ["0", "1"]
    assert all(record["workflow_steps"] == 2 for record in results)


def test_select_text_only_plans_filters_ids_and_restores_subjects():
    selected = select_text_only_plans(
        plan_records(3),

        [
            {"id": "hle-0", "subject": "Math"},
            {"id": "hle-2", "subject": "Physics"},
        ],
    )

    assert [record["dataset_id"] for record in selected] == ["hle-0", "hle-2"]
    assert [record["subject"] for record in selected] == ["Math", "Physics"]
    assert all(record["is_multimodal"] is False for record in selected)


def test_model_context_limit_uses_each_workers_own_configured_length(tmp_path):
    class LimitClient:
        def __init__(self, answer):
            self.answer = answer
            self.max_tokens = []

        async def generate(self, *, max_tokens, **kwargs):
            self.max_tokens.append(max_tokens)
            return ModelResponse(text=f"FINAL: {self.answer}")

    first = LimitClient("intermediate")
    second = LimitClient("done")
    registry = ModelRegistry([
        ModelSpec(model_idx="first", client=first, context_length=111),
        ModelSpec(model_idx="second", client=second, context_length=222),
    ])
    records = plan_records(1)
    records[0]["plan"]["workflow"] = [
        {
            "step_id": "analysis",
            "model_id": "first",
            "instruction": "Analyze.",
            "access_list": ["question"],
        },
        {
            "step_id": "final",
            "model_id": "second",
            "instruction": "Answer.",
            "access_list": ["question", "analysis"],
        },
    ]

    result = asyncio.run(run_workflow_benchmark(
        registry=registry,
        plan_records=records,
        results_path=tmp_path / "results.jsonl",
        execution_config_sha256="per-model",
        max_worker_tokens=None,
    ))

    assert result[0]["error"] is None
    assert first.max_tokens == [111]
    assert second.max_tokens == [222]


def test_model_output_limits_use_each_workers_benchmark_budget(tmp_path):
    class LimitClient:
        def __init__(self):
            self.max_tokens = []

        async def generate(self, *, max_tokens, **kwargs):
            self.max_tokens.append(max_tokens)
            return ModelResponse(text="FINAL: done")

    first = LimitClient()
    second = LimitClient()
    registry = ModelRegistry([
        ModelSpec(
            model_idx="first",
            client=first,
            context_length=111_000,
            max_output_tokens=12_288,
        ),
        ModelSpec(
            model_idx="second",
            client=second,
            context_length=222_000,
            max_output_tokens=40_960,
        ),
    ])
    records = plan_records(1)
    records[0]["plan"]["workflow"] = [
        {
            "step_id": "analysis",
            "model_id": "first",
            "instruction": "Analyze.",
            "access_list": ["question"],
        },
        {
            "step_id": "final",
            "model_id": "second",
            "instruction": "Answer.",
            "access_list": ["question", "analysis"],
        },
    ]

    asyncio.run(run_workflow_benchmark(
        registry=registry,
        plan_records=records,
        results_path=tmp_path / "results.jsonl",
        execution_config_sha256="model-output",
        max_worker_tokens=None,
        use_model_output_limits=True,
    ))

    assert first.max_tokens == [12_288]
    assert second.max_tokens == [40_960]


def test_model_context_limit_is_mutually_exclusive_with_fixed_cap(tmp_path):
    required = [
        "--plans", str(tmp_path / "plans.jsonl"),
        "--config", str(tmp_path / "models.yaml"),
        "--output-dir", str(tmp_path / "output"),
    ]
    args = workflow_benchmark.parse_args(required + ["--use-model-context-limit"])
    assert args.use_model_context_limit is True

    default_args = workflow_benchmark.parse_args(required)
    assert default_args.max_worker_tokens is None
    assert default_args.use_model_context_limit is False

    with pytest.raises(SystemExit):
        workflow_benchmark.parse_args(
            required + ["--max-worker-tokens", "100", "--use-model-context-limit"]
        )


def test_invalid_plan_is_checkpointed_as_an_item_failure(tmp_path):
    records = plan_records(1)
    records[0]["plan"] = None
    records[0]["error"] = "ConductorParseError: bad JSON"
    path = tmp_path / "results.jsonl"

    completed = asyncio.run(
        run_workflow_benchmark(
            registry=ModelRegistry([]),
            plan_records=records,
            results_path=path,
            execution_config_sha256="config",
        )
    )

    assert len(completed) == 1
    assert completed[0]["error_type"] == "ValueError"
    assert "Plan unavailable" in completed[0]["error"]
    assert len(load_results(path)[0]) == 1


def test_execution_configuration_changes_create_a_new_result(tmp_path):
    client = WorkflowClient()
    registry = ModelRegistry([ModelSpec(model_idx="worker", client=client)])
    path = tmp_path / "results.jsonl"

    for config_hash in ("first", "second"):
        asyncio.run(
            run_workflow_benchmark(
                registry=registry,
                plan_records=plan_records(1),
                results_path=path,
                execution_config_sha256=config_hash,
            )
        )

    records, keys = load_results(path)
    assert client.calls == 2
    assert len(records) == len(keys) == 2
    assert {record["execution_config_sha256"] for record in records} == {"first", "second"}


def test_retry_failures_replaces_latest_result_without_repeating_successes(tmp_path):
    class RecoveringClient(WorkflowClient):
        async def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary outage")
            return ModelResponse(text="FINAL: 0")

    client = RecoveringClient()
    registry = ModelRegistry([ModelSpec(model_idx="worker", client=client)])
    path = tmp_path / "results.jsonl"
    kwargs = {
        "registry": registry,
        "plan_records": plan_records(1),
        "results_path": path,
        "execution_config_sha256": "config",
    }

    failed = asyncio.run(run_workflow_benchmark(**kwargs))
    recovered = asyncio.run(run_workflow_benchmark(**kwargs, retry_failures=True))

    assert failed[0]["error_type"] == "RuntimeError"
    assert recovered[0]["error"] is None
    assert client.calls == 2
    assert len(path.read_text().splitlines()) == 2
    assert len(load_results(path)[0]) == 1


def test_cli_judge_verdicts_are_persisted_to_results(tmp_path, monkeypatch):
    plans = tmp_path / "plans.jsonl"
    plans.write_text(json.dumps(plan_records(1)[0]) + "\n")
    config = tmp_path / "models.yaml"
    config.write_text("models: []\n")
    output = tmp_path / "output"

    async def fake_run(**kwargs):
        record = {
            "model_id": "theo-conductor",
            "display_name": "Theo Conductor",
            "example_id": "hle-0",
            "subject": "Math",
            "question": "Return 0",
            "gold_answer": "0",
            "reference_answer": "0",
            "plan_sha256": "plan",
            "execution_config_sha256": kwargs["execution_config_sha256"],
            "response": "FINAL: 0",
            "extracted_answer": "0",
            "correct": None,
            "error": None,
            "worker_outputs": {},
            "workflow_runtime_ms": 1.0,
            "workflow_steps": 1,
            "estimated_cost_usd": 0.0,
            "latency_ms": 1.0,
        }
        kwargs["results_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["results_path"].write_text(json.dumps(record) + "\n")
        return [record]

    async def fake_judge(records, *, all_records, results_path, **kwargs):
        assert records[0] is all_records[0]
        records[0].update(correct=True, judge_correct=True, judge_model="judge", judge_error=None)
        workflow_benchmark.write_results_atomic(results_path, all_records)
        return records

    monkeypatch.setattr(workflow_benchmark, "run_workflow_benchmark", fake_run)
    monkeypatch.setattr(workflow_benchmark, "judge_records_with_checkpoints", fake_judge)

    asyncio.run(workflow_benchmark.async_main([
        "--plans", str(plans),
        "--config", str(config),
        "--output-dir", str(output),
        "--include-multimodal",
        "--no-judge-immediately",
        "--bootstrap-samples", "10",
    ]))

    persisted = json.loads((output / "results.jsonl").read_text())
    assert persisted["correct"] is True


def test_tool_capable_workers_use_iterative_path_when_batching_requested(tmp_path):
    class ToolBatchClient:
        supports_batch = True

        def __init__(self):
            self.single_calls = []

        async def generate(self, **kwargs):
            self.single_calls.append(kwargs)
            return ModelResponse(text="FINAL: 0")

        async def generate_batch(self, requests):
            raise AssertionError("tool-capable workers must use the iterative path")

    client = ToolBatchClient()
    registry = ModelRegistry([
        ModelSpec(model_idx="worker", client=client, supports_tools=True),
    ])

    results = asyncio.run(run_workflow_benchmark(
        registry=registry,
        plan_records=plan_records(1),
        results_path=tmp_path / "results.jsonl",
        execution_config_sha256="tools-default",
        batch_workers=True,
    ))

    assert results[0]["extracted_answer"] == "0"
    assert len(client.single_calls) == 1
    assert client.single_calls[0]["tools"][0]["function"]["name"] == (
        "request_clarification"
    )


def test_usage_totals_include_model_backed_tool_calls():
    totals = _usage_totals({
        "final": {
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "tool_calls": [{
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "estimated_cost_usd": 0.00035,
                },
            }],
        },
    })

    assert totals["prompt_tokens"] == 12
    assert totals["completion_tokens"] == 6
    assert totals["total_tokens"] == 18
    assert totals["estimated_cost_usd"] == pytest.approx(0.00035)
