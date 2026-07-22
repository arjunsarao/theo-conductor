import asyncio
import json

from theo_conductor.benchmark import (
    bootstrap_accuracy_ci,
    extract_final_answer,
    judge_records,
    parse_judge_batch,
    run_benchmark,
    summarize_records,
)
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.schema import ModelResponse, ModelSpec


class AnswerClient:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return ModelResponse(
            text=f"Reasoning\nFINAL: {self.answer}",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=20,
        )


class JudgeClient:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["question"].split("\n", 1)[1])
        return ModelResponse(
            text=json.dumps(
                [
                    {"id": item["id"], "correct": correct, "reason": reason}
                    for item, (correct, reason) in zip(payload, self.verdicts, strict=True)
                ]
            )
        )


def test_extract_final_answer_requires_explicit_marker_and_uses_last_one():
    assert extract_final_answer("answer: 1") is None
    assert extract_final_answer("FINAL: 1\nrevision\nFinal answer: 2") == "2"


def test_parse_judge_batch_requires_every_requested_id():
    assert parse_judge_batch(
        '```json\n[{"id":"a","correct":true,"reason":"Equivalent."}]\n```', ["a"]
    ) == {"a": (True, "Equivalent.")}
    try:
        parse_judge_batch('[{"id":"wrong","correct":true,"reason":"Equivalent."}]', ["a"])
    except ValueError as exc:
        assert "did not match" in str(exc)
    else:
        raise AssertionError("mismatched judge ids should fail")
    try:
        parse_judge_batch('[{"id":"a","correct":"yes","reason":"Equivalent."}]', ["a"])
    except ValueError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("non-boolean judge verdict should fail")


def test_judge_records_batches_requests_and_makes_judge_authoritative():
    records = [
        {
            "model_id": "solver",
            "example_id": "a",
            "question": "What is log10(0.001)?",
            "gold_answer": "0.001 = 10^-3, so the answer is -3.",
            "reference_answer": "-3",
            "response": "Calculation. FINAL: -3",
            "extracted_answer": "-3",
            "correct": None,
            "error": None,
        },
        {
            "model_id": "solver",
            "example_id": "b",
            "question": "What is 2 + 2?",
            "gold_answer": "4",
            "reference_answer": "4",
            "response": "Calculation. FINAL: 5",
            "extracted_answer": "5",
            "correct": None,
            "error": None,
        },
    ]
    client = JudgeClient(
        [(True, "The answer matches exactly."), (False, "The candidate gives the wrong value.")]
    )

    asyncio.run(judge_records(records, client=client, batch_size=10))

    assert len(client.calls) == 1
    assert records[0]["judge_correct"] is True
    assert records[0]["correct"] is True
    assert records[0]["judge_reason"] == "The answer matches exactly."
    assert records[1]["judge_correct"] is False
    assert records[1]["correct"] is False


def test_run_benchmark_evaluates_every_model_on_same_rows_and_resumes(tmp_path):
    first = AnswerClient("4")
    second = AnswerClient("5")
    registry = ModelRegistry(
        [
            ModelSpec(model_idx="first", display_name="First", client=first),
            ModelSpec(model_idx="second", display_name="Second", client=second),
        ]
    )
    dataset = [
        {"id": "a", "question": "2+2?", "answer": "4", "subject": "math"},
        {"id": "b", "question": "2+3?", "answer": "5", "subject": "math"},
    ]
    path = tmp_path / "results.jsonl"

    records = asyncio.run(run_benchmark(registry=registry, dataset=dataset, results_path=path))
    resumed = asyncio.run(run_benchmark(registry=registry, dataset=dataset, results_path=path))

    assert len(records) == len(resumed) == 4
    assert first.calls == second.calls == 2
    assert {(row["model_id"], row["example_id"]) for row in records} == {
        ("first", "a"),
        ("first", "b"),
        ("second", "a"),
        ("second", "b"),
    }
    assert len(path.read_text().splitlines()) == 4
    assert all(json.loads(line)["response"] for line in path.read_text().splitlines())
    assert all(json.loads(line)["correct"] is None for line in path.read_text().splitlines())


def test_summary_reports_accuracy_failures_usage_and_subjects():
    records = [
        {
            "model_id": "solver",
            "display_name": "Solver",
            "subject": "physics",
            "correct": True,
            "error": None,
            "extracted_answer": "4",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "latency_ms": 20,
        },
        {
            "model_id": "solver",
            "display_name": "Solver",
            "subject": "physics",
            "correct": False,
            "error": None,
            "extracted_answer": None,
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "latency_ms": 40,
        },
    ]

    metrics = summarize_records(records, bootstrap_samples=100, seed=7)["models"]["solver"]

    assert metrics["accuracy"] == 0.5
    assert metrics["answer_extraction_failure_rate"] == 0.5
    assert metrics["mean_prompt_tokens"] == 15
    assert metrics["mean_latency_ms"] == 30
    assert metrics["by_subject"]["physics"]["accuracy"] == 0.5


def test_bootstrap_accuracy_ci_is_deterministic():
    assert bootstrap_accuracy_ci([True, False, True], samples=100, seed=3) == bootstrap_accuracy_ci(
        [True, False, True], samples=100, seed=3
    )
