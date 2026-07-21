import asyncio
import json

from theo_conductor.benchmark import (
    bootstrap_accuracy_ci,
    extract_final_answer,
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


def test_extract_final_answer_requires_explicit_marker_and_uses_last_one():
    assert extract_final_answer("answer: 1") is None
    assert extract_final_answer("FINAL: 1\nrevision\nFinal answer: 2") == "2"


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
