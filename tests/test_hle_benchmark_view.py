import json

from theo_conductor.hle_benchmark_view import (
    discover_hle_benchmark_runs,
    hle_run_metrics,
    load_hle_benchmark,
    worker_step_rows,
)


def _record(example_id, *, execution_hash="current", correct=True, error=None):
    return {
        "model_id": "theo-conductor",
        "example_id": example_id,
        "plan_sha256": f"plan-{example_id}",
        "execution_config_sha256": execution_hash,
        "question": f"Question {example_id}",
        "correct": correct,
        "judge_correct": correct,
        "error": error,
        "error_type": "APIStatusError" if error else None,
        "extracted_answer": "answer" if not error else None,
        "estimated_cost_usd": 0.25 if not error else None,
        "workflow_runtime_ms": 2000 if not error else None,
        "worker_outputs": {
            "final": {
                "model_id": "worker",
                "latency_ms": 1000,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "estimated_cost_usd": 0.25,
                },
            }
        } if not error else {},
    }


def test_discovers_and_loads_only_current_hle_execution(tmp_path):
    run = tmp_path / "outputs" / "hle-benchmark"
    run.mkdir(parents=True)
    summary = {
        "plans": "outputs/hle-physics",
        "text_only": True,
        "execution_config_sha256": "current",
        "worker_token_limit_mode": "model_output_limits",
        "models": {"theo-conductor": {}},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    records = [_record("a"), _record("old", execution_hash="old")]
    (run / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    discovered = discover_hle_benchmark_runs(tmp_path)
    loaded_summary, loaded_records = load_hle_benchmark(*discovered["hle-benchmark"])

    assert loaded_summary == summary
    assert [record["example_id"] for record in loaded_records] == ["a"]


def test_loads_merged_run_with_multiple_execution_configurations(tmp_path):
    run = tmp_path / "outputs" / "hle-physics-merged"
    run.mkdir(parents=True)
    summary = {
        "plans": "outputs/hle-physics",
        "text_only": True,
        "execution_config_sha256": "mixed",
        "execution_config_sha256s": ["base", "rerun"],
        "worker_token_limit_mode": "model_output_limits",
        "models": {"theo-conductor": {}},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    records = [
        _record("base-result", execution_hash="base"),
        _record("rerun-result", execution_hash="rerun"),
        _record("unrelated", execution_hash="other"),
    ]
    (run / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    loaded_summary, loaded_records = load_hle_benchmark(
        run / "summary.json", run / "results.jsonl"
    )

    assert loaded_summary == summary
    assert [record["example_id"] for record in loaded_records] == [
        "base-result",
        "rerun-result",
    ]


def test_discovery_excludes_non_physics_and_multimodal_hle_runs(tmp_path):
    output_root = tmp_path / "outputs"
    for name, plans, text_only in (
        ("all-hle", "outputs/hle-plans", True),
        ("multimodal-physics", "outputs/hle-physics", False),
    ):
        run = output_root / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({
            "plans": plans,
            "text_only": text_only,
            "execution_config_sha256": "current",
            "worker_token_limit_mode": "model_output_limits",
            "models": {"theo-conductor": {}},
        }))
        (run / "results.jsonl").write_text(json.dumps(_record(name)) + "\n")

    assert discover_hle_benchmark_runs(tmp_path) == {}


def test_discovery_recognizes_physics_metadata_with_custom_plan_directory(tmp_path):
    for name, subjects in (
        ("hle-physics-flash-eval-21504", {"Physics": {"questions": 202}}),
        ("mixed-subjects", {"Physics": {}, "Math": {}}),
    ):
        run = tmp_path / "outputs" / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({
            "plans": "outputs/hle-plans-21504",
            "text_only": True,
            "execution_config_sha256": "current",
            "models": {"theo-conductor": {"by_subject": subjects}},
        }))
        (run / "results.jsonl").write_text(json.dumps(_record(name)) + "\n")

    assert set(discover_hle_benchmark_runs(tmp_path)) == {"hle-physics-flash-eval-21504"}


def test_metrics_distinguish_failures_and_flatten_worker_steps():
    records = [
        _record("correct"),
        _record("incorrect", correct=False),
        _record("failed", correct=False, error="APIStatusError: unavailable"),
    ]

    metrics = hle_run_metrics(records)
    steps = worker_step_rows(records)

    assert metrics["workflows"] == 3
    assert metrics["completed"] == 2
    assert metrics["failed"] == 1
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 1 / 3
    assert metrics["completed_accuracy"] == 1 / 2
    assert metrics["externally_judged"] == 2
    assert metrics["total_cost_usd"] == 0.5
    assert metrics["error_types"] == {"APIStatusError": 1}
    assert len(steps) == 2
    assert steps[0]["completion_tokens"] == 20


def test_metrics_separate_judge_failures_and_exclude_caps_and_worker_errors():
    records = [_record(str(i), correct=False) for i in range(5)]
    records[1].update(judge_error="Judge unavailable", judge_correct=None)
    records[2]["worker_outputs"]["final"]["finish_reason"] = "length"
    records[3]["worker_outputs"]["final"]["finish_reason"] = "error"
    records[4].update(error="Workflow failed", judge_error="Judge unavailable")

    metrics = hle_run_metrics(records)

    assert metrics["completed"] == 4
    assert metrics["judge_failed"] == 1
    assert metrics["incorrect"] == 3
    assert metrics["failed"] == 1
    assert metrics["clean_completed"] == 1
