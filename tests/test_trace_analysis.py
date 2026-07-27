import json

import pytest

from theo_conductor.trace_analysis import (
    TraceDataset,
    TraceQuery,
    TraceRecord,
    error_category,
    main,
    workflow_to_graphviz,
)


def _write_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [
        {
            "rank": 0, "batch": 0, "sample": 0, "reward": 0.0, "question": "Question A",
            "plan": None, "worker_outputs": {}, "error": "Completion does not contain valid JSON: nope",
            "conductor_completion": "not json", "final_answer": None,
        },
        {
            "rank": 0, "batch": 0, "sample": 1, "reward": 0.2, "question": "Question A",
            "plan": {"task_type": "math", "difficulty": "easy", "workflow": [{"step_id": "oops"}]},
            "worker_outputs": {}, "error": "Final workflow step must have step_id 'final'", "final_answer": None,
        },
        {
            "rank": 0, "batch": 1, "sample": 0, "reward": 1.0, "question": "Question B",
            "plan": {"task_type": "science", "difficulty": "hard", "workflow": [{"step_id": "final"}]},
            "worker_outputs": {"final": {
                "model_id": "solver",
                "text": "FINAL: yes",
                "latency_ms": 2000,
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }}, "error": None, "final_answer": "yes",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n{bad json\n", encoding="utf-8")
    path.with_name("trace-token-counts.json").write_text(json.dumps({
        "configured_max_completion_tokens": 10, "counts": [10, 5, 8, 1],
    }), encoding="utf-8")
    return path


def test_error_category_matches_viewer_categories():
    assert error_category("Completion does not contain valid JSON: x") == "Malformed: no valid JSON"
    assert error_category("Completion does not contain valid JSON: x", completion_saturated=True) == "Malformed: truncated at output token limit"
    assert error_category("Completion does not contain valid JSON: x", completion="") == "Malformed: empty completion"
    assert error_category("Completion does not contain valid JSON: x", completion="plain prose") == "Malformed: prose only"
    assert error_category("Completion does not contain valid JSON: x", completion='{task_type: "math"}') == "Malformed: JSON-like syntax with unquoted keys"
    assert error_category("Completion does not contain valid JSON: x", completion='prefix {"workflow": [') == "Malformed: incomplete or unclosed JSON"
    assert error_category("Completion does not contain valid JSON: x", completion='prefix ["question"] suffix') == "Malformed: JSON fragments without a workflow"
    assert error_category("Completion does not contain valid JSON: x", completion='{} then {"workflow": []}') == "Malformed: valid workflow JSON embedded in extra text"
    assert error_category(None) == "No execution/validation error"


def test_dataset_summary_and_filtering(tmp_path):
    dataset = TraceDataset.load(_write_trace(tmp_path))

    summary = dataset.summary()
    assert summary["records"] == 3
    assert summary["mean_reward"] == pytest.approx(0.4)
    assert summary["reward_distribution"][0] == {"reward": 0, "count": 1, "fraction": pytest.approx(1 / 3)}
    assert summary["completion_tokens"]["saturated"] == 1
    assert summary["worker_performance"] == [{
        "model_id": "solver",
        "runs": 1,
        "latency_samples": 1,
        "mean_latency_ms": 2000,
        "p95_latency_ms": 2000,
        "mean_prompt_tokens": 20,
        "mean_completion_tokens": 10,
        "mean_total_tokens": 30,
        "mean_output_tokens_per_second": 5,
    }]
    assert summary["worker_timing"]["calls"] == 1
    assert summary["worker_timing"]["estimated_workflow_latency_ms"] == 2000
    assert summary["worker_timing"]["models"][0]["critical_path_share"] == 1
    assert {"category": "Malformed: truncated at output token limit", "count": 1} in summary["error_categories"]
    assert len(summary["malformed_jsonl_lines"]) == 1

    failed = dataset.query(TraceQuery(rewards={0.0, 0.2}, search="question a"))
    assert [record.record_id for record in failed] == ["0:1", "0:2"]


def test_worker_timing_uses_slowest_call_per_parallel_layer():
    records = [
        {
            "batch": 0,
            "reward": 1.0,
            "timestamp": "2026-01-01T00:00:10+00:00",
            "plan": {
                "workflow": [
                    {"step_id": "a", "access_list": ["question"]},
                    {"step_id": "b", "access_list": ["question"]},
                    {"step_id": "final", "access_list": ["a", "b"]},
                ]
            },
            "worker_outputs": {
                "a": {"model_id": "fast", "latency_ms": 1000},
                "b": {"model_id": "slow", "latency_ms": 3000},
                "final": {"model_id": "fast", "latency_ms": 2000},
            },
        },
        {
            "batch": 1,
            "reward": 1.0,
            "timestamp": "2026-01-01T00:00:20+00:00",
            "plan": {"workflow": [{"step_id": "final", "access_list": ["question"]}]},
            "worker_outputs": {
                "final": {"model_id": "slow", "latency_ms": 8000},
            },
        },
    ]
    dataset = TraceDataset(
        [
            TraceRecord(
                data=record,
                record_id=f"0:{index}",
                source="memory",
                line=index,
                error_category=error_category(None),
            )
            for index, record in enumerate(records, 1)
        ]
    )

    timing = dataset.summary()["worker_timing"]
    assert timing["total_call_latency_ms"] == 14_000
    assert timing["estimated_workflow_latency_ms"] == 13_000
    assert timing["parallelism_savings_ms"] == 1000
    assert timing["workflow_interval_share"] == pytest.approx(0.8)
    assert timing["models"][0]["model_id"] == "slow"
    assert timing["models"][0]["critical_path_latency_ms"] == 11_000


def test_plan_metrics_measure_dag_runtime_and_observed_parallelism():
    record = {
        "rank": 0,
        "batch": 4,
        "reward": 1.0,
        "plan": {
            "workflow": [
                {"step_id": "a", "model_id": "fast", "access_list": ["question"]},
                {"step_id": "b", "model_id": "slow", "access_list": ["question"]},
                {"step_id": "final", "model_id": "finalizer", "access_list": ["a", "b"]},
            ]
        },
        "worker_outputs": {
            "a": {"latency_ms": 1000},
            "b": {"latency_ms": 3000},
            "final": {"latency_ms": 2000},
        },
        "workflow_runtime": {
            "observed_wall_time_ms": 5500,
            "observed_peak_concurrency": 2,
        },
    }
    dataset = TraceDataset([
        TraceRecord(
            data=record,
            record_id="0:1",
            source="memory",
            line=1,
            error_category=error_category(None),
        )
    ])

    metrics = dataset.summary()["plan_metrics"][0]
    assert metrics["num_steps"] == 3
    assert metrics["num_edges"] == 2
    assert metrics["critical_path_steps"] == 2
    assert metrics["critical_path_runtime"] == 5
    assert metrics["total_worker_runtime"] == 6
    assert metrics["maximum_width"] == 2
    assert metrics["work_span_parallelism"] == pytest.approx(1.2)
    assert metrics["observed_wall_time"] == 5.5
    assert metrics["observed_peak_concurrency"] == 2
    assert metrics["realized_parallelism"] == pytest.approx(6 / 5.5)
    assert metrics["parallelism_utilization"] == pytest.approx(6 / 5.5 / 2)

    dot = workflow_to_graphviz(record["plan"])
    assert '"a" -> "final"' in dot
    assert '"b" -> "final"' in dot
    assert 'label="a\\nmodel: fast"' in dot


def test_conductor_performance_uses_batch_latency_once():
    records = []
    for index, completion_tokens in enumerate((10, 20)):
        records.append(
            TraceRecord(
                data={
                    "rank": 0,
                    "batch": 0,
                    "reward": 1.0,
                    "conductor_performance": {
                        "model_id": "planner",
                        "batch_latency_ms": 1000,
                        "batch_size": 2,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": completion_tokens,
                            "total_tokens": 100 + completion_tokens,
                        },
                    },
                },
                record_id=f"0:{index + 1}",
                source="memory",
                line=index + 1,
                error_category=error_category(None),
            )
        )

    [performance] = TraceDataset(records).summary()["conductor_performance"]
    assert performance["model_id"] == "planner"
    assert performance["runs"] == 2
    assert performance["latency_samples"] == 1
    assert performance["mean_prompt_tokens"] == 100
    assert performance["mean_completion_tokens"] == 15
    assert performance["mean_output_tokens_per_second"] == 30


def test_judge_performance_includes_kimi_throughput():
    record = TraceRecord(
        data={
            "rank": 0,
            "batch": 0,
            "reward": 1.0,
            "judge_model": "kimi-k2.6",
            "judge_performance": {
                "model_id": "kimi-k2.6",
                "latency_ms": 2000,
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                },
            },
        },
        record_id="0:1",
        source="memory",
        line=1,
        error_category=error_category(None),
    )

    [performance] = TraceDataset([record]).summary()["judge_performance"]
    assert performance["model_id"] == "kimi-k2.6"
    assert performance["role"] == "judge"
    assert performance["mean_latency_ms"] == 2000
    assert performance["mean_output_tokens_per_second"] == 10


def test_error_groups_and_question_rollouts(tmp_path):
    dataset = TraceDataset.load([_write_trace(tmp_path)])

    errors = dataset.errors(examples=1)
    assert [group["count"] for group in errors] == [1, 1]
    assert errors[0]["examples"][0]["record_id"].startswith("0:")

    questions = dataset.questions(min_rollouts=2)
    assert len(questions) == 1
    assert questions[0]["question"] == "Question A"
    assert questions[0]["reward_distribution"] == {"0": 1, "0.2": 1}
    assert dataset.questions(disagreement_only=True)[0]["question"] == "Question A"


def test_cli_emits_json_and_can_show_record(tmp_path, capsys):
    path = _write_trace(tmp_path)

    assert main(["list", str(path), "--reward", "1"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == 1
    record_id = listed["records"][0]["record_id"]

    assert main(["show", str(path), "--id", record_id]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["final_answer"] == "yes"


def test_strict_loading_rejects_malformed_line(tmp_path):
    with pytest.raises(ValueError, match="trace.jsonl:4"):
        TraceDataset.load([_write_trace(tmp_path)], strict=True)
