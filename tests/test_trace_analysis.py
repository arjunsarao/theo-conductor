import json

import pytest

from theo_conductor.trace_analysis import TraceDataset, TraceQuery, error_category, main


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
            "worker_outputs": {"final": {"text": "FINAL: yes"}}, "error": None, "final_answer": "yes",
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
    assert error_category(None) == "No execution/validation error"


def test_dataset_summary_and_filtering(tmp_path):
    dataset = TraceDataset.load(_write_trace(tmp_path))

    summary = dataset.summary()
    assert summary["records"] == 3
    assert summary["mean_reward"] == pytest.approx(0.4)
    assert summary["reward_distribution"][0] == {"reward": 0, "count": 1, "fraction": pytest.approx(1 / 3)}
    assert summary["completion_tokens"]["saturated"] == 1
    assert {"category": "Malformed: truncated at output token limit", "count": 1} in summary["error_categories"]
    assert len(summary["malformed_jsonl_lines"]) == 1

    failed = dataset.query(TraceQuery(rewards={0.0, 0.2}, search="question a"))
    assert [record.record_id for record in failed] == ["0:1", "0:2"]


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
