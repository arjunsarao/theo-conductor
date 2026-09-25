from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .workflow_benchmark import load_results


def result_outcome(record: dict[str, Any]) -> str:
    """Classify a result without treating infrastructure failures as incorrect."""
    if record.get("error") or record.get("judge_error"):
        return "failed"
    if record.get("correct") is True:
        return "correct"
    if record.get("correct") is False:
        return "incorrect"
    return "unjudged"


def discover_hle_direct_benchmarks(root: Path) -> dict[str, tuple[Path, Path]]:
    """Find direct-model HLE runs named ``*-benchmark`` beneath outputs."""
    output_root = root / "outputs"
    runs: dict[str, tuple[Path, Path]] = {}
    if not output_root.is_dir():
        return runs
    for run_dir in output_root.glob("*-benchmark"):
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"
        if summary_path.is_file() and results_path.is_file():
            runs[run_dir.name] = (summary_path, results_path)
    return runs


def discover_hle_benchmark_runs(root: Path) -> dict[str, tuple[Path, Path]]:
    """Find text-only HLE physics workflow benchmarks beneath outputs."""
    output_root = root / "outputs"
    runs: dict[str, tuple[Path, Path]] = {}
    if not output_root.is_dir():
        return runs
    for summary_path in output_root.rglob("summary.json"):
        results_path = summary_path.with_name("results.jsonl")
        if not results_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict) or not summary.get("plans"):
            continue
        if not summary.get("text_only"):
            continue
        models = summary.get("models") or {}
        if "theo-conductor" not in models and "worker_token_limit_mode" not in summary:
            continue
        subjects = (models.get("theo-conductor") or {}).get("by_subject") or {}
        if Path(str(summary["plans"])).name != "hle-physics" and set(subjects) != {"Physics"}:
            continue
        relative = summary_path.parent.relative_to(output_root)
        label = str(relative)
        runs[label] = (summary_path, results_path)
    return runs


def load_hle_benchmark(
    summary_path: Path,
    results_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load records for the declared configuration, or all records for legacy runs."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} must contain a JSON object")
    execution_hashes = summary.get("execution_config_sha256s")
    if execution_hashes is None:
        execution_hashes = [summary.get("execution_config_sha256")]
    if not isinstance(execution_hashes, list) or not execution_hashes:
        raise ValueError(
            f"{summary_path} must declare execution_config_sha256 or "
            "execution_config_sha256s"
        )
    selected_hashes = {
        str(execution_hash)
        for execution_hash in execution_hashes
        if execution_hash
    }
    records, _ = load_results(results_path)
    if not selected_hashes:
        return summary, records
    selected = [
        record
        for record in records
        if str(record.get("execution_config_sha256") or "") in selected_hashes
    ]
    return summary, selected



def pairwise_outcome_counts(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Count correctness overlap for questions present in both runs."""
    left = benchmark_record_map(left_records)
    right = benchmark_record_map(right_records)
    counts = {
        "Both correct": 0,
        "Left only": 0,
        "Right only": 0,
        "Both incorrect": 0,
        "Left failed": 0,
        "Right failed": 0,
        "Both failed": 0,
        "Unjudged": 0,
    }
    for key in left.keys() & right.keys():
        left_outcome = result_outcome(left[key])
        right_outcome = result_outcome(right[key])
        if left_outcome == "failed" and right_outcome == "failed":
            counts["Both failed"] += 1
            continue
        if left_outcome == "failed":
            counts["Left failed"] += 1
            continue
        if right_outcome == "failed":
            counts["Right failed"] += 1
            continue
        if "unjudged" in (left_outcome, right_outcome):
            counts["Unjudged"] += 1
            continue
        left_correct = left_outcome == "correct"
        right_correct = right_outcome == "correct"
        if left_correct and right_correct:
            counts["Both correct"] += 1
        elif left_correct:
            counts["Left only"] += 1
        elif right_correct:
            counts["Right only"] += 1
        else:
            counts["Both incorrect"] += 1
    return counts


def clarification_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten structured clarification tool calls with their workflow context."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for step_id, output in (record.get("worker_outputs") or {}).items():
            if not isinstance(output, dict):
                continue
            for tool_call in output.get("tool_calls") or []:
                if not isinstance(tool_call, dict) or tool_call.get("name") != "request_clarification":
                    continue
                arguments = tool_call.get("arguments") or {}
                result = tool_call.get("result") or {}
                usage = tool_call.get("usage") or {}
                rows.append(
                    {
                        "example_id": record.get("example_id"),
                        "benchmark_position": record.get("benchmark_position"),
                        "step_id": step_id,
                        "worker_model_id": output.get("model_id"),
                        "call_id": tool_call.get("call_id"),
                        "question": arguments.get("question"),
                        "reason": arguments.get("reason"),
                        "blocking": arguments.get("blocking") is True,
                        "impact_if_unanswered": arguments.get("impact_if_unanswered"),
                        "needed_by": arguments.get("needed_by"),
                        "recommended_option": arguments.get("recommended_option"),
                        "selected_option": result.get("selected_option"),
                        "answered": result.get("answered") is True,
                        "answer": result.get("answer"),
                        "answer_source": result.get("source"),
                        "is_error": tool_call.get("is_error") is True,
                        "error": result.get("error"),
                        "resolver_model_id": tool_call.get("model_id"),
                        "duration_ms": tool_call.get("duration_ms"),
                        "total_tokens": usage.get("total_tokens"),
                        "estimated_cost_usd": usage.get("estimated_cost_usd"),
                    }
                )
    return rows

def hle_run_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("error") is None]
    outcomes = [result_outcome(record) for record in records]
    judged = [
        record
        for record, outcome in zip(records, outcomes, strict=True)
        if outcome in ("correct", "incorrect")
    ]
    correct = [record for record in judged if result_outcome(record) == "correct"]
    externally_judged = [
        record
        for record in records
        if (
            isinstance(record.get("judge_correct"), bool)
            and record.get("error") is None
            and not record.get("judge_error")
        )
    ]
    costs = [
        float(record["estimated_cost_usd"])
        for record in records
        if isinstance(record.get("estimated_cost_usd"), (int, float))
    ]
    runtimes = [
        float(record["workflow_runtime_ms"])
        for record in completed
        if isinstance(record.get("workflow_runtime_ms"), (int, float))
    ]
    return {
        "workflows": len(records),
        "judge_failed": sum(bool(record.get("judge_error")) for record in completed),
        "incorrect": outcomes.count("incorrect"),
        "clean_completed": sum(
            not record.get("judge_error") and not any(
                output.get("finish_reason") in ("length", "error") or output.get("error")
                for output in (record.get("worker_outputs") or {}).values()
                if isinstance(output, dict)
            )
            for record in completed
        ),
        "completed": len(completed),
        "workflow_failed": len(records) - len(completed),
        "failed": outcomes.count("failed"),
        "scored": len(judged),
        "externally_judged": len(externally_judged),
        "correct": len(correct),
        "accuracy": len(correct) / len(judged) if judged else None,
        "completed_accuracy": len(correct) / len(judged) if judged else None,
        "total_cost_usd": sum(costs) if costs else None,
        "mean_cost_usd": mean(costs) if costs else None,
        "mean_runtime_ms": mean(runtimes) if runtimes else None,
        "missing_final": sum(
            record.get("error") is None and record.get("extracted_answer") is None
            for record in records
        ),
        "capped_steps": sum(
            output.get("finish_reason") == "length"
            for record in records
            for output in (record.get("worker_outputs") or {}).values()
            if isinstance(output, dict)
        ),
        "error_types": dict(
            Counter(
                str(record.get("error_type") or "unknown")
                for record in records
                if record.get("error") is not None
            ).most_common()
        ),
    }


def worker_step_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for step_id, output in (record.get("worker_outputs") or {}).items():
            if not isinstance(output, dict):
                continue
            usage = output.get("usage") or {}
            rows.append(
                {
                    "example_id": record.get("example_id"),
                    "step_id": step_id,
                    "model_id": output.get("model_id"),
                    "latency_ms": output.get("latency_ms"),
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                    "total_tokens": usage.get("total_tokens"),
                    "estimated_cost_usd": usage.get("estimated_cost_usd"),
                    "finish_reason": output.get("finish_reason"),
                }
            )
    return rows

def benchmark_record_key(record: dict[str, Any]) -> str:
    """Return the most stable available identity for a benchmark question."""
    for field in ("example_id", "question_sha256", "benchmark_position"):
        value = record.get(field)
        if value is not None and str(value):
            return str(value)
    return str(record.get("question") or "")


def benchmark_record_map(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index results by question, preferring a later retry when one exists."""
    return {benchmark_record_key(record): record for record in records}


def capability_comparison(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify question-level capability retention and movement between runs."""
    baseline = benchmark_record_map(baseline_records)
    candidate = benchmark_record_map(candidate_records)
    rows: list[dict[str, Any]] = []
    for key in baseline.keys() & candidate.keys():
        baseline_outcome = result_outcome(baseline[key])
        candidate_outcome = result_outcome(candidate[key])
        baseline_correct = baseline_outcome == "correct"
        candidate_correct = candidate_outcome == "correct"
        if baseline_correct and candidate_correct:
            movement = "Retained"
        elif candidate_correct:
            movement = "Gained"
        elif baseline_correct:
            movement = "Regressed"
        else:
            movement = "Unsolved"
        rows.append(
            {
                "key": key,
                "baseline": baseline[key],
                "candidate": candidate[key],
                "baseline_outcome": baseline_outcome,
                "candidate_outcome": candidate_outcome,
                "movement": movement,
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
        positions = (
            row["baseline"].get("benchmark_position"),
            row["candidate"].get("benchmark_position"),
        )
        position = next((value for value in positions if isinstance(value, int)), None)
        return (0, position) if position is not None else (1, row["key"])

    return sorted(rows, key=sort_key)
