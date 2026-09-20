from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .workflow_benchmark import load_results


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
    """Load the summary and records for its declared execution configuration(s)."""
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
    if not selected_hashes:
        raise ValueError(
            f"{summary_path} must declare execution_config_sha256 or "
            "execution_config_sha256s"
        )
    records, _ = load_results(results_path)
    selected = [
        record
        for record in records
        if str(record.get("execution_config_sha256") or "") in selected_hashes
    ]
    return summary, selected


def hle_run_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("error") is None]
    judged = [record for record in records if isinstance(record.get("correct"), bool)]
    correct = [record for record in judged if record.get("correct") is True]
    externally_judged = [
        record
        for record in records
        if isinstance(record.get("judge_correct"), bool) and record.get("error") is None
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
        "incorrect": sum(record.get("correct") is False and not record.get("judge_error") for record in completed),
        "clean_completed": sum(
            not record.get("judge_error") and not any(
                output.get("finish_reason") in ("length", "error") or output.get("error")
                for output in (record.get("worker_outputs") or {}).values()
                if isinstance(output, dict)
            )
            for record in completed
        ),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "scored": len(judged),
        "externally_judged": len(externally_judged),
        "correct": len(correct),
        "accuracy": len(correct) / len(judged) if judged else None,
        "completed_accuracy": len(correct) / len(completed) if completed else None,
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
