#!/usr/bin/env python3
"""Aggregate independently judged HLE-Diamond runs into empirical mean@k/pass@k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any
from theo_conductor.jsonio import read_jsonl as load
from theo_conductor.evaluation_stats import repeated_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-samples", type=int, default=61)
    parser.add_argument("--model", default="openai/gpt-6-astra")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [load(path) for path in args.results]
    if not runs:
        raise ValueError("At least one results file is required")
    expected_ids = [str(row["example_id"]) for row in runs[0]]
    if len(expected_ids) != args.expect_samples or len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"Expected {args.expect_samples} unique examples in the first run")

    by_run: list[dict[str, dict[str, Any]]] = []
    for path, rows in zip(args.results, runs, strict=True):
        indexed = {str(row["example_id"]): row for row in rows}
        if set(indexed) != set(expected_ids) or len(rows) != args.expect_samples:
            raise ValueError(f"Mismatched example set in {path}")
        if any(not isinstance(row.get("correct"), bool) for row in rows):
            raise ValueError(f"Unjudged records remain in {path}")
        if any(row.get("judge_error") for row in rows):
            raise ValueError(f"Judge errors remain in {path}; rejudge before aggregating")
        by_run.append(indexed)

    k = len(by_run)
    per_question = []
    for example_id in expected_ids:
        outcomes = [run[example_id]["correct"] for run in by_run]
        per_question.append({
            "example_id": example_id,
            "correct_runs": sum(outcomes),
            "attempts": k,
            "mean_at_k": mean(outcomes),
            "pass_at_k": any(outcomes),
        })

    run_summaries = []
    for index, (path, run) in enumerate(zip(args.results, by_run, strict=True), start=1):
        outcomes = [run[example_id]["correct"] for example_id in expected_ids]
        run_summaries.append({
            "run": index,
            "results": str(path.resolve()),
            "correct": sum(outcomes),
            "accuracy": mean(outcomes),
            "generation_failures": sum(bool(run[example_id].get("error")) for example_id in expected_ids),
            "judge_errors": sum(bool(run[example_id].get("judge_error")) for example_id in expected_ids),
        })

    summary = {
        "dataset": "hle-diamond-physics",
        "model": args.model,
        "judge_models": sorted({str(row["judge_model"]) for run in runs for row in run if row.get("judge_model")}),
        "k": k,
        "questions": len(expected_ids),
        "total_attempts": len(expected_ids) * k,
        "total_correct": sum(item["correct_runs"] for item in per_question),
        "mean_at_k": mean(item["mean_at_k"] for item in per_question),
        "pass_at_k": mean(item["pass_at_k"] for item in per_question),
        "passed_questions": sum(item["pass_at_k"] for item in per_question),
        "runs": run_summaries,
        "per_question": per_question,
        "repeated_sampling": repeated_metrics([
            {**row, "sample_index": index, "status": "failed" if row.get("error") else "completed",
             "judge_correct": row["correct"]}
            for index, rows in enumerate(runs) for row in rows
        ], k),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_question"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
