#!/usr/bin/env python3
"""Merge an original workflow run with its successful retry results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from theo_conductor.workflow_benchmark import summarize_workflows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_summary(path: Path, records: list[dict[str, Any]], *, plans: Path) -> None:
    config_hashes = sorted({str(record["execution_config_sha256"]) for record in records})
    summary = {
        "plans": str(plans),
        "planner_model_id": "gpt-6-astra",
        "dataset": "hle-text",
        "text_only": True,
        "unfiltered_plan_count": len(records),
        "available_plan_count": len(records),
        "config": "configs/worker_pool_frontier.yaml",
        "execution_config_sha256": config_hashes[0] if len(config_hashes) == 1 else None,
        "execution_config_sha256s": config_hashes,
        "offset": 0,
        "max_samples": len(records),
        "concurrency": 4,
        "batch_workers": False,
        "worker_batch_size": None,
        "worker_temperature": 0.2,
        "max_worker_tokens": None,
        "worker_token_limit_mode": "model_output_limits",
        "retry_failures": True,
        "judge_enabled": True,
        "judge_immediately": True,
        "judge_model": "~deepseek/deepseek-flash-latest",
        "fallback_judge_model": "",
        **summarize_workflows(records),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--retry", required=True, type=Path)
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expect", required=True, type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if args.summary_only:
        records = load_jsonl(results_path)
    else:
        original = {
            str(record.get("example_id")): record
            for record in load_jsonl(args.original)
            if record.get("error") is None
        }
        retry = {
            str(record.get("example_id")): record for record in load_jsonl(args.retry)
        }
        overlap = set(original) & set(retry)
        if overlap:
            raise ValueError(f"Original/retry result overlap: {sorted(overlap)}")
        merged = {**original, **retry}
        if len(merged) != args.expect:
            raise ValueError(
                f"Expected {args.expect} unique successful workflows, found {len(merged)}"
            )
        records = sorted(
            merged.values(),
            key=lambda record: (int(record.get("benchmark_position", 10**9)), str(record["example_id"])),
        )
        failed = [record["example_id"] for record in records if record.get("error") is not None]
        if failed:
            raise ValueError(f"Merged output still contains workflow failures: {failed}")
        write_jsonl(results_path, records)

        destination_plans = args.output_dir / "plans"
        if destination_plans.exists():
            raise ValueError(f"Destination plans already exist: {destination_plans}")
        shutil.copytree(args.plans, destination_plans)

    write_summary(args.output_dir / "summary.json", records, plans=args.output_dir / "plans")
    print(
        f"Wrote {len(records)} workflows to {args.output_dir}: "
        f"{sum(record.get('judge_correct') is True for record in records)} correct, "
        f"{sum(bool(record.get('judge_error')) for record in records)} judge errors"
    )


if __name__ == "__main__":
    main()
