#!/usr/bin/env python3
"""Merge an original workflow run with its successful retry results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from theo_conductor.workflow_benchmark import summarize_workflows


from theo_conductor.jsonio import read_jsonl as load_jsonl


from theo_conductor.jsonio import write_jsonl_atomic as write_jsonl, write_json_atomic


def write_summary(path: Path, records: list[dict[str, Any]], *, plans: Path, sources: tuple[Path, ...] = ()) -> None:
    source_summaries = []
    # Preserve facts recorded by the source runs rather than asserting a specific experiment.
    for source in sources:
        summary_path = source.parent / "summary.json"
        if summary_path.exists():
            source_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    summary = {"plans": str(plans), "source_summaries": source_summaries,
               "execution_config_sha256s": sorted({str(record["execution_config_sha256"]) for record in records}),
               **summarize_workflows(records)}
    for key in ("planner_model_id", "dataset", "text_only", "config", "worker_temperature",
                "max_worker_tokens", "worker_token_limit_mode", "judge_model", "batch_workers"):
        values = [source.get(key) for source in source_summaries]
        summary[key] = values[0] if values and all(value == values[0] for value in values) else None
    write_json_atomic(path, summary)


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
            for record in {str(r["example_id"]): r for r in load_jsonl(args.original)}.values()
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

    write_summary(args.output_dir / "summary.json", records, plans=args.output_dir / "plans", sources=(args.original, args.retry))
    print(
        f"Wrote {len(records)} workflows to {args.output_dir}: "
        f"{sum(record.get('judge_correct') is True for record in records)} correct, "
        f"{sum(bool(record.get('judge_error')) for record in records)} judge errors"
    )


if __name__ == "__main__":
    main()
