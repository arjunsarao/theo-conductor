#!/usr/bin/env python3
"""Select source plans whose latest benchmark result has a workflow error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect", required=True, type=int)
    args = parser.parse_args()

    latest_results = {
        str(record.get("example_id")): record for record in load_jsonl(args.results)
    }
    failed_ids = {
        example_id
        for example_id, record in latest_results.items()
        if record.get("error") is not None
    }
    if len(failed_ids) != args.expect:
        raise ValueError(f"Expected {args.expect} failed workflows, found {len(failed_ids)}")

    selected = [
        record
        for record in load_jsonl(args.plans)
        if str(record.get("dataset_id")) in failed_ids
    ]
    selected_ids = {str(record.get("dataset_id")) for record in selected}
    if selected_ids != failed_ids or len(selected) != len(failed_ids):
        raise ValueError(
            f"Plan selection mismatch: missing={sorted(failed_ids - selected_ids)}, "
            f"duplicates={len(selected) - len(selected_ids)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Selected {len(selected)} failed workflows into {args.output}")


if __name__ == "__main__":
    main()
