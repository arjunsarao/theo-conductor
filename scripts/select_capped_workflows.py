#!/usr/bin/env python3
"""Select unsuccessful capped/errored HLE workflows for a clean rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


from theo_conductor.jsonio import read_jsonl as load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--plans", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument(
        "--capped-only",
        action="store_true",
        help="Select only incorrect workflows containing a length-capped worker step.",
    )
    args = parser.parse_args()

    results = list({str(record["example_id"]): record for record in load_jsonl(args.results)}.values())
    if args.capped_only:
        target_ids = {
            str(record.get("example_id"))
            for record in results
            if record.get("judge_correct") is False
            and any(
                isinstance(output, dict) and output.get("finish_reason") == "length"
                for output in (record.get("worker_outputs") or {}).values()
            )
        }
    else:
        target_ids = {
            str(record.get("example_id"))
            for record in results
            if (
                record.get("error") is not None
                and record.get("error_type") != "ValueError"
            )
            or (
                record.get("error") is None
                and record.get("judge_correct") is not True
                and any(
                    isinstance(output, dict) and output.get("finish_reason") in {"length", "error"}
                    for output in (record.get("worker_outputs") or {}).values()
                )
            )
        }
    capped_models = {
        str(output.get("model_id"))
        for record in results if str(record.get("example_id")) in target_ids
        for output in (record.get("worker_outputs") or {}).values()
        if isinstance(output, dict) and output.get("finish_reason") == "length"
    }
    plans = [record for record in load_jsonl(args.plans) if str(record.get("dataset_id")) in target_ids]
    found = {str(record.get("dataset_id")) for record in plans}
    if len(plans) != len(target_ids) or found != target_ids:
        raise ValueError(f"Missing source plans for IDs: {sorted(target_ids - found)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    from theo_conductor.jsonio import write_jsonl_atomic
    write_jsonl_atomic(args.output, plans)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    doubled: dict[str, tuple[int, int]] = {}
    for model in config.get("models", []):
        model_id = str(model.get("model_idx"))
        limit = model.get("max_output_tokens")
        if model_id in capped_models and isinstance(limit, int):
            model["max_output_tokens"] = limit * 2
            doubled[model_id] = (limit, limit * 2)
    args.config_output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Selected {len(plans)} capped/errored unsuccessful workflows into {args.output}")
    print(f"Doubled output limits: {doubled}")


if __name__ == "__main__":
    main()
