#!/usr/bin/env python3
"""Export correct HLE physics answers by classified subtopic and run setup."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "hle_physics_correct_by_subtopic.csv"
CLASSIFICATIONS = (
    ROOT
    / "outputs"
    / "hle-physics-classifications-gpt-5.6-sol"
    / "classifications.jsonl"
)

RUNS = [
    ("GPT-5.5 (direct)", "outputs/gpt-5.5-benchmark/results.jsonl"),
    ("GPT-6 Astra (direct batch)", "outputs/gpt-6-benchmark/results.jsonl"),
    ("Claude Opus 4.8 (direct)", "outputs/opus-4.8-benchmark/results.jsonl"),
    ("Qwen3.8-27B (direct)", "outputs/qwen-3-8-benchmark/results.jsonl"),
    (
        "Theo Conductor (Qwen3.8 plans, frontier workers)",
        "outputs/qwen-3-8-eval-frontier/results.jsonl",
    ),
    (
        "Theo Conductor (Qwen3.8 plans, flash workers)",
        "outputs/qwen-3-8-eval-flash/results.jsonl",
    ),
    (
        "Theo Conductor (GPT-6 Astra plans, frontier workers)",
        "outputs/gpt-6-eval-frontier/results.jsonl",
    ),
    (
        "Theo Conductor (GPT-6 Astra clarification plans, frontier workers)",
        "outputs/gpt-6-eval-frontier-clarification/results.jsonl",
    ),
]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    tags_by_id: dict[str, tuple[str, ...]] = {}
    all_subtopics: set[str] = set()
    for record in read_jsonl(CLASSIFICATIONS):
        tags = tuple(
            record[key]
            for key in ("tag_1", "tag_2", "tag_3")
            if record.get(key)
        )
        tags_by_id[record["id"]] = tags
        all_subtopics.update(tags)

    subtopics = sorted(all_subtopics)
    rows = []
    for setup, relative_path in RUNS:
        # Later records supersede earlier retry/rejudge records for the same question.
        records_by_id = OrderedDict()
        for record in read_jsonl(ROOT / relative_path):
            records_by_id[record["example_id"]] = record

        missing = sorted(set(records_by_id) - set(tags_by_id))
        if missing:
            raise ValueError(f"{relative_path}: {len(missing)} unclassified questions")

        counts = {subtopic: 0 for subtopic in subtopics}
        total_correct = 0
        for example_id, record in records_by_id.items():
            if record.get("correct") is not True:
                continue
            total_correct += 1
            for subtopic in tags_by_id[example_id]:
                counts[subtopic] += 1

        rows.append(
            {
                "setup": setup,
                "run_path": relative_path,
                "total_correct": total_correct,
                "total_questions": len(records_by_id),
                **counts,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setup",
        "run_path",
        "total_correct",
        "total_questions",
        *subtopics,
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs and {len(subtopics)} subtopics to {OUTPUT}")


if __name__ == "__main__":
    main()
