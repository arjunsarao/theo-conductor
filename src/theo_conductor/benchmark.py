from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .data import DEFAULT_MEGASCIENCE_SAMPLES, build_megascience_splits
from .grpo import answers_match
from .models.registry import ModelRegistry


DEFAULT_VALIDATION_SAMPLES = 200
DEFAULT_INSTRUCTION = (
    "Solve the problem independently. Show enough reasoning to make the result verifiable, then end "
    "with a separate line exactly formatted as FINAL: <answer>. The FINAL line should contain only "
    "the concise answer, including units when applicable."
)


def extract_final_answer(text: str) -> str | None:
    """Extract the last explicit FINAL answer from a worker completion."""
    import re

    matches = re.findall(r"(?im)^\s*final\s*(?:answer\s*)?:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def _usage_value(usage: dict[str, Any] | None, *keys: str) -> int | None:
    if not usage:
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _mean_present(records: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
    return mean(values) if values else None


def bootstrap_accuracy_ci(
    outcomes: Sequence[bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> list[float] | None:
    """Return a deterministic percentile bootstrap 95% CI for accuracy."""
    if not outcomes:
        return None
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    values = [int(value) for value in outcomes]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    low = estimates[math.floor(0.025 * (samples - 1))]
    high = estimates[math.ceil(0.975 * (samples - 1))]
    return [low, high]


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate per-question records into model and subject metrics."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model_id"])].append(record)

    models: dict[str, Any] = {}
    for model_id, model_records in grouped.items():
        outcomes = [bool(record.get("correct", False)) for record in model_records]
        successes = [record for record in model_records if record.get("error") is None]
        extracted = [record for record in successes if record.get("extracted_answer") is not None]
        subjects: dict[str, Any] = {}
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in model_records:
            by_subject[str(record.get("subject") or "unknown")].append(record)
        for subject, subject_records in sorted(by_subject.items()):
            subject_outcomes = [bool(record.get("correct", False)) for record in subject_records]
            subjects[subject] = {
                "questions": len(subject_records),
                "correct": sum(subject_outcomes),
                "accuracy": mean(subject_outcomes),
            }

        models[model_id] = {
            "display_name": model_records[0].get("display_name"),
            "questions": len(model_records),
            "correct": sum(outcomes),
            "accuracy": mean(outcomes) if outcomes else None,
            "accuracy_95_ci": bootstrap_accuracy_ci(outcomes, samples=bootstrap_samples, seed=seed),
            "failures": len(model_records) - len(successes),
            "failure_rate": (len(model_records) - len(successes)) / len(model_records),
            "answer_extraction_failures": len(successes) - len(extracted),
            "answer_extraction_failure_rate": (
                (len(successes) - len(extracted)) / len(successes) if successes else None
            ),
            "mean_prompt_tokens": _mean_present(model_records, "prompt_tokens"),
            "mean_generated_tokens": _mean_present(model_records, "completion_tokens"),
            "mean_total_tokens": _mean_present(model_records, "total_tokens"),
            "mean_latency_ms": _mean_present(model_records, "latency_ms"),
            "by_subject": subjects,
        }

    return {"models": models}


def _question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    fingerprint = record.get("question_sha256")
    if not isinstance(fingerprint, str):
        fingerprint = _question_fingerprint(str(record["question"]))
    return str(record["model_id"]), str(record["example_id"]), fingerprint


def _load_completed(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    records: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    if not path.exists():
        return records, completed

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = _record_key(record)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid benchmark record at {path}:{line_number}: {exc}") from exc
            if key not in completed:
                records.append(record)
                completed.add(key)
    return records, completed


async def run_benchmark(
    *,
    registry: ModelRegistry,
    dataset: Sequence[dict[str, Any]],
    results_path: Path,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Benchmark every registered model on every row, resuming from JSONL."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    records, completed = _load_completed(results_path)
    write_lock = asyncio.Lock()
    semaphores = {str(model_id): asyncio.Semaphore(concurrency) for model_id in registry.model_ids()}

    async def evaluate(model_id: int | str, row: dict[str, Any], position: int) -> None:
        fingerprint = _question_fingerprint(str(row["question"]))
        key = (str(model_id), str(row["id"]), fingerprint)
        if key in completed:
            return
        spec = registry.get(model_id)
        record: dict[str, Any] = {
            "model_id": str(model_id),
            "display_name": spec.display_name,
            "example_id": str(row["id"]),
            "question_sha256": fingerprint,
            "benchmark_position": position,
            "subject": row.get("subject"),
            "question": row["question"],
            "gold_answer": row["answer"],
            "reference_answer": row.get("reference_answer"),
            "response": None,
            "extracted_answer": None,
            "correct": False,
            "error": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
        }
        async with semaphores[str(model_id)]:
            try:
                response = await spec.client.generate(
                    instruction=DEFAULT_INSTRUCTION,
                    question=str(row["question"]),
                    context={},
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = extract_final_answer(response.text)
                record.update(
                    response=response.text,
                    extracted_answer=answer,
                    correct=(answer is not None and answers_match(answer, str(row["answer"]))),
                    prompt_tokens=_usage_value(response.usage, "prompt_tokens", "input_tokens"),
                    completion_tokens=_usage_value(response.usage, "completion_tokens", "output_tokens"),
                    total_tokens=_usage_value(response.usage, "total_tokens"),
                    latency_ms=response.latency_ms,
                )
            except Exception as exc:  # Keep a complete denominator when one endpoint fails.
                record["error"] = f"{type(exc).__name__}: {exc}"

        async with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            records.append(record)
            completed.add(key)
            print(
                f"[{len(completed)}] {model_id} / {row['id']}: "
                f"{'correct' if record['correct'] else 'incorrect'}",
                file=sys.stderr,
                flush=True,
            )

    await asyncio.gather(
        *(evaluate(model_id, row, position) for model_id in registry.model_ids() for position, row in enumerate(dataset))
    )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="theo-benchmark",
        description="Benchmark every configured model on the shared MegaScience validation subset.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/local_small_models.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/megascience-small-models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-samples", type=int, default=DEFAULT_MEGASCIENCE_SAMPLES)
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--max-samples", type=int, help="Limit validation rows for a smoke run.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per model endpoint.")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")

    registry = ModelRegistry.from_yaml_file(args.config)
    split = build_megascience_splits(
        seed=args.seed,
        total_samples=args.total_samples,
        validation_samples=args.validation_samples,
    )["test"]
    if args.max_samples is not None:
        split = split.select(range(min(args.max_samples, len(split))))
    dataset = [dict(row) for row in split]

    results_path = args.output_dir / "results.jsonl"
    records = await run_benchmark(
        registry=registry,
        dataset=dataset,
        results_path=results_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
    )
    expected_ids = {str(model_id) for model_id in registry.model_ids()}
    selected_keys = {
        (str(row["id"]), _question_fingerprint(str(row["question"])))
        for row in dataset
    }
    selected_records = [
        record
        for record in records
        if str(record["model_id"]) in expected_ids
        and (str(record["example_id"]), _record_key(record)[2]) in selected_keys
    ]
    summary = {
        "dataset": "MegaScience/MegaScience",
        "split": "train-derived deterministic validation subset",
        "seed": args.seed,
        "total_subset_samples": args.total_samples,
        "validation_samples": args.validation_samples,
        "evaluated_samples": len(dataset),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        **summarize_records(selected_records, bootstrap_samples=args.bootstrap_samples, seed=args.seed),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    try:
        asyncio.run(async_main(argv))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
