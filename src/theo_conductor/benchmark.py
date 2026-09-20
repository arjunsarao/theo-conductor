from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .data import TRAINING_DATASETS, build_conductor_splits, load_conductor_dataset
from .models.registry import ModelRegistry
from .models.openai_compat import OpenAICompatibleClient


DEFAULT_VALIDATION_SAMPLES = 200
DEFAULT_JUDGE_BASE_URL = "http://10.100.50.35:30080/v1"
DEFAULT_JUDGE_MODEL = "moonshotai/Kimi-K2.6"
DEFAULT_FALLBACK_JUDGE_BASE_URL = DEFAULT_JUDGE_BASE_URL
DEFAULT_FALLBACK_JUDGE_MODEL = "glm-5.2-fp8"
DEFAULT_INSTRUCTION = (
    "Solve the problem independently. Show enough reasoning to make the result verifiable, then end "
    "with a separate line exactly formatted as FINAL: <answer>. The FINAL line should contain only "
    "the concise answer, including units when applicable."
)
JUDGE_INSTRUCTION = """You are an exacting but fair evaluator of scientific answers.
Evaluate every item independently. Decide whether its candidate response correctly answers its question,
using the reference and full gold answer as the grading standard. Accept mathematically or scientifically
equivalent wording, concise answers, harmless extra explanation, and answers that are more specific than
the reference. Reject answers with a substantive contradiction, wrong value, missing required part, or
reasoning whose final conclusion is wrong. Judge the candidate's actual answer, not formatting such as
whether it used FINAL:.
"""


def build_judge_response_format(expected_ids: Sequence[str]) -> dict[str, Any]:
    """Build the strict structured-output contract for one judge request."""
    if not expected_ids:
        raise ValueError("judge response schema requires at least one id")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_verdicts",
            "strict": True,
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(expected_ids)},
                        "correct": {"type": "boolean"},
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["id", "correct", "reason"],
                    "additionalProperties": False,
                },
                "minItems": len(expected_ids),
                "maxItems": len(expected_ids),
            },
        },
    }


def extract_final_answer(text: str) -> str | None:
    """Extract the last explicit FINAL answer from a worker completion."""
    import re

    matches = re.findall(r"(?im)^\s*final\s*(?:answer\s*)?:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def parse_judge_batch(text: str, expected_ids: Sequence[str]) -> dict[str, tuple[bool, str]]:
    """Parse and strictly validate a batch of judge verdicts."""
    decoder = json.JSONDecoder()
    payload: Any = None
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, list):
            payload = candidate
            break
    if not isinstance(payload, list):
        raise ValueError("Judge response did not contain a JSON verdict array")

    verdicts: dict[str, tuple[bool, str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each judge verdict must be an object")
        item_id = item.get("id")
        correct = item.get("correct")
        reason = item.get("reason")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Judge verdict 'id' must be a non-empty string")
        if item_id in verdicts:
            raise ValueError(f"Judge returned duplicate id: {item_id}")
        if not isinstance(correct, bool):
            raise ValueError("Judge verdict 'correct' must be a boolean")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Judge verdict 'reason' must be a non-empty string")
        verdicts[item_id] = (correct, reason.strip())

    if set(verdicts) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(verdicts))
        unexpected = sorted(set(verdicts) - set(expected_ids))
        raise ValueError(f"Judge verdict ids did not match request (missing={missing}, unexpected={unexpected})")
    return verdicts


def build_judge_batch_question(batch: Sequence[tuple[str, dict[str, Any]]]) -> str:
    items = [
        {
            "id": item_id,
            "question": record.get("question") or "(missing)",
            "reference_answer": record.get("reference_answer") or "(not provided)",
            "full_gold_answer": record.get("gold_answer") or "(not provided)",
            "candidate_response": record.get("response") or "(no response)",
            "extracted_final_answer": record.get("extracted_answer"),
        }
        for item_id, record in batch
    ]
    return "Evaluate every item in this JSON array:\n" + json.dumps(items, ensure_ascii=False)


async def judge_records(
    records: Sequence[dict[str, Any]],
    *,
    client: Any,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    fallback_client: Any | None = None,
    fallback_judge_model: str = DEFAULT_FALLBACK_JUDGE_MODEL,
    concurrency: int = 8,
    batch_size: int = 10,
    max_tokens: int = 8192,
    fallback_max_tokens: int | None = None,
    attempts: int = 3,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Semantically grade records in place, resuming records already judged by this model."""
    if concurrency <= 0:
        raise ValueError("judge concurrency must be positive")
    if batch_size <= 0:
        raise ValueError("judge batch size must be positive")
    if max_tokens <= 0:
        raise ValueError("judge max tokens must be positive")
    if fallback_max_tokens is not None and fallback_max_tokens <= 0:
        raise ValueError("fallback judge max tokens must be positive")
    if attempts <= 0:
        raise ValueError("judge attempts must be positive")

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    pending: list[dict[str, Any]] = []
    accepted_judge_models = {judge_model}
    if fallback_client is not None:
        accepted_judge_models.add(fallback_judge_model)
    for record in records:
        already_judged = (
            record.get("judge_model") in accepted_judge_models
            and isinstance(record.get("judge_correct"), bool)
            and not record.get("judge_error")
        )
        if already_judged and not force:
            continue
        record["judge_model"] = judge_model
        record["judge_correct"] = None
        record["judge_reason"] = None
        record["judge_response"] = None
        record["judge_error"] = None
        if record.get("error") is not None or not record.get("response"):
            record.update(
                judge_correct=False,
                judge_reason="The model request failed or produced no response.",
                correct=False,
            )
        else:
            pending.append(record)

    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

    async def grade(batch_index: int, records_batch: Sequence[dict[str, Any]]) -> None:
        nonlocal completed
        identified_batch = [(f"batch-{batch_index}-item-{index}", record) for index, record in enumerate(records_batch)]
        expected_ids = [item_id for item_id, _ in identified_batch]
        backends = [(client, judge_model, max_tokens)]
        if fallback_client is not None:
            backends.append(
                (fallback_client, fallback_judge_model, fallback_max_tokens or max_tokens)
            )
        last_error: Exception | None = None
        judged = False
        for backend, backend_model, backend_max_tokens in backends:
            for _attempt in range(1, attempts + 1):
                try:
                    async with semaphore:
                        response = await backend.generate(
                            instruction=JUDGE_INSTRUCTION,
                            question=build_judge_batch_question(identified_batch),
                            context={},
                            max_tokens=backend_max_tokens,
                            temperature=0.0,
                            response_format=build_judge_response_format(expected_ids),
                        )
                    verdicts = parse_judge_batch(response.text, expected_ids)
                    for item_id, record in identified_batch:
                        correct, reason = verdicts[item_id]
                        record.update(
                            judge_correct=correct,
                            judge_model=backend_model,
                            judge_reason=reason,
                            judge_response=json.dumps(
                                {"id": item_id, "correct": correct, "reason": reason}, ensure_ascii=False
                            ),
                            judge_error=None,
                            correct=correct,
                        )
                    judged = True
                    break
                except Exception as exc:
                    last_error = exc
            if judged:
                break
        if not judged:
            assert last_error is not None
            for _, record in identified_batch:
                record["judge_model"] = backends[-1][1]
                record["judge_error"] = f"{type(last_error).__name__}: {last_error}"
                record["correct"] = False

        completed += len(records_batch)
        errors = sum(bool(record.get("judge_error")) for record in records_batch)
        print(
            f"[judge {completed}] batch {batch_index + 1}/{len(batches)}: "
            f"{len(records_batch) - errors} judged, {errors} errors",
            file=sys.stderr,
            flush=True,
        )

    await asyncio.gather(*(grade(index, batch) for index, batch in enumerate(batches)))
    return list(records)


def write_results_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Replace a JSONL file only after its complete updated form is on disk."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


async def judge_records_with_checkpoints(
    records: Sequence[dict[str, Any]],
    *,
    all_records: Sequence[dict[str, Any]],
    results_path: Path,
    client: Any,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    fallback_client: Any | None = None,
    fallback_judge_model: str = DEFAULT_FALLBACK_JUDGE_MODEL,
    concurrency: int = 8,
    batch_size: int = 10,
    max_tokens: int = 8192,
    fallback_max_tokens: int | None = None,
    attempts: int = 3,
    checkpoint_size: int = 25,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Judge records and atomically checkpoint the JSONL after each small batch."""
    if checkpoint_size <= 0:
        raise ValueError("judge checkpoint size must be positive")
    accepted_judge_models = {judge_model}
    if fallback_client is not None:
        accepted_judge_models.add(fallback_judge_model)
    pending = [
        record
        for record in records
        if force
        or record.get("judge_model") not in accepted_judge_models
        or not isinstance(record.get("judge_correct"), bool)
        or bool(record.get("judge_error"))
    ]
    for start in range(0, len(pending), checkpoint_size):
        await judge_records(
            pending[start : start + checkpoint_size],
            client=client,
            judge_model=judge_model,
            fallback_client=fallback_client,
            fallback_judge_model=fallback_judge_model,
            concurrency=concurrency,
            batch_size=batch_size,
            max_tokens=max_tokens,
            fallback_max_tokens=fallback_max_tokens,
            attempts=attempts,
            force=force,
        )
        write_results_atomic(results_path, all_records)
    return list(records)


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
        outcomes = [
            record["correct"]
            for record in model_records
            if isinstance(record.get("correct"), bool)
        ]
        successes = [record for record in model_records if record.get("error") is None]
        extracted = [record for record in successes if record.get("extracted_answer") is not None]
        subjects: dict[str, Any] = {}
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in model_records:
            by_subject[str(record.get("subject") or "unknown")].append(record)
        for subject, subject_records in sorted(by_subject.items()):
            subject_outcomes = [
                record["correct"]
                for record in subject_records
                if isinstance(record.get("correct"), bool)
            ]
            subjects[subject] = {
                "questions": len(subject_records),
                "judged_questions": len(subject_outcomes),
                "correct": sum(subject_outcomes),
                "accuracy": mean(subject_outcomes) if subject_outcomes else None,
            }

        models[model_id] = {
            "display_name": model_records[0].get("display_name"),
            "questions": len(model_records),
            "judged_questions": len(outcomes),
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


def oracle_routing_breakdown(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Split each oracle-solvable question equally across its correct models."""
    correct_models: dict[str, set[str]] = defaultdict(set)
    display_names: dict[str, str] = {}
    for record in records:
        question_id = str(
            record.get("question_sha256")
            or record.get("example_id")
            or record.get("question")
        )
        correct_models.setdefault(question_id, set())
        model_id = str(record.get("model_id") or "(missing)")
        display_names.setdefault(
            model_id, str(record.get("display_name") or model_id)
        )
        if record.get("correct"):
            correct_models[question_id].add(model_id)

    credits: dict[str, float] = defaultdict(float)
    solved_questions = 0
    tied_questions = 0
    for model_ids in correct_models.values():
        if not model_ids:
            continue
        solved_questions += 1
        tied_questions += len(model_ids) > 1
        credit = 1 / len(model_ids)
        for model_id in model_ids:
            credits[model_id] += credit

    return {
        "questions": len(correct_models),
        "solved_questions": solved_questions,
        "tied_questions": tied_questions,
        "models": [
            {
                "model_id": model_id,
                "display_name": display_names[model_id],
                "oracle_selection_credit": credit,
                "oracle_selection_share": (
                    credit / solved_questions if solved_questions else None
                ),
            }
            for model_id, credit in sorted(
                credits.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


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
    use_model_output_limits: bool = False,
    temperature: float = 0.0,
    concurrency: int = 4,
    batch_models: set[str] | None = None,
    worker_batch_size: int = 50_000,
) -> list[dict[str, Any]]:
    """Benchmark every registered model on every row, resuming from JSONL."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if worker_batch_size <= 0:
        raise ValueError("worker_batch_size must be positive")

    batch_models = batch_models or set()
    model_ids_by_text = {str(model_id): model_id for model_id in registry.model_ids()}
    unknown_batch_models = batch_models - set(model_ids_by_text)
    if unknown_batch_models:
        raise ValueError(f"Unknown batch models: {', '.join(sorted(unknown_batch_models))}")
    for model_id in batch_models:
        if not getattr(
            registry.get(model_ids_by_text[model_id]).client,
            "supports_batch",
            False,
        ):
            raise ValueError(f"Model {model_id!r} is not configured with native_batch support")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    records, completed = _load_completed(results_path)
    write_lock = asyncio.Lock()
    semaphores = {str(model_id): asyncio.Semaphore(concurrency) for model_id in registry.model_ids()}

    def prepare(model_id: int | str, row: dict[str, Any], position: int):
        fingerprint = _question_fingerprint(str(row["question"]))
        key = (str(model_id), str(row["id"]), fingerprint)
        if key in completed:
            return None
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
            "correct": None,
            "error": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
            "finish_reason": None,
        }
        request = {
            "instruction": DEFAULT_INSTRUCTION,
            "question": str(row["question"]),
            "context": {},
            "max_tokens": (
                spec.max_output_tokens or max_tokens
                if use_model_output_limits
                else max_tokens
            ),
            "temperature": temperature,
        }
        return key, spec, record, request

    def apply_response(record: dict[str, Any], response: Any) -> None:
        if isinstance(response, Exception):
            record["error"] = f"{type(response).__name__}: {response}"
            return
        record.update(
            response=response.text,
            extracted_answer=extract_final_answer(response.text),
            prompt_tokens=_usage_value(response.usage, "prompt_tokens", "input_tokens"),
            completion_tokens=_usage_value(response.usage, "completion_tokens", "output_tokens"),
            total_tokens=_usage_value(response.usage, "total_tokens"),
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
        )

    async def checkpoint(key: tuple[str, str, str], record: dict[str, Any]) -> None:
        async with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records.append(record)
            completed.add(key)
            print(
                f"[{len(completed)}] {record['model_id']} / {record['example_id']}: "
                f"{'error' if record['error'] else 'completed'}",
                file=sys.stderr,
                flush=True,
            )

    async def evaluate(model_id: int | str, row: dict[str, Any], position: int) -> None:
        prepared = prepare(model_id, row, position)
        if prepared is None:
            return
        key, spec, record, request = prepared
        async with semaphores[str(model_id)]:
            try:
                response = await spec.client.generate(**request)
            except Exception as exc:  # Keep a complete denominator when one endpoint fails.
                response = exc
        apply_response(record, response)
        await checkpoint(key, record)

    async def evaluate_batched_model(model_id: int | str) -> None:
        pending = [
            prepared
            for position, row in enumerate(dataset)
            if (prepared := prepare(model_id, row, position)) is not None
        ]
        for start in range(0, len(pending), worker_batch_size):
            chunk = pending[start : start + worker_batch_size]
            spec = registry.get(model_id)
            try:
                responses = await spec.client.generate_batch([item[3] for item in chunk])
                if len(responses) != len(chunk):
                    raise RuntimeError(
                        f"batch returned {len(responses)} results for {len(chunk)} requests"
                    )
            except Exception as exc:
                responses = [exc] * len(chunk)
            for (key, _, record, _), response in zip(chunk, responses, strict=True):
                apply_response(record, response)
                await checkpoint(key, record)

    ordinary = [
        evaluate(model_id, row, position)
        for model_id in registry.model_ids()
        if str(model_id) not in batch_models
        for position, row in enumerate(dataset)
    ]
    batched = [
        evaluate_batched_model(model_id)
        for model_id in registry.model_ids()
        if str(model_id) in batch_models
    ]
    await asyncio.gather(*ordinary, *batched)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="theo-benchmark",
        description="Benchmark every configured model on a shared deterministic evaluation split.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/worker_pool_small.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/megascience-small-models"))
    parser.add_argument("--dataset", choices=TRAINING_DATASETS, default="megascience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-samples",
        type=int,
        help="Optional seeded dataset cap before creating the evaluation split.",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate the complete selected dataset instead of creating a train/test split.",
    )
    parser.add_argument("--validation-samples", type=int, default=DEFAULT_VALIDATION_SAMPLES)
    parser.add_argument("--max-samples", type=int, help="Limit validation rows for a smoke run.")
    parser.add_argument(
        "--expect-samples",
        type=int,
        help="Fail before inference unless the selected evaluation set has exactly this many rows.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--use-model-output-limits",
        action="store_true",
        help="Use each model's configured max_output_tokens instead of --max-tokens.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per model endpoint.")
    parser.add_argument(
        "--batch-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="Use the provider Batch API for this model (repeatable; requires native_batch in YAML).",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=50_000,
        help="Maximum questions per provider batch job.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Kimi K2.6 with GLM fallback as semantic correctness judges (enabled by default).",
    )
    parser.add_argument("--judge-base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_JUDGE_BASE_URL))
    parser.add_argument("--judge-api-key", default=os.environ.get("KIMI_API_KEY", "change-this"))
    parser.add_argument("--judge-model", default=os.environ.get("KIMI_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument(
        "--fallback-judge-base-url",
        default=os.environ.get("GLM_BASE_URL", DEFAULT_FALLBACK_JUDGE_BASE_URL),
    )
    parser.add_argument("--fallback-judge-api-key", default=os.environ.get("GLM_API_KEY", "change-this"))
    parser.add_argument(
        "--fallback-judge-model",
        default=os.environ.get("GLM_MODEL", DEFAULT_FALLBACK_JUDGE_MODEL),
    )
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument("--judge-checkpoint-size", type=int, default=25)
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")

    registry = ModelRegistry.from_yaml_file(args.config)
    if args.full_dataset:
        split = load_conductor_dataset(
            args.dataset,
            seed=args.seed,
            max_samples=args.total_samples,
        )
        split_name = "full dataset"
    else:
        split = build_conductor_splits(
            args.dataset,
            seed=args.seed,
            total_samples=args.total_samples,
            validation_samples=args.validation_samples,
        )["test"]
        split_name = "deterministic held-out evaluation subset"
    if args.max_samples is not None:
        split = split.select(range(min(args.max_samples, len(split))))
    dataset = [dict(row) for row in split]
    if args.expect_samples is not None and len(dataset) != args.expect_samples:
        raise ValueError(
            f"Expected {args.expect_samples} evaluation samples, found {len(dataset)}"
        )

    results_path = args.output_dir / "results.jsonl"
    records = await run_benchmark(
        registry=registry,
        dataset=dataset,
        results_path=results_path,
        max_tokens=args.max_tokens,
        use_model_output_limits=args.use_model_output_limits,
        temperature=args.temperature,
        concurrency=args.concurrency,
        batch_models=set(args.batch_model),
        worker_batch_size=args.worker_batch_size,
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
    if args.judge:
        judge_client = OpenAICompatibleClient(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
        )
        fallback_judge_client = OpenAICompatibleClient(
            base_url=args.fallback_judge_base_url,
            api_key=args.fallback_judge_api_key,
            model=args.fallback_judge_model,
        )
        await judge_records_with_checkpoints(
            selected_records,
            all_records=records,
            results_path=results_path,
            client=judge_client,
            judge_model=args.judge_model,
            fallback_client=fallback_judge_client,
            fallback_judge_model=args.fallback_judge_model,
            concurrency=args.judge_concurrency,
            batch_size=args.judge_batch_size,
            max_tokens=args.judge_max_tokens,
            attempts=args.judge_attempts,
            checkpoint_size=args.judge_checkpoint_size,
        )
    summary = {
        "dataset": args.dataset,
        "split": split_name,
        "seed": args.seed,
        "total_subset_samples": args.total_samples,
        "validation_samples": args.validation_samples,
        "evaluated_samples": len(dataset),
        "max_tokens": None if args.use_model_output_limits else args.max_tokens,
        "use_model_output_limits": args.use_model_output_limits,
        "batch_models": args.batch_model,
        "worker_batch_size": args.worker_batch_size if args.batch_model else None,
        "temperature": args.temperature,
        "judge_enabled": args.judge,
        "judge_model": args.judge_model if args.judge else None,
        "fallback_judge_model": args.fallback_judge_model if args.judge else None,
        "judge_batch_size": args.judge_batch_size if args.judge else None,
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
