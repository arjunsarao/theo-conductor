#!/usr/bin/env python3
"""Run Astra on HLE-Diamond Physics, then judge with Gemini native batch."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from theo_conductor.benchmark import (
    DEFAULT_INSTRUCTION,
    JUDGE_INSTRUCTION,
    build_judge_batch_question,
    build_judge_response_format,
    extract_final_answer,
    parse_judge_batch,
    judge_records_with_checkpoints,
    summarize_records,
    write_results_atomic,
)
from theo_conductor.models.openai_compat import OpenAICompatibleClient
from theo_conductor.worker_prompt import build_worker_prompt

ASTRA_MODEL = "openai/gpt-6-astra"
JUDGE_MODEL = "google/gemini-3.8-flash"
BASE_URL = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--solver-model", default=ASTRA_MODEL)
    parser.add_argument("--solver-name", default="GPT-6 Astra")
    parser.add_argument("--solver-standard", action="store_true")
    parser.add_argument("--solver-timeout-seconds", type=float, default=1800)
    parser.add_argument("--solver-concurrency", type=int, default=4)
    parser.add_argument("--expect-samples", type=int, default=61)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-mode", choices=("batch", "sync"), default="batch")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--multimodal-concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


from .jsonio import read_jsonl as load_jsonl


from .benchmark import _usage_value as usage_value


def build_uniform_judge_response_format(allowed_ids: list[str]) -> dict[str, Any]:
    """Return one identical one-verdict schema for every item in a provider batch."""
    response_format = build_judge_response_format(allowed_ids)
    schema = response_format["json_schema"]["schema"]
    schema["minItems"] = 1
    schema["maxItems"] = 1
    return response_format


def new_record(
    row: dict[str, Any], position: int, *, solver_model: str, solver_name: str
) -> dict[str, Any]:
    return {
        "model_id": solver_model,
        "display_name": solver_name,
        "example_id": str(row["id"]),
        "benchmark_position": position,
        "subject": row.get("subject"),
        "partition": row.get("partition"),
        "is_multimodal": bool(row.get("is_multimodal")),
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


def apply_response(record: dict[str, Any], response: Any) -> None:
    if isinstance(response, Exception):
        record["error"] = f"{type(response).__name__}: {response}"
        return
    for key in ("judge_model", "judge_correct", "judge_reason", "judge_response", "judge_error", "correct"):
        record[key] = None
    response_error = None
    if not response.text or response.finish_reason == "error":
        response_error = f"Empty or failed model response (finish_reason={response.finish_reason!r})"
    record.update(
        response=response.text,

        extracted_answer=extract_final_answer(response.text),
        error=response_error,
        prompt_tokens=usage_value(response.usage, "prompt_tokens", "input_tokens"),
        completion_tokens=usage_value(response.usage, "completion_tokens", "output_tokens"),
        total_tokens=usage_value(response.usage, "total_tokens"),
        latency_ms=response.latency_ms,
        finish_reason=response.finish_reason,
    )


def text_request(row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "question": str(row["question"]),
        "context": {},
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def image_request(row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    prompt = build_worker_prompt(
        instruction=DEFAULT_INSTRUCTION,
        question=str(row["question"]),
        context={},
    )
    messages = [
        {
            "role": "system",
            "content": "You are a worker model in a multi-model reasoning workflow. Follow the instruction exactly.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": row["image"]}},
            ],
        },
    ]
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "question": str(row["question"]),
        "context": {},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": messages,
    }


async def generate_answers(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    results_path: Path,
    api_key: str,
) -> None:
    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=api_key,
        model=args.solver_model,
        timeout_seconds=args.solver_timeout_seconds,
        max_retries=0,
        native_batch=not args.solver_standard,
        batch_backend="openrouter",
        batch_poll_interval_seconds=args.poll_seconds,
    )
    by_id = {record["example_id"]: record for record in records}
    if args.solver_standard:
        pending = [row for row in rows if (not by_id[row["id"]].get("response") or by_id[row["id"]].get("error"))]
        print(f"generation pending: standard={len(pending)}", flush=True)
        semaphore = asyncio.Semaphore(args.solver_concurrency)
        checkpoint_lock = asyncio.Lock()

        async def run_standard(row: dict[str, Any]) -> None:
            record = by_id[row["id"]]
            request = (
                image_request(row, args.max_tokens)
                if row["is_multimodal"]
                else text_request(row, args.max_tokens)
            )
            last_error: Exception | None = None
            for attempt in range(1, args.attempts + 1):
                try:
                    async with semaphore:
                        response = await client.generate(**request)
                    apply_response(record, response)
                    if record.get("error"):
                        raise RuntimeError(record["error"])
                    print(f"{args.solver_name} completed {row['id']} (standard)", flush=True)
                    async with checkpoint_lock:
                        write_results_atomic(results_path, records)
                    return
                except Exception as exc:
                    last_error = exc
                    print(
                        f"{args.solver_name} standard attempt={attempt} id={row['id']} error={exc}",
                        flush=True,
                    )
            assert last_error is not None
            apply_response(record, last_error)
            async with checkpoint_lock:
                write_results_atomic(results_path, records)

        await asyncio.gather(*(run_standard(row) for row in pending))
        write_results_atomic(results_path, records)
        return

    pending_text = [row for row in rows if not row["is_multimodal"] and (not by_id[row["id"]].get("response") or by_id[row["id"]].get("error"))]
    pending_images = [row for row in rows if row["is_multimodal"] and (not by_id[row["id"]].get("response") or by_id[row["id"]].get("error"))]
    print(f"generation pending: text_batch={len(pending_text)} multimodal_standard={len(pending_images)}", flush=True)

    remaining = pending_text
    for attempt in range(1, args.attempts + 1):
        if not remaining:
            break
        print(f"submitting {args.solver_name} text batch attempt={attempt} items={len(remaining)}", flush=True)
        responses = await client.generate_batch([text_request(row, args.max_tokens) for row in remaining])
        retry: list[dict[str, Any]] = []
        for row, response in zip(remaining, responses, strict=True):
            record = by_id[row["id"]]
            apply_response(record, response)
            if record.get("error"):
                retry.append(row)
            else:
                print(f"{args.solver_name} completed {row['id']} (text batch)", flush=True)
        write_results_atomic(results_path, records)
        remaining = retry
    for row in remaining:
        by_id[row["id"]]["error"] = by_id[row["id"]].get("error") or "Astra batch failed"

    semaphore = asyncio.Semaphore(args.multimodal_concurrency)
    checkpoint_lock = asyncio.Lock()

    async def checkpoint() -> None:
        async with checkpoint_lock:
            write_results_atomic(results_path, records)

    async def run_image(row: dict[str, Any]) -> None:
        record = by_id[row["id"]]
        last_error: Exception | None = None
        for attempt in range(1, args.attempts + 1):
            try:
                async with semaphore:
                    response = await client.generate(**image_request(row, args.max_tokens))
                apply_response(record, response)
                if record.get("error"):
                    raise RuntimeError(record["error"])
                print(f"{args.solver_name} completed {row['id']} (multimodal)", flush=True)
                await checkpoint()
                return
            except Exception as exc:
                last_error = exc
                print(f"{args.solver_name} multimodal attempt={attempt} id={row['id']} error={exc}", flush=True)
        assert last_error is not None
        apply_response(record, last_error)
        await checkpoint()

    await asyncio.gather(*(run_image(row) for row in pending_images))
    write_results_atomic(results_path, records)


async def judge_answers(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    results_path: Path,
    api_key: str,
) -> None:
    if args.judge_mode == "sync":
        for record in records:
            if (
                record.get("response")
                and not record.get("error")
                and record.get("judge_reason")
                == "The model request failed or produced no response."
            ):
                record.update(
                    judge_model=None,
                    judge_correct=None,
                    judge_reason=None,
                    judge_response=None,
                    judge_error=None,
                    correct=None,
                )
        client = OpenAICompatibleClient(
            base_url=BASE_URL,
            api_key=api_key,
            model=JUDGE_MODEL,
            timeout_seconds=1800,
            max_retries=0,
        )
        print(
            f"submitting synchronous Gemini judge requests "
            f"items={len(records)} concurrency={args.judge_concurrency}",
            flush=True,
        )
        await judge_records_with_checkpoints(
            records,
            all_records=records,
            results_path=results_path,
            client=client,
            judge_model=JUDGE_MODEL,
            concurrency=args.judge_concurrency,
            batch_size=1,
            max_tokens=args.judge_max_tokens,
            attempts=args.attempts,
            checkpoint_size=8,
        )
        return

    client = OpenAICompatibleClient(
        base_url=BASE_URL,
        api_key=api_key,
        model=JUDGE_MODEL,
        timeout_seconds=1800,
        max_retries=0,
        native_batch=True,
        batch_backend="openrouter",
        batch_poll_interval_seconds=args.poll_seconds,
    )
    pending = [
        record for record in records
        if record.get("judge_model") != JUDGE_MODEL
        or not isinstance(record.get("judge_correct"), bool)
        or record.get("judge_error")
    ]
    for record in pending:
        record.update(
            judge_model=JUDGE_MODEL,
            judge_correct=None,
            judge_reason=None,
            judge_response=None,
            judge_error=None,
        )
    judgeable = [record for record in pending if record.get("response") and not record.get("error")]
    for record in pending:
        if record not in judgeable:
            record.update(
                judge_correct=False,
                judge_reason="The model request failed or produced no response.",
                correct=False,
            )

    remaining = judgeable
    for attempt in range(1, args.attempts + 1):
        if not remaining:
            break
        requests = []
        allowed_ids = [str(record["example_id"]) for record in remaining]
        response_format = build_uniform_judge_response_format(allowed_ids)
        for record in remaining:
            item_id = str(record["example_id"])
            requests.append({
                "instruction": JUDGE_INSTRUCTION,
                "question": build_judge_batch_question([(item_id, record)]),
                "context": {},
                "max_tokens": args.judge_max_tokens,
                "temperature": 0.0,
                "response_format": response_format,
            })
        print(f"submitting Gemini judge batch attempt={attempt} items={len(remaining)}", flush=True)
        responses = await client.generate_batch(requests)
        retry = []
        for record, response in zip(remaining, responses, strict=True):
            item_id = str(record["example_id"])
            try:
                if isinstance(response, Exception):
                    raise response
                correct, reason = parse_judge_batch(response.text, [item_id])[item_id]
                record.update(
                    judge_correct=correct,
                    judge_reason=reason,
                    judge_response=response.text,
                    judge_error=None,
                    correct=correct,
                )
                print(f"Gemini judged {item_id}: {'correct' if correct else 'incorrect'}", flush=True)
            except Exception as exc:
                record["judge_error"] = f"{type(exc).__name__}: {exc}"
                retry.append(record)
        write_results_atomic(results_path, records)
        remaining = retry
    for record in remaining:
        record["correct"] = False
    write_results_atomic(results_path, records)


async def run(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.dry_run:
        raise ValueError(f"{args.api_key_env} is not set")
    rows = load_jsonl(args.input)
    if len(rows) != args.expect_samples:
        raise ValueError(f"Expected {args.expect_samples} questions, found {len(rows)}")
    if sum(bool(row.get("is_multimodal")) for row in rows) != 21:
        raise ValueError("Expected exactly 21 multimodal questions")
    for row in rows:
        if row.get("is_multimodal") and not str(row.get("image") or "").startswith("data:image/"):
            raise ValueError(f"Missing embedded image for {row['id']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    existing = {record["example_id"]: record for record in load_jsonl(results_path)} if results_path.exists() else {}
    for row in rows:
        previous = existing.get(row["id"])
        if previous and (previous.get("model_id") != args.solver_model or previous.get("question") != row["question"] or previous.get("gold_answer") != row["answer"]):
            raise ValueError("Existing results belong to a different model or input; use a new output directory")
    records = [
        existing.get(
            row["id"],
            new_record(
                row,
                position,
                solver_model=args.solver_model,
                solver_name=args.solver_name,
            ),
        )
        for position, row in enumerate(rows)
    ]
    write_results_atomic(results_path, records)
    print(f"dataset={len(rows)} text=40 multimodal=21 output={args.output_dir}", flush=True)
    if args.dry_run:
        return 0

    if not args.judge_only:
        await generate_answers(args=args, rows=rows, records=records, results_path=results_path, api_key=api_key)
    for record in records:
        if not record.get("response") or record.get("finish_reason") == "error":
            record["error"] = record.get("error") or (
                f"Empty or failed model response (finish_reason={record.get("finish_reason")!r})"
            )
    write_results_atomic(results_path, records)
    completed = sum(bool(record.get("response")) and not record.get("error") for record in records)
    print(f"all generation requests collected: successful={completed}/61", flush=True)
    await judge_answers(args=args, records=records, results_path=results_path, api_key=api_key)

    summary = {
        "dataset": "hle-diamond-physics",
        "evaluated_samples": len(records),
        "solver_model": args.solver_model,
        "solver_text_mode": (
            "OpenRouter standard requests" if args.solver_standard else "OpenRouter native batch"
        ),
        "solver_multimodal_mode": "OpenRouter standard requests",
        "text_questions": 40,
        "multimodal_questions": 21,
        "judge_enabled": True,
        "judge_model": JUDGE_MODEL,
        "judge_mode": (
            "OpenRouter native batch" if args.judge_mode == "batch"
            else "OpenRouter synchronous requests"
        ),
        "generation_successes": completed,
        "judge_errors": sum(bool(record.get("judge_error")) for record in records),
        **summarize_records(records, bootstrap_samples=args.bootstrap_samples, seed=42),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 1 if summary["judge_errors"] or (not args.judge_only and completed != len(records)) else 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
