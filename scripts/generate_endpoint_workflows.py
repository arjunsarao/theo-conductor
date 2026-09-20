#!/usr/bin/env python3
"""Generate one conductor workflow per HLE Physics question and endpoint model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from theo_conductor.data import load_hle_physics_text_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = REPO_ROOT / "CONDUCTOR_PROMPT.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hle-endpoint-workflows"
DEFAULT_BASE_URL = "http://10.100.50.35:30080/v1"
QUESTION_MARKER = "<USER QUESTION>"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate independent conductor workflows for all 202 text-only HLE Physics "
            "questions using Kimi and GLM."
        )
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--expect-samples", type=int, default=202)
    return parser.parse_args()


def endpoint_specs() -> list[dict[str, str]]:
    return [
        {
            "endpoint_model": "kimi",
            "base_url": os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL),
            "api_key": os.environ.get("KIMI_API_KEY", "change-this"),
            "model": os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K2.6"),
        },
        {
            "endpoint_model": "glm",
            "base_url": os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL),
            "api_key": os.environ.get("GLM_API_KEY", "change-this"),
            "model": os.environ.get("GLM_MODEL", "glm-5.2-fp8"),
        },
    ]


def load_prompt_template(path: Path) -> tuple[str, list[str]]:
    template = path.read_text(encoding="utf-8")
    if template.count(QUESTION_MARKER) != 1:
        raise ValueError(f"{path} must contain exactly one {QUESTION_MARKER!r} marker")
    model_ids = list(dict.fromkeys(re.findall(r'^- model_id="([^"]+)"', template, re.MULTILINE)))
    if not model_ids:
        raise ValueError(f"Could not find any worker model_id entries in {path}")
    return template, model_ids


def workflow_response_format(model_ids: list[str]) -> dict[str, Any]:
    step = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1},
            "model_id": {"type": "string", "enum": model_ids},
            "instruction": {"type": "string", "minLength": 1},
            "access_list": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["step_id", "model_id", "instruction", "access_list"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "conductor_workflow",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "physics", "chemistry", "biology", "mathematics",
                            "computer_science", "engineering", "earth_and_space_science",
                            "medicine_and_health", "social_science", "humanities",
                            "interdisciplinary", "other",
                        ],
                    },
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "workflow": {"type": "array", "items": step, "minItems": 1, "maxItems": 7},
                },
                "required": ["task_type", "difficulty", "workflow"],
                "additionalProperties": False,
            },
        },
    }


def validate_plan(plan: Any, model_ids: set[str]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("completion is not a JSON object")
    workflow = plan.get("workflow")
    if not isinstance(workflow, list) or not 1 <= len(workflow) <= 7:
        raise ValueError("workflow must contain between 1 and 7 steps")
    seen: set[str] = set()
    for index, step in enumerate(workflow):
        if not isinstance(step, dict):
            raise ValueError(f"workflow step {index} is not an object")
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id or step_id in seen:
            raise ValueError(f"workflow step {index} has an empty or duplicate step_id")
        if step.get("model_id") not in model_ids:
            raise ValueError(f"workflow step {step_id!r} has an unavailable model_id")
        access_list = step.get("access_list")
        if not isinstance(access_list, list) or any(
            item != "question" and item not in seen for item in access_list
        ):
            raise ValueError(f"workflow step {step_id!r} has a forward or unknown dependency")
        seen.add(step_id)
    return plan


def usage_dict(completion: Any) -> dict[str, Any] | None:
    usage = getattr(completion, "usage", None)
    return usage.model_dump(mode="json") if usage is not None else None


async def generate_one(
    *,
    client: AsyncOpenAI,
    spec: dict[str, str],
    row: dict[str, Any],
    dataset_index: int,
    prompt_file: Path,
    prompt_template: str,
    prompt_sha256: str,
    model_ids: set[str],
    response_format: dict[str, Any],
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    question = str(row["question"])
    prompt = prompt_template.replace(QUESTION_MARKER, question)
    started = time.perf_counter()
    completion: Any | None = None
    try:
        async with semaphore:
            request_started = time.perf_counter()
            completion = await client.chat.completions.create(
                model=spec["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format=response_format,
            )
            latency_ms = (time.perf_counter() - request_started) * 1000
        choice = completion.choices[0]
        raw_completion = choice.message.content or ""
        plan = validate_plan(json.loads(raw_completion), model_ids)
        error_type = None
        error = None
        finish_reason = choice.finish_reason
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        raw_completion = ""
        finish_reason = None
        if completion is not None and getattr(completion, "choices", None):
            choice = completion.choices[0]
            raw_completion = choice.message.content or ""
            finish_reason = choice.finish_reason
        plan = None
        error_type = type(exc).__name__
        error = f"{error_type}: {exc}"

    usage = usage_dict(completion) if completion is not None else None
    return {
        "timestamp": utc_now(),
        "dataset": "hle-physics-text",
        "dataset_id": str(row["id"]),
        "dataset_index": dataset_index,
        "question": question,
        "gold_answer": row.get("answer"),
        "reference_answer": row.get("reference_answer"),
        "answer_type": row.get("answer_type"),
        "subject": row.get("subject"),
        "endpoint_model": spec["endpoint_model"],
        "served_model": spec["model"],
        "base_url": spec["base_url"],
        "prompt_file": str(prompt_file),
        "prompt_sha256": prompt_sha256,
        "plan": plan,
        "conductor_completion": raw_completion,
        "latency_ms": latency_ms,
        "usage": usage,
        "input_tokens": usage.get("prompt_tokens") if usage else None,
        "output_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "finish_reason": finish_reason,
        "error_type": error_type,
        "error": error,
    }


def load_latest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[(str(record["endpoint_model"]), str(record["dataset_id"]))] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed checkpoint record: {exc}") from exc
    return latest


def write_summary(
    path: Path,
    records: list[dict[str, Any]],
    *,
    prompt_file: Path,
    prompt_sha256: str,
) -> None:
    models: dict[str, Any] = {}
    for endpoint_model in ("kimi", "glm"):
        subset = [record for record in records if record["endpoint_model"] == endpoint_model]
        valid = [record for record in subset if record.get("plan") is not None and not record.get("error")]
        latencies = [float(record["latency_ms"]) for record in subset if record.get("latency_ms") is not None]
        models[endpoint_model] = {
            "records": len(subset),
            "valid": len(valid),
            "failed": len(subset) - len(valid),
            "input_tokens": sum(int(record.get("input_tokens") or 0) for record in subset),
            "output_tokens": sum(int(record.get("output_tokens") or 0) for record in subset),
            "total_tokens": sum(int(record.get("total_tokens") or 0) for record in subset),
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "errors": dict(Counter(str(record.get("error_type")) for record in subset if record.get("error"))),
        }
    payload = {
        "updated_at": utc_now(),
        "dataset": "hle-physics-text",
        "expected_questions": 202,
        "expected_requests": 404,
        "batching": False,
        "max_output_tokens_sent": False,
        "prompt_file": str(prompt_file),
        "prompt_sha256": prompt_sha256,
        "records": len(records),
        "models": models,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    prompt_template, available_model_ids = load_prompt_template(args.prompt_file)
    prompt_sha256 = hashlib.sha256(args.prompt_file.read_bytes()).hexdigest()
    dataset = load_hle_physics_text_dataset()
    if len(dataset) != args.expect_samples:
        raise ValueError(f"Expected {args.expect_samples} HLE questions, found {len(dataset)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "workflows.jsonl"
    summary_path = args.output_dir / "summary.json"
    latest = load_latest(output_path)
    completed = {key for key, record in latest.items() if record.get("plan") and not record.get("error")}
    specs = endpoint_specs()
    clients = {
        spec["endpoint_model"]: AsyncOpenAI(
            base_url=spec["base_url"],
            api_key=spec["api_key"],
            timeout=1800.0,
            max_retries=0,
        )
        for spec in specs
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    response_format = workflow_response_format(available_model_ids)
    tasks = []
    for dataset_index, dataset_row in enumerate(dataset):
        row = dict(dataset_row)
        for spec in specs:
            key = (spec["endpoint_model"], str(row["id"]))
            if key in completed:
                continue
            tasks.append(asyncio.create_task(generate_one(
                client=clients[spec["endpoint_model"]],
                spec=spec,
                row=row,
                dataset_index=dataset_index,
                prompt_file=args.prompt_file,
                prompt_template=prompt_template,
                prompt_sha256=prompt_sha256,
                model_ids=set(available_model_ids),
                response_format=response_format,
                temperature=args.temperature,
                semaphore=semaphore,
            )))

    generated = failed = 0
    with output_path.open("a", encoding="utf-8") as stream:
        for task in asyncio.as_completed(tasks):
            record = await task
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            latest[(record["endpoint_model"], record["dataset_id"])] = record
            generated += 1
            failed += int(bool(record["error"]))
            if record["error"]:
                print(
                    f"[error] {record['endpoint_model']} {record['dataset_id']}: {record['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            if generated % 20 == 0 or generated == len(tasks):
                print(
                    f"completed={generated}/{len(tasks)} failed={failed} resumed={len(completed)}",
                    flush=True,
                )

    records = sorted(latest.values(), key=lambda item: (item["dataset_index"], item["endpoint_model"]))
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    write_summary(summary_path, records, prompt_file=args.prompt_file, prompt_sha256=prompt_sha256)
    for client in clients.values():
        await client.close()
    return 1 if any(record.get("error") for record in records) else 0


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    try:
        return asyncio.run(run(parse_args()))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
