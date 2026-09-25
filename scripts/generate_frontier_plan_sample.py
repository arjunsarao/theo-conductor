#!/usr/bin/env python3
"""Generate matched HLE Physics workflow plans with frontier models via OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from theo_conductor.data import load_hle_physics_text_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = REPO_ROOT / "CONDUCTOR_PROMPT.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hle-frontier-plan-sample"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
QUESTION_MARKER = "<USER QUESTION>"
DATASET_NAME = "hle-physics-text"

# Standard, non-batch OpenRouter list prices as of 2026-09-16. The live catalog
# is snapshotted at run time and supersedes these values when it is available.
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "gpt-6-astra",
        "display_name": "GPT-6 Astra",
        "served_model": "openai/gpt-6-astra",
        "input_usd_per_million": 10.0,
        "output_usd_per_million": 50.0,
        "pricing_url": "https://openrouter.ai/openai/gpt-6-astra",
    },
    {
        "model_id": "claude-fable-5.1",
        "display_name": "Claude Fable 5.1",
        "served_model": "anthropic/claude-fable-5.1",
        "input_usd_per_million": 10.0,
        "output_usd_per_million": 50.0,
        "pricing_url": "https://openrouter.ai/anthropic/claude-fable-5.1",
    },
    {
        "model_id": "gemini-3.8-flash",
        "display_name": "Gemini 3.8 Flash",
        "served_model": "google/gemini-3.8-flash",
        "input_usd_per_million": 0.75,
        "output_usd_per_million": 3.75,
        "pricing_url": "https://openrouter.ai/google/gemini-3.8-flash",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate matched workflow plans for a deterministic sample from the "
            "202-question text-only HLE Physics set."
        )
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--planner-model-id",
        action="append",
        choices=[spec["model_id"] for spec in MODEL_SPECS],
        help="Generate only this planner model (repeat to select several; default: all).",
    )
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--expect-samples", type=int, default=202)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=120.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=30.0)
    parser.add_argument("--request-interval-seconds", type=float, default=1.0)
    parser.add_argument("--request-jitter-seconds", type=float, default=0.5)
    return parser.parse_args()


def load_prompt_template(path: Path) -> tuple[str, list[str]]:
    template = path.read_text(encoding="utf-8")
    if template.count(QUESTION_MARKER) != 1:
        raise ValueError(f"{path} must contain exactly one {QUESTION_MARKER!r} marker")
    model_ids = list(dict.fromkeys(re.findall(r'^- model_id="([^"]+)"', template, re.MULTILINE)))
    if not model_ids:
        raise ValueError(f"Could not find worker model_id entries in {path}")
    return template, model_ids


def response_format(worker_model_ids: list[str]) -> dict[str, Any]:
    step = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1},
            "model_id": {"type": "string", "enum": worker_model_ids},
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


def validate_plan(plan: Any, worker_model_ids: set[str]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("completion is not a JSON object")
    if plan.get("difficulty") not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty is missing or invalid")
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
        if step.get("model_id") not in worker_model_ids:
            raise ValueError(f"workflow step {step_id!r} has an unavailable model_id")
        access_list = step.get("access_list")
        if not isinstance(access_list, list) or any(
            dependency != "question" and dependency not in seen for dependency in access_list
        ):
            raise ValueError(f"workflow step {step_id!r} has a forward or unknown dependency")
        seen.add(step_id)
    return plan


def json_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else None


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def token_count(usage: dict[str, Any] | None, *names: str) -> int | None:
    if not usage:
        return None
    for name in names:
        value = usage.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def nested_token_count(usage: dict[str, Any] | None, section: str, name: str) -> int | None:
    details = (usage or {}).get(section)
    return token_count(details, name) if isinstance(details, dict) else None


def estimate_cost(usage: dict[str, Any] | None, pricing: dict[str, Any]) -> float | None:
    prompt_tokens = token_count(usage, "prompt_tokens", "input_tokens")
    completion_tokens = token_count(usage, "completion_tokens", "output_tokens")
    input_rate = number(pricing.get("input_usd_per_token"))
    output_rate = number(pricing.get("output_usd_per_token"))
    if None in (prompt_tokens, completion_tokens, input_rate, output_rate):
        return None
    cached_tokens = nested_token_count(usage, "prompt_tokens_details", "cached_tokens") or 0
    cached_tokens = min(cached_tokens, prompt_tokens)
    cache_rate = number(pricing.get("cache_read_usd_per_token"))
    billed_input = (prompt_tokens - cached_tokens) * input_rate
    billed_cache = cached_tokens * (cache_rate if cache_rate is not None else input_rate)
    return billed_input + billed_cache + completion_tokens * output_rate


def fallback_pricing(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "hardcoded_fallback",
        "snapshot_at": utc_now(),
        "input_usd_per_token": spec["input_usd_per_million"] / 1_000_000,
        "output_usd_per_token": spec["output_usd_per_million"] / 1_000_000,
        "cache_read_usd_per_token": None,
        "input_usd_per_million": spec["input_usd_per_million"],
        "output_usd_per_million": spec["output_usd_per_million"],
        "pricing_url": spec["pricing_url"],
    }


async def fetch_pricing(base_url: str, specs: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    prices = {spec["model_id"]: fallback_pricing(spec) for spec in specs}
    catalog_url = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(catalog_url)
            response.raise_for_status()
            catalog = {item.get("id"): item for item in response.json().get("data", [])}
        for spec in specs:
            item = catalog.get(spec["served_model"])
            if not item:
                continue
            raw = item.get("pricing") or {}
            prompt_rate = number(raw.get("prompt"))
            completion_rate = number(raw.get("completion"))
            if prompt_rate is None or completion_rate is None:
                continue
            prices[spec["model_id"]] = {
                "source": catalog_url,
                "snapshot_at": utc_now(),
                "input_usd_per_token": prompt_rate,
                "output_usd_per_token": completion_rate,
                "cache_read_usd_per_token": number(raw.get("input_cache_read")),
                "input_usd_per_million": prompt_rate * 1_000_000,
                "output_usd_per_million": completion_rate * 1_000_000,
                "pricing_url": spec["pricing_url"],
                "catalog_pricing": raw,
                "context_length": item.get("context_length"),
                "top_provider": item.get("top_provider"),
            }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        print(f"[warning] Could not snapshot live OpenRouter prices: {exc}", file=sys.stderr)
    return prices


def choose_questions(
    dataset: Any,
    *,
    path: Path,
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sample_count") != sample_count or payload.get("seed") != seed:
            raise ValueError(
                f"Existing {path} uses sample_count={payload.get('sample_count')} and "
                f"seed={payload.get('seed')}; use a new output directory"
            )
        indices = payload.get("dataset_indices")
    else:
        indices = sorted(random.Random(seed).sample(range(len(dataset)), sample_count))
        payload = {
            "created_at": utc_now(),
            "dataset": DATASET_NAME,
            "dataset_size": len(dataset),
            "sample_count": sample_count,
            "seed": seed,
            "dataset_indices": indices,
            "dataset_ids": [str(dataset[index]["id"]) for index in indices],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not isinstance(indices, list) or len(indices) != sample_count:
        raise ValueError(f"Malformed question selection in {path}")
    return [{**dict(dataset[index]), "dataset_index": index} for index in indices]


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
                latest[(str(record["planner_model_id"]), str(record["dataset_id"]))] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: malformed record: {exc}") from exc
    return latest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed record: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(record)
    return records


class RequestPacer:
    """Serialize request starts with a minimum interval and random jitter."""

    def __init__(self, *, interval_seconds: float, jitter_seconds: float, seed: int):
        self.interval_seconds = interval_seconds
        self.jitter_seconds = jitter_seconds
        self._random = random.Random(seed)
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> float:
        """Wait for the next globally permitted start and return delay in milliseconds."""
        entered = time.perf_counter()
        async with self._lock:
            delay = max(0.0, self._next_start - time.perf_counter())
            if delay:
                await asyncio.sleep(delay)
            started = time.perf_counter()
            jitter = self._random.uniform(0.0, self.jitter_seconds)
            self._next_start = started + self.interval_seconds + jitter
        return (time.perf_counter() - entered) * 1000


async def generate_one(
    *,
    client: AsyncOpenAI,
    spec: dict[str, Any],
    pricing: dict[str, Any],
    row: dict[str, Any],
    prompt_file: Path,
    prompt_template: str,
    prompt_sha256: str,
    worker_model_ids: set[str],
    output_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    attempts: int,
    retry_delay_seconds: float,
    retry_jitter_seconds: float,
    request_seed: int,
    pacer: RequestPacer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    question = str(row["question"])
    prompt = prompt_template.replace(QUESTION_MARKER, question)
    attempt_records: list[dict[str, Any]] = []
    final_record: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        completion: Any | None = None
        request_started_at = utc_now()
        started = time.perf_counter()
        try:
            queue_ms = await pacer.wait()
            request_started_at = utc_now()
            request_started = time.perf_counter()
            completion = await client.chat.completions.create(
                model=spec["served_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=output_format,
                seed=request_seed,
                extra_body={"usage": {"include": True}},
            )
            latency_ms = (time.perf_counter() - request_started) * 1000
            choice = completion.choices[0]
            raw_completion = choice.message.content or ""
            plan = validate_plan(json.loads(raw_completion), worker_model_ids)
            error_type = None
            error = None
            finish_reason = choice.finish_reason
        except Exception as exc:
            queue_ms = locals().get("queue_ms", 0.0)
            latency_ms = (time.perf_counter() - started) * 1000 - queue_ms
            raw_completion = ""
            finish_reason = None
            if completion is not None and getattr(completion, "choices", None):
                choice = completion.choices[0]
                raw_completion = choice.message.content or ""
                finish_reason = choice.finish_reason
            plan = None
            error_type = type(exc).__name__
            error = f"{error_type}: {exc}"

        usage = json_dump(getattr(completion, "usage", None)) if completion is not None else None
        provider_cost = number((usage or {}).get("cost"))
        completion_dump = json_dump(completion) if completion is not None else None
        record = {
            "timestamp": utc_now(),
            "request_started_at": request_started_at,
            "dataset": DATASET_NAME,
            "dataset_id": str(row["id"]),
            "dataset_index": row["dataset_index"],
            "question": question,
            "gold_answer": row.get("answer"),
            "reference_answer": row.get("reference_answer"),
            "answer_type": row.get("answer_type"),
            "subject": row.get("subject"),
            "planner_model_id": spec["model_id"],
            "planner_display_name": spec["display_name"],
            "requested_model": spec["served_model"],
            "response_model": getattr(completion, "model", None),
            "provider": getattr(completion, "provider", None),
            "response_id": getattr(completion, "id", None),
            "system_fingerprint": getattr(completion, "system_fingerprint", None),
            "prompt_file": str(prompt_file),
            "prompt_sha256": prompt_sha256,
            "rendered_prompt_sha256": sha256_text(prompt),
            "request_parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": request_seed,
                "response_format_sha256": sha256_text(
                    json.dumps(output_format, sort_keys=True, separators=(",", ":"))
                ),
            },
            "plan": plan,
            "conductor_completion": raw_completion,
            "attempt": attempt,
            "queue_latency_ms": queue_ms,
            "latency_ms": max(latency_ms, 0.0),
            "usage": usage,
            "input_tokens": token_count(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": token_count(usage, "completion_tokens", "output_tokens"),
            "total_tokens": token_count(usage, "total_tokens"),
            "cached_input_tokens": nested_token_count(
                usage, "prompt_tokens_details", "cached_tokens"
            ),
            "reasoning_tokens": nested_token_count(
                usage, "completion_tokens_details", "reasoning_tokens"
            ),
            "provider_reported_cost_usd": provider_cost,
            "list_price_estimated_cost_usd": estimate_cost(usage, pricing),
            "pricing": pricing,
            "finish_reason": finish_reason,
            "native_finish_reason": (completion_dump or {}).get("native_finish_reason"),
            "error_type": error_type,
            "error": error,
        }
        attempt_records.append(record)
        final_record = record
        if error is None:
            break
        if attempt < attempts and (retry_delay_seconds or retry_jitter_seconds):
            retry_jitter = random.Random(
                f"{request_seed}:{spec['model_id']}:{row['id']}:{attempt}"
            ).uniform(0.0, retry_jitter_seconds)
            retry_wait = retry_delay_seconds + retry_jitter
            print(
                f"[retry] model={spec['model_id']} question={row['dataset_index']} "
                f"attempt={attempt}/{attempts} error={error_type} "
                f"waiting_s={retry_wait:.2f}",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(retry_wait)
    assert final_record is not None
    return final_record, attempt_records


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("plan") is not None and not record.get("error")]
    latencies = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    provider_costs = [
        float(record["provider_reported_cost_usd"])
        for record in records
        if record.get("provider_reported_cost_usd") is not None
    ]
    estimated_costs = [
        float(record["list_price_estimated_cost_usd"])
        for record in records
        if record.get("list_price_estimated_cost_usd") is not None
    ]
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in records)
    total_latency_ms = sum(latencies)
    return {
        "records": len(records),
        "valid": len(valid),
        "failed": len(records) - len(valid),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "output_tokens": output_tokens,
        "total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "cached_input_tokens": sum(int(record.get("cached_input_tokens") or 0) for record in records),
        "reasoning_tokens": sum(int(record.get("reasoning_tokens") or 0) for record in records),
        "provider_reported_cost_usd": sum(provider_costs) if provider_costs else None,
        "list_price_estimated_cost_usd": sum(estimated_costs) if estimated_costs else None,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "output_tokens_per_second": (
            output_tokens * 1000 / total_latency_ms if total_latency_ms > 0 else None
        ),
        "finish_reasons": dict(Counter(str(record.get("finish_reason")) for record in records)),
        "providers": dict(Counter(str(record.get("provider")) for record in records)),
        "errors": dict(Counter(str(record.get("error_type")) for record in records if record.get("error"))),
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> int:
    if args.sample_count <= 0 or args.concurrency <= 0 or args.max_tokens <= 0 or args.attempts <= 0:
        raise ValueError("sample-count, concurrency, max-tokens, and attempts must be positive")
    for name in (
        "retry_delay_seconds",
        "retry_jitter_seconds",
        "request_interval_seconds",
        "request_jitter_seconds",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")

    prompt_template, worker_model_ids = load_prompt_template(args.prompt_file)
    prompt_sha256 = hashlib.sha256(args.prompt_file.read_bytes()).hexdigest()
    dataset = load_hle_physics_text_dataset()
    if len(dataset) != args.expect_samples:
        raise ValueError(f"Expected {args.expect_samples} HLE questions, found {len(dataset)}")
    if args.sample_count > len(dataset):
        raise ValueError("sample-count cannot exceed the dataset size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_questions(
        dataset,
        path=args.output_dir / "selected_questions.json",
        sample_count=args.sample_count,
        seed=args.seed,
    )
    selected_specs = tuple(
        spec for spec in MODEL_SPECS
        if not args.planner_model_id or spec["model_id"] in args.planner_model_id
    )
    prices = await fetch_pricing(args.base_url, selected_specs)
    write_json_atomic(args.output_dir / "pricing.json", prices)

    plans_path = args.output_dir / "plans.jsonl"
    requests_path = args.output_dir / "requests.jsonl"
    latest = load_latest(plans_path)
    completed = {
        key for key, record in latest.items() if record.get("plan") is not None and not record.get("error")
    }
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=httpx.Timeout(1800.0, connect=15.0),
        max_retries=0,
    )
    pacer = RequestPacer(
        interval_seconds=args.request_interval_seconds,
        jitter_seconds=args.request_jitter_seconds,
        seed=args.seed,
    )
    output_format = response_format(worker_model_ids)
    work_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in selected:
        for spec in selected_specs:
            key = (str(spec["model_id"]), str(row["id"]))
            if key in completed:
                continue
            work_items.append((row, spec))

    work_queue: asyncio.Queue[tuple[dict[str, Any], dict[str, Any]] | None] = asyncio.Queue()
    result_queue: asyncio.Queue[tuple[dict[str, Any], list[dict[str, Any]]]] = asyncio.Queue()
    for item in work_items:
        work_queue.put_nowait(item)
    worker_count = min(args.concurrency, len(work_items))
    for _ in range(worker_count):
        work_queue.put_nowait(None)

    async def queue_worker() -> None:
        while True:
            item = await work_queue.get()
            try:
                if item is None:
                    return
                row, spec = item
                result = await generate_one(
                    client=client,
                    spec=spec,
                    pricing=prices[spec["model_id"]],
                    row=row,
                    prompt_file=args.prompt_file,
                    prompt_template=prompt_template,
                    prompt_sha256=prompt_sha256,
                    worker_model_ids=set(worker_model_ids),
                    output_format=output_format,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    attempts=args.attempts,
                    retry_delay_seconds=args.retry_delay_seconds,
                    retry_jitter_seconds=args.retry_jitter_seconds,
                    request_seed=args.seed,
                    pacer=pacer,
                )
                await result_queue.put(result)
            finally:
                work_queue.task_done()

    workers = [asyncio.create_task(queue_worker()) for _ in range(worker_count)]

    generated = failed = 0
    with plans_path.open("a", encoding="utf-8") as plans_stream, requests_path.open(
        "a", encoding="utf-8"
    ) as requests_stream:
        for _ in range(len(work_items)):
            record, attempt_records = await result_queue.get()
            for attempt_record in attempt_records:
                requests_stream.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
            requests_stream.flush()
            plans_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            plans_stream.flush()
            latest[(record["planner_model_id"], record["dataset_id"])] = record
            generated += 1
            failed += int(bool(record.get("error")))
            print(
                f"completed={generated}/{len(work_items)} failed={failed} "
                f"model={record['planner_model_id']} question={record['dataset_index']} "
                f"latency_s={record['latency_ms'] / 1000:.2f} "
                f"tokens={record.get('total_tokens')} cost={record.get('provider_reported_cost_usd')}",
                flush=True,
            )

    await work_queue.join()
    await asyncio.gather(*workers)

    await client.close()
    records = sorted(
        latest.values(),
        key=lambda item: (int(item["dataset_index"]), str(item["planner_model_id"])),
    )
    write_jsonl_atomic(plans_path, records)
    request_records = load_jsonl(requests_path)
    selected_ids = {str(row["id"]) for row in selected}
    models: dict[str, Any] = {}
    for spec in selected_specs:
        subset = [record for record in records if record["planner_model_id"] == spec["model_id"]]
        request_subset = [
            record
            for record in request_records
            if record.get("planner_model_id") == spec["model_id"]
        ]
        model_question_ids = {record["dataset_id"] for record in subset}
        models[spec["model_id"]] = {
            "display_name": spec["display_name"],
            "served_model": spec["served_model"],
            "same_question_set_complete": model_question_ids == selected_ids,
            "pricing": prices[spec["model_id"]],
            **summarize_model(subset),
            "request_attempts": summarize_model(request_subset),
        }
    total_provider_costs = [
        float(record["provider_reported_cost_usd"])
        for record in request_records
        if record.get("provider_reported_cost_usd") is not None
    ]
    total_estimated_costs = [
        float(record["list_price_estimated_cost_usd"])
        for record in request_records
        if record.get("list_price_estimated_cost_usd") is not None
    ]
    summary = {
        "updated_at": utc_now(),
        "dataset": DATASET_NAME,
        "dataset_size": len(dataset),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "planner_model_ids": [spec["model_id"] for spec in selected_specs],
        "expected_requests": args.sample_count * len(selected_specs),
        "records": len(records),
        "request_attempts": len(request_records),
        "all_models_share_question_set": all(
            model["same_question_set_complete"] for model in models.values()
        ),
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": prompt_sha256,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "request_interval_seconds": args.request_interval_seconds,
        "request_jitter_seconds": args.request_jitter_seconds,
        "retry_delay_seconds": args.retry_delay_seconds,
        "retry_jitter_seconds": args.retry_jitter_seconds,
        "cost_scope": "all logged request attempts, including retries",
        "provider_reported_cost_usd": sum(total_provider_costs) if total_provider_costs else None,
        "list_price_estimated_cost_usd": sum(total_estimated_costs) if total_estimated_costs else None,
        "models": models,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if any(record.get("error") for record in records) else 0


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    try:
        return asyncio.run(run(parse_args()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
