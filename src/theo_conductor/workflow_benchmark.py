from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from .benchmark import (
    extract_final_answer,
    judge_records,
    judge_records_with_checkpoints,
    summarize_records,
    write_results_atomic,
)
from .models.openai_compat import OpenAICompatibleClient
from .models.registry import ModelRegistry
from .plan import PLAN_FILENAME, load_plan_records
from .runner import Runner
from .scheduler import topological_sort
from .schema import Task
from .validate import validate_task


RESULTS_FILENAME = "results.jsonl"
MODEL_ID = "theo-conductor"
DEFAULT_WORKFLOW_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
# This OpenRouter alias deliberately follows the current DeepSeek Flash release.
DEFAULT_WORKFLOW_JUDGE_MODEL = "~deepseek/deepseek-flash-latest"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execution_config_hash(
    *,
    config_bytes: bytes,
    max_worker_tokens: int | None,
    worker_temperature: float,
    worker_token_limit_mode: str = "fixed",
    batch_workers: bool = False,
    worker_batch_size: int = 50_000,
) -> str:
    """Identify every setting that can change a workflow execution result."""
    settings = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "max_worker_tokens": max_worker_tokens,
        "worker_token_limit_mode": worker_token_limit_mode,
        "worker_temperature": worker_temperature,
        "runner_contract": 1,
    }
    # Preserve hashes (and therefore resumability) for the established
    # non-batched runner. Batching has its own result identity.
    if batch_workers:
        settings.update(
            batch_workers=True,
            worker_batch_size=worker_batch_size,
            runner_contract=2,
        )
    return _json_hash(settings)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["example_id"]),
        str(record["plan_sha256"]),
        str(record["execution_config_sha256"]),
    )


def load_results(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    records_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = _record_key(record)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid workflow result at {path}:{line_number}: {exc}") from exc
            records_by_key[key] = record
    return list(records_by_key.values()), set(records_by_key)


def _usage_totals(worker_outputs: dict[str, dict[str, Any]]) -> dict[str, int | float | None]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost = 0.0
    saw_prompt = saw_completion = saw_total = saw_cost = False
    for output in worker_outputs.values():
        usage = output.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        cost = usage.get("estimated_cost_usd")
        if isinstance(prompt, (int, float)) and not isinstance(prompt, bool):
            prompt_tokens += int(prompt)
            saw_prompt = True
        if isinstance(completion, (int, float)) and not isinstance(completion, bool):
            completion_tokens += int(completion)
            saw_completion = True
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            total_tokens += int(total)
            saw_total = True
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            estimated_cost += float(cost)
            saw_cost = True
        for tool_call in output.get("tool_calls") or []:
            tool_usage = tool_call.get("usage")
            if not isinstance(tool_usage, dict):
                continue
            tool_prompt = tool_usage.get("prompt_tokens", tool_usage.get("input_tokens"))
            tool_completion = tool_usage.get("completion_tokens", tool_usage.get("output_tokens"))
            tool_total = tool_usage.get("total_tokens")
            tool_cost = tool_usage.get("estimated_cost_usd")
            if isinstance(tool_prompt, (int, float)) and not isinstance(tool_prompt, bool):
                prompt_tokens += int(tool_prompt)
                saw_prompt = True
            if isinstance(tool_completion, (int, float)) and not isinstance(tool_completion, bool):
                completion_tokens += int(tool_completion)
                saw_completion = True
            if isinstance(tool_total, (int, float)) and not isinstance(tool_total, bool):
                total_tokens += int(tool_total)
                saw_total = True
            if isinstance(tool_cost, (int, float)) and not isinstance(tool_cost, bool):
                estimated_cost += float(tool_cost)
                saw_cost = True
    return {
        "prompt_tokens": prompt_tokens if saw_prompt else None,
        "completion_tokens": completion_tokens if saw_completion else None,
        "total_tokens": total_tokens if saw_total else None,
        "estimated_cost_usd": estimated_cost if saw_cost else None,
    }


def _base_result(
    plan_record: dict[str, Any],
    *,
    plan_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "display_name": "Theo Conductor",
        "example_id": str(plan_record.get("dataset_id")),
        "benchmark_position": plan_record.get("dataset_index"),
        "subject": plan_record.get("subject"),
        "question": plan_record.get("question"),
        "gold_answer": plan_record.get("gold_answer"),
        "reference_answer": plan_record.get("reference_answer"),
        "answer_type": plan_record.get("answer_type"),
        "plan": plan_record.get("plan"),
        "plan_sha256": plan_sha256,
        "execution_config_sha256": config_sha256,
        "response": None,
        "extracted_answer": None,
        "correct": None,
        "error": None,
        "error_type": None,
        "worker_outputs": {},
        "workflow_runtime_ms": None,
        "workflow_peak_concurrency": None,
        "workflow_steps": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "estimated_cost_usd": None,
        "latency_ms": None,
    }


async def run_workflow_benchmark(
    *,
    registry: ModelRegistry,
    plan_records: Sequence[dict[str, Any]],
    results_path: Path,
    execution_config_sha256: str,
    concurrency: int = 4,
    max_worker_tokens: int | None = None,
    use_model_output_limits: bool = False,
    worker_temperature: float = 0.2,
    retry_failures: bool = False,
    batch_workers: bool = False,
    worker_batch_size: int = 50_000,
    judge_client: Any | None = None,
    judge_model: str = DEFAULT_WORKFLOW_JUDGE_MODEL,
    fallback_judge_client: Any | None = None,
    fallback_judge_model: str | None = None,
    judge_concurrency: int = 8,
    judge_max_tokens: int = 8192,
    judge_attempts: int = 3,
) -> list[dict[str, Any]]:
    """Execute conductor plans with item-level isolation and resumable JSONL output."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if max_worker_tokens is not None and max_worker_tokens <= 0:
        raise ValueError("max-worker-tokens must be positive")
    if worker_batch_size <= 0:
        raise ValueError("worker-batch-size must be positive")
    if judge_concurrency <= 0:
        raise ValueError("judge-concurrency must be positive")
    if judge_max_tokens <= 0:
        raise ValueError("judge-max-tokens must be positive")
    if judge_attempts <= 0:
        raise ValueError("judge-attempts must be positive")
    if fallback_judge_client is not None and not fallback_judge_model:
        raise ValueError("fallback judge client requires a fallback judge model")
    if batch_workers and any(
        spec.supports_tools for spec in registry._models.values()
    ):
        # Tool calls require iterative continuations, which native batch jobs cannot provide.
        batch_workers = False
    if batch_workers and judge_client is not None:
        raise ValueError("--batch-workers cannot judge immediately; omit it for per-workflow judging")
    if batch_workers:
        return await _run_batched_workflow_benchmark(
            registry=registry,
            plan_records=plan_records,
            results_path=results_path,
            execution_config_sha256=execution_config_sha256,
            concurrency=concurrency,
            max_worker_tokens=max_worker_tokens,
            use_model_output_limits=use_model_output_limits,
            worker_temperature=worker_temperature,
            retry_failures=retry_failures,
            worker_batch_size=worker_batch_size,
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_records, completed = load_results(results_path)
    existing_by_key = {_record_key(record): record for record in all_records}
    selected_keys: set[tuple[str, str, str]] = set()
    semaphore = asyncio.Semaphore(concurrency)
    judge_semaphore = asyncio.Semaphore(judge_concurrency)
    write_lock = asyncio.Lock()

    async def judge_completed_record(record: dict[str, Any]) -> None:
        """Grade a single workflow before its completion checkpoint is written."""
        if judge_client is None:
            return
        async with judge_semaphore:
            await judge_records(
                [record],
                client=judge_client,
                judge_model=judge_model,
                fallback_client=fallback_judge_client,
                fallback_judge_model=fallback_judge_model or "",
                concurrency=1,
                batch_size=1,
                max_tokens=judge_max_tokens,
                attempts=judge_attempts,
            )

    async def execute(plan_record: dict[str, Any]) -> None:
        plan = plan_record.get("plan")
        plan_sha256 = _json_hash(plan)
        key = (str(plan_record.get("dataset_id")), plan_sha256, execution_config_sha256)
        selected_keys.add(key)
        if key in completed and not (
            retry_failures and existing_by_key[key].get("error") is not None
        ):
            return

        record = _base_result(
            plan_record,
            plan_sha256=plan_sha256,
            config_sha256=execution_config_sha256,
        )
        try:
            if not isinstance(plan, dict) or plan_record.get("error"):
                detail = plan_record.get("error") or "record does not contain a valid plan"
                raise ValueError(f"Plan unavailable: {detail}")
            # Frontier planner records keep the question beside the compact
            # structured workflow rather than duplicating it inside `plan`.
            # The runtime Task contract still owns the question, so restore it
            # from the canonical benchmark record before validation/execution.
            task = Task.from_dict({**plan, "question": plan_record.get("question", "")})
            runner = Runner(
                model_registry=registry,
                max_worker_tokens=max_worker_tokens,
                use_model_output_limits=use_model_output_limits,
                worker_temperature=worker_temperature,
            )
            async with semaphore:
                result = await runner.run(task)
            outputs = {
                step_id: output.model_dump(mode="json")
                for step_id, output in result.outputs.items()
            }
            final_step_id = task.workflow[-1].step_id
            response = outputs[final_step_id]["text"]
            record.update(
                response=response,
                extracted_answer=extract_final_answer(response),
                worker_outputs=outputs,
                workflow_runtime_ms=result.observed_wall_time_ms,
                workflow_peak_concurrency=result.observed_peak_concurrency,
                workflow_steps=len(outputs),
                latency_ms=result.observed_wall_time_ms,
                **_usage_totals(outputs),
            )
        except Exception as exc:  # A failed workflow must not abort the evaluation set.
            record["error_type"] = type(exc).__name__
            record["error"] = f"{type(exc).__name__}: {exc}"

        # Judging happens after the runner has completed (or failed) but before
        # the record is visible to a resume run.
        await judge_completed_record(record)

        async with write_lock:
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            all_records.append(record)
            completed.add(key)
            existing_by_key[key] = record
            current = sum(item in completed for item in selected_keys)
            print(
                f"[workflow {current}/{len(plan_records)}] {record['example_id']}: "
                f"{'error' if record['error'] else 'completed'}",
                file=sys.stderr,
                flush=True,
            )

    await asyncio.gather(*(execute(record) for record in plan_records))
    latest_records, _ = load_results(results_path)
    return [record for record in latest_records if _record_key(record) in selected_keys]


async def _run_batched_workflow_benchmark(
    *,
    registry: ModelRegistry,
    plan_records: Sequence[dict[str, Any]],
    results_path: Path,
    execution_config_sha256: str,
    concurrency: int,
    max_worker_tokens: int | None,
    use_model_output_limits: bool,
    worker_temperature: float,
    retry_failures: bool,
    worker_batch_size: int,
) -> list[dict[str, Any]]:
    """Pool ready steps across workflows by dependency layer and worker model."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_records, completed = load_results(results_path)
    existing_by_key = {_record_key(record): record for record in all_records}
    selected_keys: set[tuple[str, str, str]] = set()
    states: list[dict[str, Any]] = []
    fallback_semaphore = asyncio.Semaphore(concurrency)

    def persist(state: dict[str, Any]) -> None:
        record = state["record"]
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        all_records.append(record)
        completed.add(state["key"])
        existing_by_key[state["key"]] = record
        current = sum(key in completed for key in selected_keys)
        print(
            f"[workflow {current}/{len(plan_records)}] {record['example_id']}: "
            f"{'error' if record['error'] else 'completed'}",
            file=sys.stderr,
            flush=True,
        )

    def finish(state: dict[str, Any], error: Exception | None = None) -> None:
        record = state["record"]
        outputs = {
            step_id: output.model_dump(mode="json")
            for step_id, output in state["outputs"].items()
        }
        elapsed_ms = (time.perf_counter() - state["started_at"]) * 1000
        record.update(
            worker_outputs=outputs,
            workflow_runtime_ms=elapsed_ms,
            workflow_peak_concurrency=state["peak_concurrency"],
            workflow_steps=len(outputs),
            latency_ms=elapsed_ms,
            **_usage_totals(outputs),
        )
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error"] = f"{type(error).__name__}: {error}"
        else:
            final_step_id = state["task"].workflow[-1].step_id
            response = outputs[final_step_id]["text"]
            record.update(
                response=response,
                extracted_answer=extract_final_answer(response),
            )
        state["finished"] = True
        persist(state)

    for plan_record in plan_records:
        plan = plan_record.get("plan")
        plan_sha256 = _json_hash(plan)
        key = (str(plan_record.get("dataset_id")), plan_sha256, execution_config_sha256)
        selected_keys.add(key)
        if key in completed and not (
            retry_failures and existing_by_key[key].get("error") is not None
        ):
            continue
        record = _base_result(
            plan_record,
            plan_sha256=plan_sha256,
            config_sha256=execution_config_sha256,
        )
        state: dict[str, Any] = {
            "key": key,
            "record": record,
            "outputs": {},
            "started_at": time.perf_counter(),
            "peak_concurrency": 0,
            "finished": False,
        }
        try:
            if not isinstance(plan, dict) or plan_record.get("error"):
                detail = plan_record.get("error") or "record does not contain a valid plan"
                raise ValueError(f"Plan unavailable: {detail}")
            task = Task.from_dict({**plan, "question": plan_record.get("question", "")})
            validate_task(task, registry)
            state.update(
                task=task,
                layers=topological_sort(task),
                runner=Runner(
                    model_registry=registry,
                    max_worker_tokens=max_worker_tokens,
                    use_model_output_limits=use_model_output_limits,
                    worker_temperature=worker_temperature,
                ),
            )
            states.append(state)
        except Exception as exc:
            state.update(task=None, layers=[], runner=None)
            finish(state, exc)

    max_layers = max((len(state["layers"]) for state in states), default=0)
    for layer_index in range(max_layers):
        buckets: dict[int | str, list[tuple[dict[str, Any], Any, Any, dict[str, Any]]]] = defaultdict(list)
        for state in states:
            if state["finished"] or layer_index >= len(state["layers"]):
                continue
            layer = state["layers"][layer_index]
            state["peak_concurrency"] = max(state["peak_concurrency"], len(layer))
            try:
                for step in layer:
                    spec, request = state["runner"].prepare_step(
                        step, state["task"], state["outputs"]
                    )
                    buckets[step.model_id].append((state, step, spec, request))
            except Exception as exc:
                finish(state, exc)

        async def run_fallback(request: dict[str, Any], client: Any) -> Any:
            async with fallback_semaphore:
                try:
                    return await client.generate(**request)
                except Exception as exc:
                    return exc

        async def run_chunk(items: list[tuple[dict[str, Any], Any, Any, dict[str, Any]]]):
            client = items[0][2].client
            if getattr(client, "supports_batch", False):
                try:
                    responses = await client.generate_batch([item[3] for item in items])
                except Exception as exc:
                    responses = [exc] * len(items)
            else:
                responses = await asyncio.gather(
                    *(run_fallback(item[3], client) for item in items)
                )
            return list(zip(items, responses))

        chunks = [
            items[start : start + worker_batch_size]
            for items in buckets.values()
            for start in range(0, len(items), worker_batch_size)
        ]
        chunk_results = await asyncio.gather(*(run_chunk(chunk) for chunk in chunks))
        for chunk_result in chunk_results:
            for (state, step, spec, _), response in chunk_result:
                if state["finished"]:
                    continue
                if isinstance(response, Exception):
                    finish(state, response)
                    continue
                try:
                    state["outputs"][step.step_id] = state["runner"].complete_step(
                        step, spec, response
                    )
                except Exception as exc:
                    finish(state, exc)

        for state in states:
            if not state["finished"] and layer_index == len(state["layers"]) - 1:
                finish(state)

    latest_records, _ = load_results(results_path)
    return [record for record in latest_records if _record_key(record) in selected_keys]


def summarize_workflows(
    records: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    base = summarize_records(records, bootstrap_samples=bootstrap_samples, seed=seed)
    costs = [float(record["estimated_cost_usd"]) for record in records if record.get("estimated_cost_usd") is not None]
    runtimes = [float(record["workflow_runtime_ms"]) for record in records if record.get("workflow_runtime_ms") is not None]
    steps = [int(record["workflow_steps"]) for record in records if record.get("workflow_steps") is not None]
    return {
        "workflows": len(records),
        "completed": sum(record.get("error") is None for record in records),
        "failed": sum(record.get("error") is not None for record in records),
        "judged": sum(isinstance(record.get("judge_correct"), bool) for record in records),
        "total_estimated_cost_usd": sum(costs) if costs else None,
        "mean_workflow_runtime_ms": mean(runtimes) if runtimes else None,
        "mean_workflow_steps": mean(steps) if steps else None,
        **base,
    }


def _load_plans(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / PLAN_FILENAME
    if not path.is_file():
        raise ValueError(f"Plan file not found: {path}")
    return load_plan_records(path)


def select_text_only_plans(
    plan_records: Sequence[dict[str, Any]],
    text_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter mixed HLE plans by canonical dataset IDs and restore metadata."""
    metadata = {str(row["id"]): row for row in text_rows}
    selected: list[dict[str, Any]] = []
    for record in plan_records:
        dataset_id = str(record.get("dataset_id") or "")
        row = metadata.get(dataset_id)
        if row is None:
            continue
        enriched = dict(record)
        enriched["subject"] = row.get("subject")
        enriched["is_multimodal"] = False
        selected.append(enriched)
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="theo-workflow-benchmark",
        description="Execute, judge, and summarize pregenerated conductor workflows.",
    )
    parser.add_argument("--plans", type=Path, required=True, help="plans.jsonl or its containing directory")
    parser.add_argument(
        "--planner-model-id",
        help="Execute only plans produced by this conductor model ID.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N plan records")
    parser.add_argument("--max-samples", type=int, help="Execute only the first N plan records")
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Include image-dependent HLE plans (default: canonical text-only questions)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent workflows, or concurrent fallback requests with --batch-workers",
    )
    parser.add_argument(
        "--batch-workers",
        action="store_true",
        help=(
            "Pool ready workflow steps by dependency layer and model. Clients configured with "
            "native_batch use a provider batch job; other clients use bounded concurrent requests."
        ),
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=50_000,
        help="Maximum requests in one provider-native worker batch (default: 50000)",
    )
    token_limit = parser.add_mutually_exclusive_group()
    token_limit.add_argument(
        "--max-worker-tokens",
        type=int,
        help="Use one fixed per-worker output cap",
    )
    token_limit.add_argument(
        "--use-model-output-limits",
        action="store_true",
        help="Use each model's benchmark-grounded max_output_tokens (default)",
    )
    token_limit.add_argument(
        "--use-model-context-limit",
        action="store_true",
        help="Use each worker model's configured context_length as its max_tokens request",
    )
    parser.add_argument("--worker-temperature", type=float, default=0.2)
    parser.add_argument("--retry-failures", action="store_true", help="Rerun checkpointed workflow errors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--judge-immediately",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Grade each workflow before its completed result is checkpointed (default).",
    )
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL", DEFAULT_WORKFLOW_JUDGE_BASE_URL))
    parser.add_argument("--judge-api-key", default=os.getenv("JUDGE_API_KEY", os.getenv("OPENROUTER_API_KEY", "change-this")))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", DEFAULT_WORKFLOW_JUDGE_MODEL))
    parser.add_argument("--fallback-judge-base-url", default=os.getenv("FALLBACK_JUDGE_BASE_URL", ""))
    parser.add_argument("--fallback-judge-api-key", default=os.getenv("FALLBACK_JUDGE_API_KEY", ""))
    parser.add_argument("--fallback-judge-model", default=os.getenv("FALLBACK_JUDGE_MODEL", ""))
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--judge-batch-size", type=int, default=10)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-attempts", type=int, default=3)
    parser.add_argument("--judge-checkpoint-size", type=int, default=25)
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    load_dotenv()
    args = parse_args(argv)
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("max-samples must be non-negative")
    registry = ModelRegistry.from_yaml_file(args.config)
    if args.max_worker_tokens is not None:
        token_limit_mode = "fixed"
        max_worker_tokens = args.max_worker_tokens
    elif args.use_model_context_limit:
        token_limit_mode = "model_context_length"
        max_worker_tokens = None
    else:
        token_limit_mode = "model_output_limits"
        max_worker_tokens = None
    plan_records = _load_plans(args.plans)
    if args.planner_model_id:
        plan_records = [
            record for record in plan_records
            if record.get("planner_model_id") == args.planner_model_id
        ]
        if not plan_records:
            raise ValueError(f"No plans found for planner model {args.planner_model_id!r}")
    unfiltered_plan_count = len(plan_records)
    if not args.include_multimodal:
        from .data import load_conductor_dataset

        text_dataset = load_conductor_dataset("hle-text", seed=args.seed)
        text_rows = [dict(text_dataset[index]) for index in range(len(text_dataset))]
        plan_records = select_text_only_plans(plan_records, text_rows)
        if not plan_records:
            raise ValueError("No canonical text-only HLE IDs matched the supplied plans")
    filtered_plan_count = len(plan_records)
    stop = None if args.max_samples is None else args.offset + args.max_samples
    plan_records = plan_records[args.offset : stop]
    config_hash = execution_config_hash(
        config_bytes=args.config.read_bytes(),
        max_worker_tokens=max_worker_tokens,
        worker_temperature=args.worker_temperature,
        worker_token_limit_mode=token_limit_mode,
        batch_workers=args.batch_workers,
        worker_batch_size=args.worker_batch_size,
    )
    results_path = args.output_dir / RESULTS_FILENAME
    judge = fallback = None
    if args.judge:
        judge = OpenAICompatibleClient(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
            max_retries=0,
        )
        if args.fallback_judge_model:
            fallback = OpenAICompatibleClient(
                base_url=args.fallback_judge_base_url or args.judge_base_url,
                api_key=args.fallback_judge_api_key or args.judge_api_key,
                model=args.fallback_judge_model,
                max_retries=0,
            )
    records = await run_workflow_benchmark(
        registry=registry,
        plan_records=plan_records,
        results_path=results_path,
        execution_config_sha256=config_hash,
        concurrency=args.concurrency,
        max_worker_tokens=max_worker_tokens,
        use_model_output_limits=token_limit_mode == "model_output_limits",
        worker_temperature=args.worker_temperature,
        retry_failures=args.retry_failures,
        batch_workers=args.batch_workers,
        worker_batch_size=args.worker_batch_size,
        judge_client=judge if args.judge_immediately else None,
        judge_model=args.judge_model,
        fallback_judge_client=fallback if args.judge_immediately else None,
        fallback_judge_model=args.fallback_judge_model or None,
        judge_concurrency=args.judge_concurrency,
        judge_max_tokens=args.judge_max_tokens,
        judge_attempts=args.judge_attempts,
    )
    all_records, _ = load_results(results_path)
    selected_keys = {_record_key(record) for record in records}
    # Judging checkpoints rewrite all result configurations in the file, so
    # select the current run from that exact object graph before mutating it.
    records = [record for record in all_records if _record_key(record) in selected_keys]
    if args.judge and not args.judge_immediately:
        assert judge is not None
        await judge_records_with_checkpoints(
            records,
            all_records=all_records,
            results_path=results_path,
            client=judge,
            judge_model=args.judge_model,
            fallback_client=fallback,
            fallback_judge_model=args.fallback_judge_model or "",
            concurrency=args.judge_concurrency,
            batch_size=args.judge_batch_size,
            max_tokens=args.judge_max_tokens,
            attempts=args.judge_attempts,
            checkpoint_size=args.judge_checkpoint_size,
        )
    summary = {
        "plans": str(args.plans),
        "planner_model_id": args.planner_model_id,
        "dataset": "hle-all" if args.include_multimodal else "hle-text",
        "text_only": not args.include_multimodal,
        "unfiltered_plan_count": unfiltered_plan_count,
        "available_plan_count": filtered_plan_count,
        "config": str(args.config),
        "execution_config_sha256": config_hash,
        "offset": args.offset,
        "max_samples": args.max_samples,
        "concurrency": args.concurrency,
        "batch_workers": args.batch_workers,
        "worker_batch_size": args.worker_batch_size if args.batch_workers else None,
        "worker_temperature": args.worker_temperature,
        "max_worker_tokens": max_worker_tokens,
        "worker_token_limit_mode": token_limit_mode,
        "retry_failures": args.retry_failures,
        "judge_enabled": args.judge,
        "judge_immediately": args.judge_immediately if args.judge else False,
        "judge_model": args.judge_model if args.judge else None,
        "fallback_judge_model": args.fallback_judge_model if args.judge else None,
        **summarize_workflows(records, bootstrap_samples=args.bootstrap_samples, seed=args.seed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
