from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models.openai_compat import OpenAICompatibleClient
from .models.registry import ModelRegistry
from .prompt import build_conductor_prompt, build_conductor_response_format
from .scheduler import topological_sort


PLAN_FILENAME = "plans.jsonl"
PLAN_DATASETS = ("megascience", "hle", "hle-all", "gpqa", "hle-gpqa")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_path(output_dir: Path, shard_index: int, shard_count: int) -> Path:
    if shard_count == 1:
        return output_dir / PLAN_FILENAME
    return output_dir / f"plans-shard-{shard_index:05d}.jsonl"


def load_plan_records(paths: Path | Iterable[Path]) -> list[dict[str, Any]]:
    """Load the latest record for every dataset ID from resumable JSONL files."""
    if isinstance(paths, Path):
        paths = [paths]
    latest: dict[str, dict[str, Any]] = {}
    order: dict[str, int] = {}
    ordinal = 0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be a JSON object")
                dataset_id = str(record.get("dataset_id") or "")
                if not dataset_id:
                    raise ValueError(f"{path}:{line_number}: record is missing dataset_id")
                latest[dataset_id] = record
                order.setdefault(dataset_id, ordinal)
                ordinal += 1
    return sorted(
        latest.values(),
        key=lambda record: (record.get("dataset_index", order[str(record["dataset_id"])]), str(record["dataset_id"])),
    )


def _write_records_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("plan") and not record.get("error")]
    errors = [record for record in records if record.get("error")]
    model_counts: Counter[str] = Counter()
    step_counts: Counter[int] = Counter()
    layer_counts: Counter[int] = Counter()
    for record in valid:
        plan = record["plan"]
        workflow = plan.get("workflow", [])
        step_counts[len(workflow)] += 1
        for step in workflow:
            model_counts[str(step.get("model_id"))] += 1
        try:
            from .schema import Task

            layer_counts[len(topological_sort(Task.from_dict(plan)))] += 1
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "records": len(records),
        "valid": len(valid),
        "invalid": len(errors),
        "unique_questions": len({record.get("dataset_id") for record in records}),
        "models": dict(model_counts.most_common()),
        "steps_per_workflow": {str(key): value for key, value in sorted(step_counts.items())},
        "layers_per_workflow": {str(key): value for key, value in sorted(layer_counts.items())},
        "errors": dict(Counter(str(record.get("error_type") or "unknown") for record in errors).most_common()),
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan") or {}
    question = str(record.get("question") or "")
    return {
        "dataset_id": record.get("dataset_id"),
        "dataset_index": record.get("dataset_index"),
        "question": question[:239] + "…" if len(question) > 240 else question,
        "valid": bool(plan) and not record.get("error"),
        "error": record.get("error"),
        "task_type": plan.get("task_type"),
        "difficulty": plan.get("difficulty"),
        "steps": len(plan.get("workflow", [])),
        "conductor_model": record.get("conductor_model"),
        "attempt": record.get("attempt"),
    }


async def _generate_one(
    *,
    row: dict[str, Any],
    dataset_index: int,
    registry: ModelRegistry,
    client: Any,
    conductor_model: str,
    max_tokens: int,
    temperature: float,
    attempts: int,
    retry_delay_seconds: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    # grpo imports the optional training stack; keep it off inspection-only CLI paths.
    from .grpo import parse_conductor_json

    question = str(row["question"])
    prompt = build_conductor_prompt(question, registry)
    response_format = build_conductor_response_format(registry)
    raw_completion = ""
    usage = None
    latency_ms = None
    finish_reason = None
    error: Exception | None = None
    used_attempts = 0

    for attempt in range(1, attempts + 1):
        used_attempts = attempt
        try:
            async with semaphore:
                response = await client.generate(
                    instruction=prompt,
                    question=question,
                    context={},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            raw_completion = response.text
            usage = response.usage
            latency_ms = response.latency_ms
            finish_reason = response.finish_reason
            task = parse_conductor_json(raw_completion, question=question, model_registry=registry)
            plan = task.model_dump(mode="json")
            error = None
            break
        except Exception as exc:  # Persist item-level failures without aborting the expensive run.
            error = exc
            plan = None
            if attempt < attempts and retry_delay_seconds:
                await asyncio.sleep(retry_delay_seconds)

    dataset_id = str(row.get("id") or f"row-{dataset_index}")
    return {
        "timestamp": _utc_now(),
        "dataset_id": dataset_id,
        "dataset_index": dataset_index,
        "dataset": row.get("dataset"),
        "question": question,
        "gold_answer": row.get("answer"),
        "reference_answer": row.get("reference_answer"),
        "answer_type": row.get("answer_type"),
        "subject": row.get("subject"),
        "rank": 0,
        "batch": dataset_index,
        "sample": 0,
        "reward": 0.5 if plan is not None else 0.0,
        "plan": plan,
        "worker_outputs": {},
        "workflow_runtime": None,
        "final_answer": None,
        "conductor_completion": raw_completion,
        "conductor_model": conductor_model,
        "conductor_performance": {
            "model_id": conductor_model,
            "usage": usage,
            "batch_latency_ms": latency_ms,
            "batch_size": 1,
            "finish_reason": finish_reason,
        },
        "prompt_hash": _json_hash({"prompt": prompt, "response_format": response_format}),
        "attempt": used_attempts,
        "error_type": type(error).__name__ if error is not None else None,
        "error": f"{type(error).__name__}: {error}" if error is not None else None,
    }


async def generate_plans(
    *,
    rows: Sequence[dict[str, Any]],
    registry: ModelRegistry,
    client: Any,
    conductor_model: str,
    output_path: Path,
    shard_index: int = 0,
    shard_count: int = 1,
    concurrency: int = 16,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    attempts: int = 2,
    retry_delay_seconds: float = 1.0,
    retry_invalid: bool = False,
) -> dict[str, Any]:
    if concurrency <= 0 or attempts <= 0:
        raise ValueError("concurrency and attempts must be positive")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_plan_records(output_path) if output_path.is_file() else []
    completed = {
        str(record["dataset_id"])
        for record in existing
        if not retry_invalid or (record.get("plan") and not record.get("error"))
    }
    selected = [
        (index, dict(row))
        for index, row in enumerate(rows)
        if index % shard_count == shard_index
        and str(row.get("id") or f"row-{index}") not in completed
    ]
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            _generate_one(
                row=row,
                dataset_index=index,
                registry=registry,
                client=client,
                conductor_model=conductor_model,
                max_tokens=max_tokens,
                temperature=temperature,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
                semaphore=semaphore,
            )
        )
        for index, row in selected
    ]
    generated = 0
    failed = 0
    with output_path.open("a", encoding="utf-8") as stream:
        for task in asyncio.as_completed(tasks):
            record = await task
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            generated += 1
            failed += int(bool(record.get("error")))
            if generated % 25 == 0 or generated == len(tasks):
                print(
                    f"planned={generated}/{len(tasks)} failed={failed} resumed={len(completed)}",
                    flush=True,
                )
    records = load_plan_records(output_path)
    # Collapse retried IDs and restore deterministic dataset order after a clean run.
    _write_records_atomic(output_path, records)
    return {**summarize_records(records), "generated": generated, "resumed": len(completed)}


def merge_shards(output_dir: Path, *, expected_shards: int | None = None) -> dict[str, Any]:
    shard_paths = sorted(output_dir.glob("plans-shard-*.jsonl"))
    if not shard_paths:
        raise ValueError(f"No plan shards found in {output_dir}")
    if expected_shards is not None and len(shard_paths) != expected_shards:
        raise ValueError(f"Expected {expected_shards} shards, found {len(shard_paths)}")
    records_by_id: dict[str, dict[str, Any]] = {}
    source_by_id: dict[str, Path] = {}
    for shard_path in shard_paths:
        for record in load_plan_records(shard_path):
            dataset_id = str(record["dataset_id"])
            if dataset_id in source_by_id:
                raise ValueError(
                    f"Dataset ID {dataset_id!r} occurs in both {source_by_id[dataset_id]} and {shard_path}"
                )
            source_by_id[dataset_id] = shard_path
            records_by_id[dataset_id] = record
    records = sorted(
        records_by_id.values(),
        key=lambda record: (record.get("dataset_index", 0), str(record["dataset_id"])),
    )
    target = output_dir / PLAN_FILENAME
    _write_records_atomic(target, records)
    invalid_path = output_dir / "invalid.jsonl"
    with invalid_path.open("w", encoding="utf-8") as stream:
        for record in records:
            if record.get("error"):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_records(records)
    (output_dir / "manifest.json").write_text(
        json.dumps({"merged_at": _utc_now(), "shards": len(shard_paths), **summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"output": str(target), "invalid_output": str(invalid_path), **summary}


def _records_for_cli(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        merged = path / PLAN_FILENAME
        if merged.is_file():
            return load_plan_records(merged)
        shards = sorted(path.glob("plans-shard-*.jsonl"))
        if shards:
            return load_plan_records(shards)
        raise ValueError(f"No plans found in {path}")
    return load_plan_records(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="theo-plan", description="Pregenerate and inspect conductor workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate resumable workflows without executing workers")
    generate.add_argument("--dataset", choices=PLAN_DATASETS, default="hle-all")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--conductor-base-url", default=os.getenv("CONDUCTOR_BASE_URL", "http://127.0.0.1:8007/v1"))
    generate.add_argument("--conductor-api-key", default=os.getenv("CONDUCTOR_API_KEY", "EMPTY"))
    generate.add_argument("--conductor-model")
    generate.add_argument("--conductor-source", help="Base checkpoint or adapter provenance recorded in the manifest")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--max-samples", type=int)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--shard-count", type=int, default=1)
    generate.add_argument("--concurrency", type=int, default=16)
    generate.add_argument("--max-tokens", type=int, default=2048)
    generate.add_argument("--temperature", type=float, default=0.1)
    generate.add_argument("--attempts", type=int, default=2)
    generate.add_argument("--retry-delay-seconds", type=float, default=1.0)
    generate.add_argument("--retry-invalid", action="store_true")

    merge = subparsers.add_parser("merge", help="Merge array-job shards into plans.jsonl")
    merge.add_argument("output_dir", type=Path)
    merge.add_argument("--expected-shards", type=int)

    for name in ("summary", "list", "show"):
        command = subparsers.add_parser(name, help=f"{name.title()} pregenerated workflows")
        command.add_argument("plans", type=Path)
        if name == "list":
            validity = command.add_mutually_exclusive_group()
            validity.add_argument("--invalid-only", action="store_true")
            validity.add_argument("--valid-only", action="store_true")
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--limit", type=int, default=50)
            command.add_argument("--full", action="store_true")
        elif name == "show":
            command.add_argument("--id", required=True, dest="dataset_id")
    return parser


def _write_generation_manifest(args: argparse.Namespace, model: str, path: Path, summary: dict[str, Any]) -> None:
    name = "manifest.json" if args.shard_count == 1 else f"manifest-shard-{args.shard_index:05d}.json"
    payload = {
        "created_at": _utc_now(),
        "dataset": args.dataset,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "conductor_model": model,
        "conductor_source": args.conductor_source or model,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "plans": str(path),
        **summary,
    }
    (args.output_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        from .data import load_conductor_dataset

        registry = ModelRegistry.from_yaml_file(args.config)
        model = args.conductor_model or registry.conductor_model
        if not model:
            raise ValueError("Set --conductor-model or conductor_model in the worker config")
        dataset = load_conductor_dataset(args.dataset, seed=args.seed, max_samples=args.max_samples)
        rows = [dict(dataset[index]) for index in range(len(dataset))]
        client = OpenAICompatibleClient(
            base_url=args.conductor_base_url,
            api_key=args.conductor_api_key,
            model=model,
            max_retries=0,
        )
        path = plan_path(args.output_dir, args.shard_index, args.shard_count)
        summary = asyncio.run(
            generate_plans(
                rows=rows,
                registry=registry,
                client=client,
                conductor_model=model,
                output_path=path,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                attempts=args.attempts,
                retry_delay_seconds=args.retry_delay_seconds,
                retry_invalid=args.retry_invalid,
            )
        )
        _write_generation_manifest(args, model, path, summary)
        if args.shard_count == 1:
            records = load_plan_records(path)
            with (args.output_dir / "invalid.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    if record.get("error"):
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"output": str(path), **summary}
    if args.command == "merge":
        return merge_shards(args.output_dir, expected_shards=args.expected_shards)

    records = _records_for_cli(args.plans)
    if args.command == "summary":
        return summarize_records(records)
    if args.command == "show":
        for record in records:
            if str(record.get("dataset_id")) == args.dataset_id:
                return record
        raise ValueError(f"Plan {args.dataset_id!r} was not found")
    if args.invalid_only:
        records = [record for record in records if record.get("error")]
    elif args.valid_only:
        records = [record for record in records if record.get("plan") and not record.get("error")]
    records = records[args.offset : args.offset + args.limit]
    return records if args.full else [_compact_record(record) for record in records]


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run_cli(argv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
