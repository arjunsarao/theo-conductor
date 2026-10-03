#!/usr/bin/env python3
"""Classify physics questions with Codex."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from string import Template
from typing import Any

from openai import AsyncOpenAI

from theo_conductor.data import load_hle_physics_text_dataset


TAGS = {
    "Condensed Matter Physics",
    "Quantum Information, Science & Technology",
    "Atomic, Molecular & Optical Physics",
    "High Energy & Particle Physics",
    "Mathematical Physics",
    "Gravitation, Cosmology & Astrophysics",
    "Statistical Physics & Thermodynamics",
    "Nuclear Physics",
    "Nonlinear Dynamics & Classical Mechanics",
    "Fluid & Plasma Dynamics",
    "Biophysics",
    "Electromagnetism",
    "Optics & Photonics",
    "Experimental & Applied Physics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Optional local JSONL input; defaults to the canonical HLE Physics set.",
    )
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument("--expect-samples", type=int, default=202)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_classification(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"tag_1", "tag_2", "tag_3"}:
        raise ValueError("response must contain exactly tag_1, tag_2, and tag_3")
    tags = [value[f"tag_{index}"] for index in range(1, 4)]
    if any(tag not in TAGS for tag in tags):
        raise ValueError(f"response contains an unknown tag: {tags}")
    if len(set(tags)) != 3:
        raise ValueError(f"response tags are not distinct: {tags}")
    return {f"tag_{index}": tags[index - 1] for index in range(1, 4)}


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            validate_classification(
                {key: record[key] for key in ("tag_1", "tag_2", "tag_3")}
            )
            completed[str(record["id"])] = record
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result at {path}:{line_number}: {exc}") from exc
    return completed


def parse_response(text: str) -> dict[str, str]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    return validate_classification(json.loads(candidate))


async def classify_once(
    *,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    response_format: dict[str, Any],
    prompt: str,
) -> dict[str, str]:
    completion = await client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        response_format=response_format,
        extra_body={"reasoning": {"effort": args.reasoning_effort}},
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned an empty response")
    return parse_response(content)


async def run(args: argparse.Namespace) -> int:
    if args.concurrency <= 0 or args.attempts <= 0 or args.timeout_seconds <= 0:
        raise ValueError("concurrency, attempts, and timeout-seconds must be positive")
    if not args.prompt.is_file():
        raise FileNotFoundError(args.prompt)
    if not args.schema.is_file():
        raise FileNotFoundError(args.schema)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"{args.api_key_env} is not set")

    schema = json.loads(args.schema.read_text())
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "hle_physics_classification",
            "strict": True,
            "schema": schema,
        },
    }
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout_seconds,
        max_retries=0,
    )

    if args.input_jsonl:
        if not args.input_jsonl.is_file():
            raise FileNotFoundError(args.input_jsonl)
        rows = [
            json.loads(line)
            for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = list(load_hle_physics_text_dataset())
    if len(rows) != args.expect_samples:
        raise ValueError(f"Expected {args.expect_samples} HLE questions, found {len(rows)}")
    prompt_template = Template(args.prompt.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "classifications.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    completed = read_completed(results_path)
    pending = [row for row in rows if row["id"] not in completed]
    print(
        f"dataset={len(rows)} completed={len(completed)} pending={len(pending)} "
        f"model={args.model}",
        flush=True,
    )
    if args.dry_run:
        return 0

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: list[dict[str, Any]] = []

    async def classify(row: dict[str, Any]) -> None:
        prompt = prompt_template.substitute(
            question=row["question"],
            reference_answer=row["reference_answer"],
        )
        error = ""
        async with semaphore:
            for attempt in range(1, args.attempts + 1):
                try:
                    tags = await classify_once(
                        args=args,
                        client=client,
                        response_format=response_format,
                        prompt=prompt,
                    )
                    record = {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "reference_answer": row["reference_answer"],
                        **tags,
                        "model": args.model,
                    }
                    serialized = json.dumps(record, ensure_ascii=False)
                    async with write_lock:
                        with results_path.open("a", encoding="utf-8") as handle:
                            handle.write(serialized + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        completed[row["id"]] = record
                        print(
                            f"completed={len(completed)}/{len(rows)} id={row['id']}",
                            flush=True,
                        )
                    return
                except Exception as exc:  # Keep the batch resumable after individual failures.
                    error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"attempt={attempt}/{args.attempts} id={row['id']} error={error}",
                        flush=True,
                    )
                    if attempt < args.attempts:
                        await asyncio.sleep(min(30, 2**attempt))
        failures.append({"id": row["id"], "error": error})

    await asyncio.gather(*(classify(row) for row in pending))

    ordered = [completed[row["id"]] for row in rows if row["id"] in completed]
    replacement = results_path.with_suffix(".jsonl.tmp")
    with replacement.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(replacement, results_path)

    with failures_path.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "expected": len(rows),
        "completed": len(completed),
        "failed": len(failures),
        "results": str(results_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0 if len(completed) == len(rows) else 1


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
