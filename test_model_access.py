#!/usr/bin/env python3
"""Verify that every worker in a model-pool config can generate a response."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "worker_pool_large.yaml"
DEFAULT_QUESTION = "Reply with exactly: MODEL_ACCESS_OK"


@dataclass(frozen=True)
class Worker:
    worker_id: str
    display_name: str
    base_url: str
    model: str
    api_key: str


def load_workers(config_path: str | Path) -> list[Worker]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path} must contain a non-empty 'models' list")

    workers: list[Worker] = []
    for index, entry in enumerate(models):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: model {index} must be a mapping")
        client = entry.get("client")
        if not isinstance(client, dict):
            raise ValueError(f"{path}: model {index} is missing client configuration")

        worker_id = str(
            entry.get("model_idx")
            or entry.get("name")
            or entry.get("display_name")
            or f"model-{index}"
        )
        display_name = str(entry.get("display_name") or worker_id)
        base_url = str(client.get("base_url") or "").rstrip("/")
        model = str(client.get("model") or "")
        api_key = str(client.get("api_key", "EMPTY"))
        if not base_url or not model:
            raise ValueError(f"{path}: worker {worker_id!r} needs client.base_url and client.model")

        workers.append(Worker(worker_id, display_name, base_url, model, api_key))

    return workers


def request_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
        method = "POST"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return result


def advertised_models(worker: Worker, timeout: float) -> set[str]:
    payload = request_json(
        f"{worker.base_url}/models",
        api_key=worker.api_key,
        timeout=timeout,
    )
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("/models response is missing its data list")
    return {
        str(item["id"])
        for item in data
        if isinstance(item, dict) and item.get("id") is not None
    }


def generate(worker: Worker, question: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    return request_json(
        f"{worker.base_url}/chat/completions",
        api_key=worker.api_key,
        payload={
            "model": worker.model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )


def completion_parts(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("completion response did not contain a choice")

    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("completion choice did not contain a message")

    content = message.get("content")
    answer = content if isinstance(content, str) else ""
    reasoning_value = message.get("reasoning_content", message.get("reasoning", ""))
    reasoning = reasoning_value if isinstance(reasoning_value, str) else ""
    finish_reason = choice.get("finish_reason")
    return reasoning, answer, str(finish_reason) if finish_reason is not None else None


def check_worker(worker: Worker, question: str, max_tokens: int, timeout: float) -> None:
    print(f"=== {worker.worker_id} ({worker.display_name}) ===", flush=True)
    print(f"Endpoint: {worker.base_url}", flush=True)
    print(f"Served model: {worker.model}", flush=True)

    available = advertised_models(worker, min(timeout, 30))
    if worker.model not in available:
        raise ValueError(
            f"configured model {worker.model!r} is not advertised; available: {sorted(available)!r}"
        )
    print("Availability: OK", flush=True)

    payload = generate(worker, question, max_tokens, timeout)
    reasoning, answer, finish_reason = completion_parts(payload)
    print(f"Finish reason: {finish_reason or '(none)'}")
    if reasoning.strip():
        print("Reasoning:")
        print(reasoning)
    print("Response:")
    print(answer or "(empty)")

    if not answer.strip():
        raise ValueError(
            f"empty completion content (reasoning length={len(reasoning)}, "
            f"finish_reason={finish_reason!r})"
        )
    print("Generation: OK", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check model advertisement and generation for every worker in a YAML pool."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"Prompt sent to every model (default: {DEFAULT_QUESTION!r}).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Worker-pool YAML file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Token budget for reasoning plus answer (default: 1024).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Per-generation timeout in seconds (default: 300).",
    )
    return parser.parse_args()


def describe_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode().strip()
        except (OSError, UnicodeDecodeError):
            pass
        return f"HTTP {exc.code} {exc.reason}" + (f": {detail}" if detail else "")
    if isinstance(exc, urllib.error.URLError):
        return f"connection error: {exc.reason}"
    return str(exc)


def main() -> int:
    args = parse_args()
    if args.max_tokens <= 0:
        print("Error: --max-tokens must be positive", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("Error: --timeout must be positive", file=sys.stderr)
        return 2

    try:
        workers = load_workers(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Could not load model config: {exc}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for worker in workers:
        try:
            check_worker(worker, args.question, args.max_tokens, args.timeout)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            error = describe_error(exc)
            failures.append((worker.worker_id, error))
            print(f"FAILED: {error}", file=sys.stderr, flush=True)
        print()

    if failures:
        print(f"{len(failures)}/{len(workers)} workers failed:", file=sys.stderr)
        for worker_id, error in failures:
            print(f"- {worker_id}: {error}", file=sys.stderr)
        return 1

    print(f"All {len(workers)} workers returned non-empty responses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
