#!/usr/bin/env python3
"""Check Kimi K2.6 access and optionally ask it one question."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from theo_conductor.benchmark import (  # noqa: E402
    JUDGE_INSTRUCTION,
    build_judge_batch_question,
    parse_judge_batch,
)
from theo_conductor.models.openai_compat import build_message  # noqa: E402


BASE_URL = "http://10.10.0.1:80/v1"
API_KEY = "change-this"
TARGET_MODEL = "moonshotai/Kimi-K2.6"


def post_json(path: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_models() -> list[dict[str, object]]:
    request = urllib.request.Request(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)

    return payload.get("data", [])


def ask_question(question: str, max_tokens: int) -> dict[str, object]:
    return post_json(
        "/chat/completions",
        {
            "model": TARGET_MODEL,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=120,
    )


def build_batch_request(batch_size: int, max_tokens: int) -> tuple[dict[str, object], dict[str, bool]]:
    """Build one training-shaped judge request containing known control answers."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    expected: dict[str, bool] = {}
    records: list[tuple[str, dict[str, object]]] = []
    for index in range(batch_size):
        item_id = f"batch-check-{index}"
        correct = index % 2 == 0
        expected[item_id] = correct
        records.append(
            (
                item_id,
                {
                    "question": "What is 2 + 2?",
                    "reference_answer": "4",
                    "gold_answer": "Adding two and two gives 4.",
                    "response": f"Calculation complete. FINAL: {'4' if correct else '5'}",
                    "extracted_answer": "4" if correct else "5",
                },
            )
        )

    messages = build_message(
        instruction=JUDGE_INSTRUCTION,
        question=build_judge_batch_question(records),
        context={},
    )
    return (
        {
            "model": TARGET_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        expected,
    )


def ask_batch(batch_size: int, max_tokens: int) -> tuple[dict[str, object], dict[str, bool]]:
    payload, expected = build_batch_request(batch_size, max_tokens)
    return post_json("/chat/completions", payload, timeout=300), expected


def response_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Batch response did not contain a completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Batch response choice did not contain a message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Batch response message did not contain text content")
    return content


def validate_batch_response(payload: dict[str, object], expected: dict[str, bool]) -> dict[str, tuple[bool, str]]:
    verdicts = parse_judge_batch(response_text(payload), list(expected))
    mismatches = {
        item_id: {"expected": correct, "actual": verdicts[item_id][0]}
        for item_id, correct in expected.items()
        if verdicts[item_id][0] is not correct
    }
    if mismatches:
        raise ValueError(f"Judge returned incorrect control verdicts: {mismatches}")
    return verdicts


def print_batch_response(payload: dict[str, object], expected: dict[str, bool]) -> int:
    verdicts = validate_batch_response(payload, expected)
    print(f"OK: {TARGET_MODEL} judged {len(verdicts)} items in one request.")
    for item_id, (correct, reason) in verdicts.items():
        print(f"- {item_id}: correct={str(correct).lower()} — {reason}")
    return 0


def check_model_access() -> int:
    models = get_models()
    model_ids = {str(model.get("id")) for model in models}
    roots = {str(model.get("root")) for model in models}

    if TARGET_MODEL in model_ids or TARGET_MODEL in roots:
        print(f"OK: {TARGET_MODEL} is available.")
        return 0

    print(f"Missing: {TARGET_MODEL} was not found.")
    print("Available models:")
    for model_id in sorted(model_ids):
        print(f"- {model_id}")
    return 2


def print_chat_response(payload: dict[str, object]) -> int:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        print("No choices returned.")
        print(json.dumps(payload, indent=2))
        return 1

    choice = choices[0]
    if not isinstance(choice, dict):
        print("Unexpected choice format.")
        print(json.dumps(payload, indent=2))
        return 1

    message = choice.get("message")
    if not isinstance(message, dict):
        print("No message returned.")
        print(json.dumps(payload, indent=2))
        return 1

    reasoning = message.get("reasoning")
    answer = message.get("content")

    print("Reasoning:")
    print(reasoning or "(none)")
    print()
    print("Answer:")
    print(answer or "(none)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Kimi K2.6 access or ask it one question."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. Omit this to only check model availability.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Completion token budget for reasoning plus answer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Send one training-shaped judge request containing this many control answers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.question and args.batch_size is not None:
            raise ValueError("Provide either a question or --batch-size, not both")
        if args.batch_size is not None:
            response, expected = ask_batch(args.batch_size, args.max_tokens)
            return print_batch_response(response, expected)
        if args.question:
            response = ask_question(args.question, args.max_tokens)
            return print_chat_response(response)

        return check_model_access()
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}")
        try:
            print(exc.read().decode())
        except UnicodeDecodeError:
            pass
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc.reason}")
        return 1
    except TimeoutError:
        print("Connection timed out.")
        return 1
    except json.JSONDecodeError as exc:
        print(f"Could not parse JSON response: {exc}")
        return 1
    except ValueError as exc:
        print(f"Invalid response: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
