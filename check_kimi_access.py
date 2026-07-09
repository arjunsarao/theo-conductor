#!/usr/bin/env python3
"""Check Kimi K2.6 access and optionally ask it one question."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
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


if __name__ == "__main__":
    sys.exit(main())
