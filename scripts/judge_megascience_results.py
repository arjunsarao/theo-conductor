#!/usr/bin/env python3
"""Rescore a MegaScience results JSONL with Kimi K2.6.

The file is updated atomically in small, resumable checkpoints. Existing
successful verdicts from the same judge model are skipped unless --force is
provided. The adjacent summary.json is refreshed after judging completes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theo_conductor.benchmark import (  # noqa: E402
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    _load_completed,
    judge_records_with_checkpoints,
    summarize_records,
)
from theo_conductor.models.openai_compat import OpenAICompatibleClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=ROOT / "outputs/megascience-small-models/results.jsonl",
    )
    parser.add_argument("--summary", type=Path, help="Summary to refresh; defaults beside results.jsonl.")
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_JUDGE_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("KIMI_API_KEY", "change-this"))
    parser.add_argument("--model", default=os.environ.get("KIMI_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true", help="Replace successful existing judge verdicts.")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    results_path = args.results.resolve()
    summary_path = (args.summary or results_path.with_name("summary.json")).resolve()
    records, _ = _load_completed(results_path)
    if not records:
        raise ValueError(f"No benchmark records found in {results_path}")

    client = OpenAICompatibleClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    await judge_records_with_checkpoints(
        records,
        all_records=records,
        results_path=results_path,
        client=client,
        judge_model=args.model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        attempts=args.attempts,
        checkpoint_size=args.checkpoint_size,
        force=args.force,
    )

    summary: dict[str, object] = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            summary = loaded
    seed = int(summary.get("seed", 42))
    summary.update(
        judge_enabled=True,
        judge_model=args.model,
        judge_batch_size=args.batch_size,
        **summarize_records(records, bootstrap_samples=args.bootstrap_samples, seed=seed),
    )
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(summary_path)

    errors = sum(bool(record.get("judge_error")) for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    print(f"Judged {len(records)} records: {correct} correct, {errors} judge errors.")
    print(f"Updated {results_path}")
    print(f"Updated {summary_path}")
    return 1 if errors else 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; completed checkpoints were preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
