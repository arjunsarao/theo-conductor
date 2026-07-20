#!/usr/bin/env python3
"""Produce exact conductor-completion token counts for a GRPO JSONL trace.

The trace stores raw text rather than model-token IDs, so this sidecar
re-tokenizes every completion with the conductor tokenizer. It never changes
the input trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_LIMIT = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Create conductor completion token-count sidecar JSON.")
    parser.add_argument("trace", type=Path, help="Path to plans-and-worker-outputs JSONL trace")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Conductor tokenizer (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_LIMIT, help=f"Generation cap for saturation markers (default: {DEFAULT_LIMIT})")
    parser.add_argument("--output", type=Path, help="Sidecar path (default: trace name with -token-counts.json)")
    parser.add_argument("--local-files-only", action="store_true", help="Require an already-cached tokenizer")
    args = parser.parse_args()

    output = args.output or args.trace.with_name(f"{args.trace.stem}-token-counts.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = [len(tokenizer(record.get("conductor_completion") or "", add_special_tokens=False)["input_ids"]) for record in records]
    output.write_text(json.dumps({
        "source_trace": args.trace.name,
        "tokenizer": args.model,
        "add_special_tokens": False,
        "configured_max_completion_tokens": args.max_completion_tokens,
        "counts": counts,
    }, indent=2) + "\n", encoding="utf-8")
    saturated = sum(count >= args.max_completion_tokens for count in counts)
    print(f"Wrote {output}: max={max(counts, default=0)} tokens; {saturated}/{len(counts)} at or above {args.max_completion_tokens}.")


if __name__ == "__main__":
    main()
