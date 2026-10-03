#!/usr/bin/env python3
"""Direct Anthropic batches followed by synchronous OpenRouter Gemini judging."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx

from theo_conductor import diamond_benchmark as benchmark


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def batch_request(row: dict, max_tokens: int) -> dict:
    prompt = benchmark.build_worker_prompt(
        instruction=benchmark.DEFAULT_INSTRUCTION, question=str(row["question"]), context={}
    )
    content = [{"type": "text", "text": prompt}]
    if row.get("is_multimodal"):
        metadata, data = row["image"].split(",", 1)
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": metadata[5:].split(";")[0], "data": data,
        }})
    return {"custom_id": row["id"], "params": {
        "model": "claude-fable-5-1", "max_tokens": max_tokens,
        "system": "You are a worker model in a multi-model reasoning workflow. Follow the instruction exactly.",
        "messages": [{"role": "user", "content": content}],
    }}


async def generate(args: argparse.Namespace) -> None:
    rows = benchmark.load_jsonl(args.input)
    if len(rows) != 61 or sum(bool(r.get("is_multimodal")) for r in rows) != 21:
        raise ValueError("Expected 61 questions including 21 image questions")
    path = args.output_dir / "results.jsonl"
    existing = {r["example_id"]: r for r in benchmark.load_jsonl(path)} if path.exists() else {}
    records = [existing.get(row["id"], benchmark.new_record(
        row, i, solver_model="anthropic/claude-fable-5.1", solver_name="Claude-Fable-5.1"
    )) for i, row in enumerate(rows)]
    by_id = {r["example_id"]: r for r in records}
    benchmark.write_results_atomic(path, records)
    async with httpx.AsyncClient(
        base_url="https://api.anthropic.com", timeout=120,
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
    ) as client:
        for attempt in range(1, args.attempts + 1):
            state_path = args.output_dir / f"anthropic-batch-{attempt}.json"
            pending = [r for r in rows if not by_id[r["id"]].get("response")]
            if not pending:
                break
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if not state.get("id"):
                    raise RuntimeError(f"Ambiguous submission: inspect Anthropic batch list before clearing {state_path}")
            else:
                # A failed POST may still have created a batch. Never blindly resubmit it.
                write_json(state_path, {"submission_pending": True, "custom_ids": [r["id"] for r in pending]})
                response = await client.post("/v1/messages/batches", json={
                    "requests": [batch_request(r, args.max_tokens) for r in pending],
                })
                response.raise_for_status()
                state = response.json()
                write_json(state_path, state)
                print(f"Submitted attempt={attempt} batch={state['id']} requests={len(pending)}", flush=True)
            while state["processing_status"] != "ended":
                await asyncio.sleep(args.poll_seconds)
                try:
                    response = await client.get(f"/v1/messages/batches/{state['id']}")
                    response.raise_for_status()
                    state = response.json()
                    write_json(state_path, state)
                    print(f"batch={state['id']} counts={state['request_counts']}", flush=True)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    print(f"Batch poll failed; retrying: {type(exc).__name__}", flush=True)
            raw_path = args.output_dir / f"anthropic-batch-{attempt}-results.jsonl"
            if not raw_path.exists():
                response = await client.get(f"/v1/messages/batches/{state['id']}/results")
                response.raise_for_status()
                temporary = raw_path.with_suffix(".tmp")
                temporary.write_text(response.text)
                temporary.replace(raw_path)
            for item in benchmark.load_jsonl(raw_path):
                record = by_id[item["custom_id"]]
                result = item["result"]
                if result["type"] != "succeeded":
                    record["error"] = json.dumps(result)
                    continue
                message = result["message"]
                usage = message["usage"]
                text = "\n".join(b["text"] for b in message["content"] if b["type"] == "text")
                benchmark.apply_response(record, SimpleNamespace(
                    text=text, usage={**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]},
                    latency_ms=None,
                    finish_reason="length" if message["stop_reason"] == "max_tokens" else message["stop_reason"],
                ))
                record.update(anthropic_usage=usage, anthropic_batch_id=state["id"],
                              generation_mode="Anthropic Message Batches API",
                              estimated_generation_cost_usd=(usage["input_tokens"] * 5 + usage["output_tokens"] * 25) / 1e6)
            benchmark.write_results_atomic(path, records)
        if any(not r.get("response") or r.get("error") for r in records):
            raise RuntimeError("Generation failures remain; checkpoint saved for recovery")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128000)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(key):
            raise ValueError(f"{key} missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    await generate(args)
    status = await benchmark.run(argparse.Namespace(
        input=args.input, output_dir=args.output_dir, api_key_env="OPENROUTER_API_KEY",
        expect_samples=61, dry_run=False, judge_only=True,
        solver_model="anthropic/claude-fable-5.1", solver_name="Claude-Fable-5.1",
        solver_standard=False, judge_mode="sync", judge_concurrency=8,
        judge_max_tokens=8192, attempts=3, bootstrap_samples=10000,
    ))
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(solver_text_mode="Anthropic Message Batches API",
                   solver_multimodal_mode="Anthropic Message Batches API",
                   generation_billing="Anthropic account", judging_billing="OpenRouter account",
                   estimated_generation_cost_usd=sum(r.get("estimated_generation_cost_usd", 0)
                       for r in benchmark.load_jsonl(args.output_dir / "results.jsonl")))
    write_json(summary_path, summary)
    return status


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
