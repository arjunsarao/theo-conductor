"""Queryable, model-friendly error analysis for conductor JSONL traces.

The public :class:`TraceDataset` API is dependency-free.  The module also
provides a JSON-first CLI::

    python -m theo_conductor.trace_analysis summary trace.jsonl
    python -m theo_conductor.trace_analysis errors trace.jsonl
    python -m theo_conductor.trace_analysis list trace.jsonl --reward 0,0.2
    python -m theo_conductor.trace_analysis show trace.jsonl 0:17
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def error_category(
    error: Any,
    *,
    completion: Any = None,
    completion_saturated: bool = False,
) -> str:
    """Normalize verbose parser/validation errors into stable categories."""
    if not error:
        return "No execution/validation error"
    value = str(error).strip()
    rules = (
        ("Completion does not contain valid JSON", "Malformed: no valid JSON"),
        ("Final step is missing required access keys", "Invalid: final step missing required inputs"),
        ("Final workflow step must have step_id 'final'", "Invalid: final step is not named final"),
        ("not found. Valid models", "Invalid: unknown model ID"),
        ("accesses unknown or future key", "Invalid: invalid dependency/access key"),
        ("validation error for Step", "Invalid: step schema validation"),
    )
    for needle, category in rules:
        if needle in value:
            if needle == "Completion does not contain valid JSON":
                return _malformed_json_category(
                    completion,
                    completion_saturated=completion_saturated,
                )
            return category
    return value.splitlines()[0][:110]


def _malformed_json_category(completion: Any, *, completion_saturated: bool) -> str:
    if completion_saturated:
        return "Malformed: truncated at output token limit"

    text = "" if completion is None else str(completion).strip()
    if not text:
        # Keep the historical category when callers only have an error string.
        return "Malformed: no valid JSON" if completion is None else "Malformed: empty completion"

    if _contains_workflow_payload(text):
        return "Malformed: valid workflow JSON embedded in extra text"
    if "{" not in text and "[" not in text:
        return "Malformed: prose only"
    if re.search(r"[{,]\s*[A-Za-z_][\w.-]*\s*:", text):
        return "Malformed: JSON-like syntax with unquoted keys"
    if _has_unclosed_json_delimiter(text):
        return "Malformed: incomplete or unclosed JSON"
    if _contains_json_fragment(text):
        return "Malformed: JSON fragments without a workflow"
    return "Malformed: invalid JSON syntax"


def _decoded_json_values(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        yield payload


def _contains_workflow_payload(text: str) -> bool:
    return any(
        (isinstance(payload, dict) and "workflow" in payload)
        or (
            isinstance(payload, list)
            and bool(payload)
            and all(isinstance(step, dict) and "step_id" in step for step in payload)
        )
        for payload in _decoded_json_values(text)
    )


def _contains_json_fragment(text: str) -> bool:
    return next(iter(_decoded_json_values(text)), None) is not None


def _has_unclosed_json_delimiter(text: str) -> bool:
    """Check delimiter balance while ignoring braces inside JSON strings."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
    return bool(stack) or in_string


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _usage_number(usage: Any, *keys: str) -> float | None:
    if not isinstance(usage, dict):
        return None
    for key in keys:
        value = _number(usage.get(key))
        if value is not None:
            return value
    return None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return a linearly interpolated percentile without a numeric dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _worker_performance(records: Sequence[TraceRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        outputs = record.data.get("worker_outputs")
        if not isinstance(outputs, dict):
            continue
        for output in outputs.values():
            if isinstance(output, dict):
                model_id = output.get("model_id")
                grouped[str(model_id if model_id is not None else "(missing)")].append(output)

    rows: list[dict[str, Any]] = []
    for model_id, outputs in grouped.items():
        latencies = [
            value
            for output in outputs
            if (value := _number(output.get("latency_ms"))) is not None
        ]
        prompt_tokens = [
            value
            for output in outputs
            if (value := _usage_number(output.get("usage"), "prompt_tokens", "input_tokens")) is not None
        ]
        completion_tokens = [
            value
            for output in outputs
            if (value := _usage_number(output.get("usage"), "completion_tokens", "output_tokens")) is not None
        ]
        total_tokens = [
            value
            for output in outputs
            if (value := _usage_number(output.get("usage"), "total_tokens")) is not None
        ]
        throughput_samples = [
            (completion / latency) * 1000
            for output in outputs
            if (latency := _number(output.get("latency_ms"))) is not None
            and latency > 0
            and (
                completion := _usage_number(
                    output.get("usage"), "completion_tokens", "output_tokens"
                )
            )
            is not None
        ]
        rows.append(
            {
                "model_id": model_id,
                "runs": len(outputs),
                "latency_samples": len(latencies),
                "mean_latency_ms": _mean(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "mean_prompt_tokens": _mean(prompt_tokens),
                "mean_completion_tokens": _mean(completion_tokens),
                "mean_total_tokens": _mean(total_tokens),
                "mean_output_tokens_per_second": _mean(throughput_samples),
            }
        )
    return sorted(rows, key=lambda row: (-row["runs"], row["model_id"]))


def _conductor_performance(records: Sequence[TraceRecord]) -> list[dict[str, Any]]:
    """Aggregate conductor generation data, including legacy token sidecars."""
    grouped: dict[str, list[tuple[TraceRecord, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        performance = record.data.get("conductor_performance")
        performance = performance if isinstance(performance, dict) else {}
        usage = performance.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        if record.completion_tokens is not None and not any(
            key in usage for key in ("completion_tokens", "output_tokens")
        ):
            usage = {**usage, "completion_tokens": record.completion_tokens}
        if not usage and _number(performance.get("batch_latency_ms")) is None:
            continue
        model_id = str(performance.get("model_id") or "Conductor")
        grouped[model_id].append((record, {**performance, "usage": usage}))

    rows: list[dict[str, Any]] = []
    for model_id, samples in grouped.items():
        prompt_tokens = [
            value
            for _, sample in samples
            if (value := _usage_number(sample["usage"], "prompt_tokens", "input_tokens")) is not None
        ]
        completion_tokens = [
            value
            for _, sample in samples
            if (
                value := _usage_number(
                    sample["usage"], "completion_tokens", "output_tokens"
                )
            )
            is not None
        ]
        total_tokens = [
            value
            for _, sample in samples
            if (value := _usage_number(sample["usage"], "total_tokens")) is not None
        ]

        # Batch generation latency is repeated on every rollout in that batch.
        # Count it once, and divide the batch's total output tokens by it.
        batches: dict[tuple[str, Any, Any], dict[str, Any]] = {}
        for record, sample in samples:
            latency = _number(sample.get("batch_latency_ms"))
            if latency is None or latency <= 0:
                continue
            key = (record.source, record.data.get("rank"), record.data.get("batch"))
            batch = batches.setdefault(key, {"latency_ms": latency, "completion_tokens": 0.0})
            completion = _usage_number(
                sample["usage"], "completion_tokens", "output_tokens"
            )
            if completion is not None:
                batch["completion_tokens"] += completion
        latencies = [float(batch["latency_ms"]) for batch in batches.values()]
        throughput_samples = [
            float(batch["completion_tokens"]) * 1000 / float(batch["latency_ms"])
            for batch in batches.values()
            if float(batch["completion_tokens"]) > 0
        ]
        rows.append(
            {
                "model_id": model_id,
                "runs": len(samples),
                "latency_samples": len(latencies),
                "mean_latency_ms": _mean(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "mean_prompt_tokens": _mean(prompt_tokens),
                "mean_completion_tokens": _mean(completion_tokens),
                "mean_total_tokens": _mean(total_tokens),
                "mean_output_tokens_per_second": _mean(throughput_samples),
                "role": "conductor",
                "latency_basis": "generation batch",
            }
        )
    return sorted(rows, key=lambda row: (-row["runs"], row["model_id"]))


def _judge_performance(records: Sequence[TraceRecord]) -> list[dict[str, Any]]:
    """Aggregate successful Kimi judge requests when response metadata exists."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        performance = record.data.get("judge_performance")
        if not isinstance(performance, dict):
            continue
        usage = performance.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        latency = _number(performance.get("latency_ms"))
        if not usage and latency is None:
            continue
        model_id = str(
            performance.get("model_id")
            or record.data.get("judge_model")
            or "Kimi K2.6 judge"
        )
        grouped[model_id].append({**performance, "usage": usage})

    rows: list[dict[str, Any]] = []
    for model_id, samples in grouped.items():
        latencies = [
            value
            for sample in samples
            if (value := _number(sample.get("latency_ms"))) is not None
        ]
        prompt_tokens = [
            value
            for sample in samples
            if (value := _usage_number(sample["usage"], "prompt_tokens", "input_tokens")) is not None
        ]
        completion_tokens = [
            value
            for sample in samples
            if (
                value := _usage_number(
                    sample["usage"], "completion_tokens", "output_tokens"
                )
            )
            is not None
        ]
        total_tokens = [
            value
            for sample in samples
            if (value := _usage_number(sample["usage"], "total_tokens")) is not None
        ]
        throughput_samples = [
            completion * 1000 / latency
            for sample in samples
            if (latency := _number(sample.get("latency_ms"))) is not None
            and latency > 0
            and (
                completion := _usage_number(
                    sample["usage"], "completion_tokens", "output_tokens"
                )
            )
            is not None
        ]
        rows.append(
            {
                "model_id": model_id,
                "runs": len(samples),
                "latency_samples": len(latencies),
                "mean_latency_ms": _mean(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "mean_prompt_tokens": _mean(prompt_tokens),
                "mean_completion_tokens": _mean(completion_tokens),
                "mean_total_tokens": _mean(total_tokens),
                "mean_output_tokens_per_second": _mean(throughput_samples),
                "role": "judge",
                "latency_basis": "request",
            }
        )
    return sorted(rows, key=lambda row: (-row["runs"], row["model_id"]))


def _workflow_steps(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or not isinstance(plan.get("workflow"), list):
        return []
    return [step for step in plan["workflow"] if isinstance(step, dict)]


def _workflow_graph(
    plan: Any,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], list[list[str]]]:
    steps = _workflow_steps(plan)
    step_ids = {
        str(step["step_id"])
        for step in steps
        if step.get("step_id") is not None
    }
    dependencies = {
        str(step.get("step_id")): {
            str(access)
            for access in (step.get("access_list") or [])
            if str(access) in step_ids
        }
        for step in steps
        if step.get("step_id") is not None
    }
    remaining = dict(dependencies)
    completed: set[str] = set()
    layers: list[list[str]] = []
    while remaining:
        layer = [
            step_id
            for step_id, deps in remaining.items()
            if deps <= completed
        ]
        if not layer:
            # Invalid/cyclic plans still get inspectable structural statistics.
            layers.append(list(remaining))
            break
        layers.append(layer)
        completed.update(layer)
        for step_id in layer:
            remaining.pop(step_id)
    return steps, dependencies, layers


def workflow_to_graphviz(plan: Any) -> str:
    """Return a safe DOT representation of a conductor workflow."""
    steps, dependencies, _ = _workflow_graph(plan)

    def quoted(value: Any) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

    lines = [
        "digraph conductor_workflow {",
        '  rankdir="LR";',
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.55"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#edf4f7", color="#6d8796"];',
        '  edge [color="#718692"];',
        '  "__question__" [label="question", shape="oval", fillcolor="#fff7df"];',
    ]
    for step in steps:
        step_id = str(step.get("step_id"))
        label = f"{step_id}\nmodel: {step.get('model_id', '')}"
        lines.append(f"  {quoted(step_id)} [label={quoted(label)}];")
        accesses = [str(value) for value in (step.get("access_list") or [])]
        if "question" in accesses:
            lines.append(f'  "__question__" -> {quoted(step_id)};')
    for step_id, deps in dependencies.items():
        for dependency in sorted(deps):
            lines.append(f"  {quoted(dependency)} -> {quoted(step_id)};")
    lines.append("}")
    return "\n".join(lines)


PLAN_METRIC_NAMES = (
    "num_steps",
    "num_edges",
    "critical_path_steps",
    "critical_path_runtime",
    "total_worker_runtime",
    "maximum_width",
    "work_span_parallelism",
    "observed_wall_time",
    "observed_peak_concurrency",
    "realized_parallelism",
    "parallelism_utilization",
)


def _plan_metrics(record: TraceRecord) -> dict[str, Any] | None:
    steps, dependencies, layers = _workflow_graph(record.data.get("plan"))
    if not steps:
        return None
    step_ids = [str(step.get("step_id")) for step in steps]
    longest_steps: dict[str, int] = {}
    for layer in layers:
        for step_id in layer:
            longest_steps[step_id] = 1 + max(
                (longest_steps.get(parent, 0) for parent in dependencies.get(step_id, set())),
                default=0,
            )

    outputs = record.data.get("worker_outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    latencies_ms = {
        step_id: latency
        for step_id in step_ids
        if isinstance(outputs.get(step_id), dict)
        and (latency := _number(outputs[step_id].get("latency_ms"))) is not None
        and latency >= 0
    }
    has_complete_runtime = len(latencies_ms) == len(step_ids)
    critical_runtime_ms: float | None = None
    total_runtime_ms: float | None = None
    if has_complete_runtime:
        longest_runtime: dict[str, float] = {}
        for layer in layers:
            for step_id in layer:
                longest_runtime[step_id] = latencies_ms[step_id] + max(
                    (
                        longest_runtime.get(parent, 0.0)
                        for parent in dependencies.get(step_id, set())
                    ),
                    default=0.0,
                )
        critical_runtime_ms = max(longest_runtime.values(), default=0.0)
        total_runtime_ms = sum(latencies_ms.values())

    runtime = record.data.get("workflow_runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    observed_wall_ms = _number(runtime.get("observed_wall_time_ms"))
    observed_peak = _number(runtime.get("observed_peak_concurrency"))
    work_span = (
        total_runtime_ms / critical_runtime_ms
        if total_runtime_ms is not None and critical_runtime_ms
        else None
    )
    realized = (
        total_runtime_ms / observed_wall_ms
        if total_runtime_ms is not None and observed_wall_ms and observed_wall_ms > 0
        else None
    )
    maximum_width = max((len(layer) for layer in layers), default=0)
    return {
        "record_id": record.record_id,
        "num_steps": len(step_ids),
        "num_edges": sum(len(parents) for parents in dependencies.values()),
        "critical_path_steps": max(longest_steps.values(), default=0),
        "critical_path_runtime": critical_runtime_ms / 1000 if critical_runtime_ms is not None else None,
        "total_worker_runtime": total_runtime_ms / 1000 if total_runtime_ms is not None else None,
        "maximum_width": maximum_width,
        "work_span_parallelism": work_span,
        "observed_wall_time": observed_wall_ms / 1000 if observed_wall_ms is not None else None,
        "observed_peak_concurrency": int(observed_peak) if observed_peak is not None else None,
        "realized_parallelism": realized,
        "parallelism_utilization": (
            realized / maximum_width if realized is not None and maximum_width else None
        ),
    }


def _plan_statistics(records: Sequence[TraceRecord]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [metrics for record in records if (metrics := _plan_metrics(record)) is not None]
    aggregate = {
        name: _mean(
            [float(row[name]) for row in rows if _number(row.get(name)) is not None]
        )
        for name in PLAN_METRIC_NAMES
    }
    return rows, aggregate


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _worker_timing(records: Sequence[TraceRecord]) -> dict[str, Any] | None:
    """Estimate workflow wall time and attribute each layer to its slowest call."""
    reserved = {"question", "system_prompt", "tool_docs"}
    model_rows: dict[str, dict[str, float | int | str]] = {}
    batch_worker_ms: dict[Any, float] = defaultdict(float)
    batch_timestamps: dict[Any, list[datetime]] = defaultdict(list)
    total_call_ms = 0.0
    critical_path_ms = 0.0
    calls = 0

    for record in records:
        data = record.data
        batch = data.get("batch")
        if (timestamp := _parse_timestamp(data.get("timestamp"))) is not None:
            batch_timestamps[batch].append(timestamp)
        outputs = data.get("worker_outputs")
        workflow = (data.get("plan") or {}).get("workflow")
        if not isinstance(outputs, dict) or not isinstance(workflow, list):
            continue

        remaining = {
            str(step.get("step_id")): step
            for step in workflow
            if isinstance(step, dict) and step.get("step_id") is not None
        }
        completed: set[str] = set()
        while remaining:
            layer = [
                step
                for step in remaining.values()
                if all(
                    access in completed or access in reserved
                    for access in (step.get("access_list") or [])
                )
            ]
            if not layer:
                break
            layer_calls: list[tuple[float, str]] = []
            for step in layer:
                output = outputs.get(str(step.get("step_id")))
                if not isinstance(output, dict):
                    continue
                latency = _number(output.get("latency_ms"))
                if latency is None or latency < 0:
                    continue
                raw_model_id = output.get("model_id")
                model_id = str(raw_model_id if raw_model_id is not None else "(missing)")
                row = model_rows.setdefault(
                    model_id,
                    {
                        "model_id": model_id,
                        "calls": 0,
                        "total_call_latency_ms": 0.0,
                        "critical_path_latency_ms": 0.0,
                        "bottleneck_layers": 0,
                    },
                )
                row["calls"] = int(row["calls"]) + 1
                row["total_call_latency_ms"] = float(row["total_call_latency_ms"]) + latency
                total_call_ms += latency
                calls += 1
                layer_calls.append((latency, model_id))

            if layer_calls:
                layer_latency, bottleneck_model = max(layer_calls)
                critical_path_ms += layer_latency
                batch_worker_ms[batch] += layer_latency
                row = model_rows[bottleneck_model]
                row["critical_path_latency_ms"] = float(row["critical_path_latency_ms"]) + layer_latency
                row["bottleneck_layers"] = int(row["bottleneck_layers"]) + 1
            for step in layer:
                step_id = str(step.get("step_id"))
                completed.add(step_id)
                remaining.pop(step_id, None)

    if not calls:
        return None

    observed_interval_ms = 0.0
    matched_worker_ms = 0.0
    observed_batches = 0
    ordered_batches = sorted(
        (
            (min(timestamps), max(timestamps), batch)
            for batch, timestamps in batch_timestamps.items()
            if timestamps
        ),
        key=lambda item: item[0],
    )
    for previous, current in zip(ordered_batches, ordered_batches[1:]):
        interval_ms = (current[1] - previous[1]).total_seconds() * 1000
        worker_ms = batch_worker_ms.get(current[2], 0.0)
        if interval_ms > 0 and worker_ms > 0:
            observed_interval_ms += interval_ms
            matched_worker_ms += worker_ms
            observed_batches += 1

    rows = []
    for row in model_rows.values():
        call_latency = float(row["total_call_latency_ms"])
        critical_latency = float(row["critical_path_latency_ms"])
        rows.append(
            {
                **row,
                "call_latency_share": call_latency / total_call_ms if total_call_ms else 0,
                "critical_path_share": critical_latency / critical_path_ms if critical_path_ms else 0,
            }
        )
    rows.sort(key=lambda row: (-float(row["critical_path_latency_ms"]), str(row["model_id"])))
    return {
        "calls": calls,
        "total_call_latency_ms": total_call_ms,
        "estimated_workflow_latency_ms": critical_path_ms,
        "parallelism_savings_ms": total_call_ms - critical_path_ms,
        "observed_batch_intervals": observed_batches,
        "observed_batch_interval_ms": observed_interval_ms or None,
        "matched_workflow_latency_ms": matched_worker_ms or None,
        "workflow_interval_share": (
            matched_worker_ms / observed_interval_ms if observed_interval_ms else None
        ),
        "models": rows,
    }


@dataclass(frozen=True)
class TraceRecord:
    """A trace record plus provenance that is not present in the JSONL."""

    data: dict[str, Any]
    record_id: str
    source: str
    line: int
    error_category: str
    completion_tokens: int | None = None
    completion_saturated: bool | None = None

    def full(self) -> dict[str, Any]:
        return {
            **self.data,
            "record_id": self.record_id,
            "source": self.source,
            "line": self.line,
            "error_category": self.error_category,
            "completion_tokens": self.completion_tokens,
            "completion_saturated": self.completion_saturated,
        }

    def compact(self, *, question_chars: int = 240) -> dict[str, Any]:
        question = str(self.data.get("question") or "")
        if len(question) > question_chars:
            question = question[: question_chars - 1] + "…"
        plan = self.data.get("plan") or {}
        workflow = plan.get("workflow") if isinstance(plan, dict) else []
        outputs = self.data.get("worker_outputs") or {}
        return {
            "record_id": self.record_id,
            "rank": self.data.get("rank"),
            "batch": self.data.get("batch"),
            "sample": self.data.get("sample"),
            "reward": self.data.get("reward"),
            "question": question,
            "error_category": self.error_category,
            "has_error": bool(self.data.get("error")),
            "has_plan": bool(self.data.get("plan")),
            "plan_steps": len(workflow) if isinstance(workflow, list) else 0,
            "worker_outputs": len(outputs) if isinstance(outputs, dict) else 0,
            "task_type": plan.get("task_type") if isinstance(plan, dict) else None,
            "difficulty": plan.get("difficulty") if isinstance(plan, dict) else None,
            "completion_tokens": self.completion_tokens,
            "completion_saturated": self.completion_saturated,
        }


@dataclass
class TraceQuery:
    """Composable filters shared by the Python API and CLI."""

    rewards: set[float] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    batches: set[int] = field(default_factory=set)
    ranks: set[int] = field(default_factory=set)
    search: str | None = None
    question: str | None = None
    has_plan: bool | None = None
    has_error: bool | None = None

    def matches(self, record: TraceRecord) -> bool:
        data = record.data
        reward = _number(data.get("reward"))
        if self.rewards and reward not in self.rewards:
            return False
        if self.categories and record.error_category not in self.categories:
            return False
        if self.batches and data.get("batch") not in self.batches:
            return False
        if self.ranks and data.get("rank") not in self.ranks:
            return False
        if self.has_plan is not None and bool(data.get("plan")) != self.has_plan:
            return False
        if self.has_error is not None and bool(data.get("error")) != self.has_error:
            return False
        if self.question and self.question.casefold() not in str(data.get("question") or "").casefold():
            return False
        if self.search:
            fields = (
                data.get("question"), data.get("error"), data.get("conductor_completion"),
                data.get("final_answer"), data.get("gold_answer"), data.get("plan"),
                data.get("worker_outputs"),
            )
            haystack = " ".join(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v or "") for v in fields)
            if self.search.casefold() not in haystack.casefold():
                return False
        return True


class TraceDataset:
    """An in-memory collection of trace records with analysis operations."""

    def __init__(self, records: Sequence[TraceRecord], *, malformed_lines: Sequence[dict[str, Any]] = ()) -> None:
        self.records = list(records)
        self.malformed_lines = list(malformed_lines)

    @classmethod
    def load(cls, paths: str | Path | Iterable[str | Path], *, strict: bool = False) -> "TraceDataset":
        if isinstance(paths, (str, Path)):
            paths = [paths]
        records: list[TraceRecord] = []
        malformed: list[dict[str, Any]] = []
        for source_index, raw_path in enumerate(paths):
            path = Path(raw_path)
            token_counts, token_limit = _load_token_sidecar(path)
            source_record_index = 0
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise ValueError("record is not a JSON object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        issue = {"source": str(path), "line": line_number, "error": str(exc)}
                        if strict:
                            raise ValueError(f"{path}:{line_number}: {exc}") from exc
                        malformed.append(issue)
                        continue
                    tokens = token_counts.get(source_record_index)
                    source_record_index += 1
                    saturated = tokens >= token_limit if tokens is not None and token_limit is not None else None
                    records.append(TraceRecord(
                        data=data,
                        record_id=f"{source_index}:{line_number}",
                        source=str(path),
                        line=line_number,
                        error_category=error_category(
                            data.get("error"),
                            completion=data.get("conductor_completion"),
                            completion_saturated=saturated is True,
                        ),
                        completion_tokens=tokens,
                        completion_saturated=saturated,
                    ))
        return cls(records, malformed_lines=malformed)

    def query(self, query: TraceQuery | None = None) -> list[TraceRecord]:
        query = query or TraceQuery()
        return [record for record in self.records if query.matches(record)]

    def get(self, record_id: str) -> TraceRecord:
        for record in self.records:
            if record.record_id == record_id:
                return record
        raise KeyError(f"record {record_id!r} was not found")

    def summary(self, query: TraceQuery | None = None) -> dict[str, Any]:
        records = self.query(query)
        plan_metrics, plan_statistics = _plan_statistics(records)
        rewards = [_number(r.data.get("reward")) for r in records]
        numeric_rewards = [r for r in rewards if r is not None]
        reward_counts = Counter(numeric_rewards)
        batch_counts: dict[Any, Counter] = defaultdict(Counter)
        for record in records:
            batch_counts[record.data.get("batch")][_number(record.data.get("reward"))] += 1
        token_values = [r.completion_tokens for r in records if r.completion_tokens is not None]
        return {
            "records": len(records),
            "mean_reward": sum(numeric_rewards) / len(numeric_rewards) if numeric_rewards else None,
            "reward_distribution": [
                {"reward": _display_number(reward), "count": count, "fraction": count / len(records) if records else 0}
                for reward, count in sorted(reward_counts.items())
            ],
            "parsed_plans": sum(bool(r.data.get("plan")) for r in records),
            "worker_runs": sum(bool(r.data.get("worker_outputs")) for r in records),
            "unique_questions": len({r.data.get("question") for r in records}),
            "error_categories": _category_rows(records),
            "batches": [
                {"batch": batch, "count": sum(counts.values()), "mean_reward": sum((reward or 0) * count for reward, count in counts.items()) / sum(counts.values()),
                 "rewards": {str(_display_number(reward)): count for reward, count in sorted(counts.items())}}
                for batch, counts in sorted(batch_counts.items(), key=lambda item: (item[0] is None, item[0]))
            ],
            "completion_tokens": ({"available": len(token_values), "min": min(token_values), "max": max(token_values),
                                   "mean": sum(token_values) / len(token_values),
                                   "saturated": sum(r.completion_saturated is True for r in records)} if token_values else None),
            "conductor_performance": _conductor_performance(records),
            "judge_performance": _judge_performance(records),
            "worker_performance": _worker_performance(records),
            "worker_timing": _worker_timing(records),
            "plan_metrics": plan_metrics,
            "plan_statistics": plan_statistics,
            "malformed_jsonl_lines": self.malformed_lines,
        }

    def errors(self, query: TraceQuery | None = None, *, examples: int = 3) -> list[dict[str, Any]]:
        records = [r for r in self.query(query) if r.data.get("error")]
        grouped: dict[str, list[TraceRecord]] = defaultdict(list)
        for record in records:
            grouped[record.error_category].append(record)
        rows = []
        for category, members in grouped.items():
            counts = Counter(_number(r.data.get("reward")) for r in members)
            rows.append({
                "category": category,
                "count": len(members),
                "reward_distribution": {str(_display_number(k)): v for k, v in sorted(counts.items()) if k is not None},
                "examples": [{"record_id": r.record_id, "question": r.data.get("question"), "error": r.data.get("error")} for r in members[:examples]],
            })
        return sorted(rows, key=lambda row: (-row["count"], row["category"]))

    def questions(
        self,
        query: TraceQuery | None = None,
        *,
        min_rollouts: int = 1,
        disagreement_only: bool = False,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[TraceRecord]] = defaultdict(list)
        for record in self.query(query):
            grouped[str(record.data.get("question") or "")].append(record)
        rows = []
        for question, members in grouped.items():
            if len(members) < min_rollouts:
                continue
            rewards = [_number(r.data.get("reward")) for r in members]
            numeric = [r for r in rewards if r is not None]
            counts = Counter(numeric)
            if disagreement_only and len(counts) < 2:
                continue
            rows.append({
                "question": question,
                "rollouts": len(members),
                "mean_reward": sum(numeric) / len(numeric) if numeric else None,
                "min_reward": min(numeric) if numeric else None,
                "max_reward": max(numeric) if numeric else None,
                "reward_distribution": {str(_display_number(k)): v for k, v in sorted(counts.items())},
                "record_ids": [r.record_id for r in members],
                "error_categories": dict(Counter(r.error_category for r in members if r.data.get("error"))),
            })
        return sorted(rows, key=lambda row: (row["mean_reward"] is None, row["mean_reward"], -row["rollouts"], row["question"]))


def _load_token_sidecar(path: Path) -> tuple[dict[int, int], int | None]:
    sidecar = path.with_name(f"{path.stem}-token-counts.json")
    if not sidecar.is_file():
        return {}, None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return {index: int(value) for index, value in enumerate(payload.get("counts", []))}, int(payload["configured_max_completion_tokens"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}, None


def _category_rows(records: Sequence[TraceRecord]) -> list[dict[str, Any]]:
    counts = Counter(r.error_category for r in records if r.data.get("error"))
    return [{"category": category, "count": count} for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _csv_numbers(values: Sequence[str], cast: type = float) -> set[Any]:
    return {cast(item.strip()) for value in values for item in value.split(",") if item.strip()}


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reward", action="append", default=[], help="Reward(s), repeat or comma-separate")
    parser.add_argument("--category", action="append", default=[], help="Exact normalized error category")
    parser.add_argument("--batch", action="append", default=[], help="Batch number(s), repeat or comma-separate")
    parser.add_argument("--rank", action="append", default=[], help="Rank number(s), repeat or comma-separate")
    parser.add_argument("--search", help="Case-insensitive search across question, errors, plan, and outputs")
    parser.add_argument("--question", help="Case-insensitive question substring")
    parser.add_argument("--has-plan", choices=("yes", "no"))
    parser.add_argument("--has-error", choices=("yes", "no"))


def _query_from_args(args: argparse.Namespace) -> TraceQuery:
    return TraceQuery(
        rewards=_csv_numbers(args.reward), categories=set(args.category), batches=_csv_numbers(args.batch, int),
        ranks=_csv_numbers(args.rank, int), search=args.search, question=args.question,
        has_plan=None if args.has_plan is None else args.has_plan == "yes",
        has_error=None if args.has_error is None else args.has_error == "yes",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="theo-trace", description="JSON-first error analysis for conductor JSONL traces.")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output")
    parser.add_argument("--strict", action="store_true", help="Fail instead of reporting malformed JSONL lines")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("summary", "Aggregate rewards, failures, batches, and token saturation"),
        ("errors", "Group failures into normalized categories"),
        ("list", "Return compact matching records"),
        ("questions", "Compare rollout outcomes per question"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("traces", nargs="+", type=Path)
        _add_filters(command)
        if name == "errors":
            command.add_argument("--examples", type=int, default=3)
        if name in {"list", "questions"}:
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--limit", type=int, default=50)
        if name == "list":
            command.add_argument("--full", action="store_true", help="Return complete records rather than compact projections")
        if name == "questions":
            command.add_argument("--min-rollouts", type=int, default=1)
            command.add_argument("--disagreement-only", action="store_true", help="Only questions with more than one reward value")
    show = subparsers.add_parser("show", help="Return one complete record by ID from list/errors output")
    show.add_argument("traces", nargs="+", type=Path)
    show.add_argument("--id", required=True, dest="record_id")
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    dataset = TraceDataset.load(args.traces, strict=args.strict)
    if args.command == "show":
        return dataset.get(args.record_id).full()
    query = _query_from_args(args)
    if args.command == "summary":
        return dataset.summary(query)
    if args.command == "errors":
        rows = dataset.errors(query, examples=max(args.examples, 0))
        return {"groups": len(rows), "errors": rows}
    if args.command == "list":
        matches = dataset.query(query)
        page = matches[max(args.offset, 0): max(args.offset, 0) + max(args.limit, 0)]
        return {"total": len(matches), "offset": max(args.offset, 0), "returned": len(page),
                "records": [record.full() if args.full else record.compact() for record in page]}
    rows = dataset.questions(
        query,
        min_rollouts=max(args.min_rollouts, 1),
        disagreement_only=args.disagreement_only,
    )
    page = rows[max(args.offset, 0): max(args.offset, 0) + max(args.limit, 0)]
    return {"total": len(rows), "offset": max(args.offset, 0), "returned": len(page), "questions": page}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = list(argv) if argv is not None else sys.argv[1:]
        pretty = "--pretty" in args
        result = run_cli(args)
        print(json.dumps(result, ensure_ascii=False, indent=2 if pretty else None, sort_keys=False))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
