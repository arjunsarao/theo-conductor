"""Common statistics for plans, model completions, workflows and their attempts."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def usage_metrics(usage: dict | None) -> dict:
    usage = usage or {}
    prompt = numeric(usage.get('prompt_tokens', usage.get('input_tokens')))
    completion = numeric(usage.get('completion_tokens', usage.get('output_tokens')))
    total = numeric(usage.get('total_tokens'))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': total,
            'cost_usd': numeric(usage.get('cost', usage.get('estimated_cost_usd'))),
            'reported_cost_usd': numeric(usage.get('cost')),
            'estimated_cost_usd': numeric(usage.get('estimated_cost_usd'))}


def metric_summary(items: list[dict]) -> dict:
    result = {'observations': len(items)}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd', 'reported_cost_usd', 'estimated_cost_usd', 'latency_ms'):
        values = [value for item in items if (value := numeric(item.get(key))) is not None]
        ordered = sorted(values)
        result[key] = {'known': len(values), 'unknown': len(items) - len(values),
                       'sum': sum(values) if values else None, 'mean': mean(values) if values else None,
                       'p50': ordered[(len(ordered)-1)//2] if ordered else None,
                       'p95': ordered[math.ceil(len(ordered)*.95)-1] if ordered else None}
    return result


def repeated_metrics(records: list[dict], samples: int) -> dict:
    """Recovery retries never count as independent samples; incomplete cohorts stay null."""
    if samples <= 0:
        raise ValueError('Independent sample count must be positive')
    groups = defaultdict(dict)
    for record in records:
        sample = int(record.get('sample_index', 0))
        if sample in groups[str(record['example_id'])]:
            raise ValueError('Duplicate independent sample in aggregate input')
        groups[str(record['example_id'])][sample] = record
    complete = []
    for runs in groups.values():
        if set(runs) != set(range(samples)):
            continue
        outcomes = []
        for record in runs.values():
            if record.get('plan_error'):
                break
            if record.get('status') == 'failed' and record.get('error'):
                outcomes.append(False)
            elif record.get('judge_error') or not isinstance(record.get('judge_correct'), bool):
                break
            else:
                outcomes.append(record['judge_correct'])
        if len(outcomes) == samples:
            complete.append(outcomes)
    by_k = {}
    for k in range(1, samples + 1):
        estimates = [1 - (math.comb(samples - sum(outcomes), k) / math.comb(samples, k)
                          if samples - sum(outcomes) >= k else 0) for outcomes in complete]
        full = len(complete) == len(groups) and bool(groups)
        by_k[str(k)] = {'mean_at_k': mean(mean(outcomes) for outcomes in complete) if full else None,
                        'pass_at_k': mean(estimates) if full else None,
                        'complete_questions': len(complete), 'expected_questions': len(groups)}
    return {'independent_samples': samples, 'by_k': by_k,
            'mean_at_k': by_k[str(samples)]['mean_at_k'], 'pass_at_k': by_k[str(samples)]['pass_at_k']}


def summarize_evaluation(records: list[dict], *, samples: int = 1) -> dict:
    events = [event for record in records for event in record.get('attempt_history', [])]
    active = [event for record in records for event in record.get('in_flight_requests', {}).values()]
    events.extend(active)
    stages = defaultdict(list)
    models = defaultdict(list)
    failures = Counter()
    for event in events:
        metrics = {**usage_metrics(event.get('usage')), 'latency_ms': event.get('latency_ms')}
        stages[event['stage']].append(metrics)
        models[f"{event['stage']}:{event.get('model_id', 'unknown')}"].append(metrics)
        if event.get('failure_reason'):
            failures[event['failure_reason']] += 1
    step_models = Counter(str(step['model_id']) for record in records
                          for step in (record.get('plan') or {}).get('workflow', []))
    source_events = [event for record in records for event in record.get('source_planner_history', [])]
    source_items = [{**usage_metrics(event.get('usage')), 'latency_ms': event.get('latency_ms')} for event in source_events]
    imported_events = [event for record in records for event in record.get('source_completion_history', [])]
    imported_items = [{**usage_metrics(event.get('usage')), 'latency_ms': event.get('latency_ms')} for event in imported_events]
    return {'records': len(records), 'statuses': dict(Counter(record.get('status', 'unknown') for record in records)),
            'in_flight_requests': len(active),
            'generation_failures': sum(bool(record.get('error')) for record in records),
            'judge_errors': sum(bool(record.get('judge_error')) for record in records),
            'failure_reasons': dict(failures), 'planned_model_distribution': dict(step_models),
            'attempt_model_distribution': dict(Counter(event.get('model_id', 'unknown') for event in events)),
            'stages': {stage: metric_summary(items) for stage, items in stages.items()},
            'models': {model: metric_summary(items) for model, items in models.items()},
            'all_attempts': metric_summary([{**usage_metrics(event.get('usage')), 'latency_ms': event.get('latency_ms')} for event in events]),
            'source_planner': metric_summary(source_items),
            'source_completions': metric_summary(imported_items),
            'end_to_end_observed': metric_summary(source_items + imported_items + [{**usage_metrics(event.get('usage')), 'latency_ms': event.get('latency_ms')} for event in events]),
            'workflow_latency_ms': metric_summary([{'latency_ms': record.get('workflow_runtime_ms')} for record in records])['latency_ms'],
            'repeated_sampling': repeated_metrics(records, samples)}
