"""Resumable evaluation commands sharing checkpoints, catalogues and statistics."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .benchmark import (DEFAULT_INSTRUCTION, JUDGE_INSTRUCTION, RUBRIC_JUDGE_INSTRUCTION,
                        build_judge_batch_question, build_judge_response_format,
                        extract_final_answer, parse_judge_batch, parse_rubric_judge_batch)
from .evaluation_stats import summarize_evaluation, usage_metrics
from .jsonio import read_jsonl, write_json_atomic, write_jsonl_atomic
from .models.registry import ModelRegistry
from .prompt import (DEFAULT_PROMPT_PATH, QUESTION_MARKER, load_prompt_template,
                     validate_plan, workflow_response_format)
from .runner import Runner, WorkflowStepError, _usage_with_estimated_cost
from .schema import Step, StepOutput, Task
from .tools import ToolRegistry
from .worker_prompt import DEFAULT_WORKER_PROMPT_PATH


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def spec_identity(spec: Any) -> dict:
    client = spec.client
    return {'model_id': str(spec.model_idx or spec.name), 'model': getattr(client, 'model', None),
            'base_url': getattr(client, 'base_url', None), 'context_length': spec.context_length,
            'max_output_tokens': spec.max_output_tokens, 'supports_tools': spec.supports_tools,
            'input_price': spec.cost_per_1m_input_tokens, 'output_price': spec.cost_per_1m_output_tokens}


class RunStore:
    """Single-writer run directory; each item is an independently durable snapshot."""
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / '.run.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError(f'Another process is writing {root}') from None
        self.manifest = json.loads((root / 'manifest.json').read_text()) if (root / 'manifest.json').exists() else None
        self.records = {record['record_id']: record for path in sorted((root / 'checkpoints').glob('*.json'))
                        for record in [json.loads(path.read_text())]}

    def initialize(self, identity: dict, settings: dict, *, samples: int) -> None:
        if self.manifest and self.manifest['identity'] != identity:
            raise ValueError('Output directory belongs to a different input/model/mode. Use a new output directory.')
        if self.manifest and samples < self.manifest['samples']:
            raise ValueError('Cannot reduce sample count in an existing run')
        if not self.manifest:
            if (self.root / 'results.jsonl').exists():
                raise ValueError('Existing results have no manifest; import them with judge --results into a new directory')
            self.manifest = {'schema_version': 1, 'run_id': self.root.name, 'created_at': now(),
                             'identity': identity, 'settings_history': []}
        revision = {'fingerprint': fingerprint(settings), 'settings': settings, 'timestamp': now()}
        if not self.manifest['settings_history'] or self.manifest['settings_history'][-1]['fingerprint'] != revision['fingerprint']:
            self.manifest['settings_history'].append(revision)
        self.manifest.update(samples=samples, updated_at=now())
        write_json_atomic(self.root / 'manifest.json', self.manifest)

    def save(self, record: dict) -> None:
        record['updated_at'] = now()
        self.records[record['record_id']] = record
        write_json_atomic(self.root / 'checkpoints' / f"{record['record_id']}.json", record)

    def export(self) -> dict:
        records = sorted(self.records.values(), key=lambda record: (record.get('benchmark_position', 0), record['example_id'], record['sample_index']))
        write_jsonl_atomic(self.root / 'results.jsonl', records)
        plans = [{**record, 'dataset_id': record['example_id'], 'dataset_index': record.get('benchmark_position'),
                  'error': record.get('plan_error'), 'model_id': record.get('planner_model_id')}
                 for record in records if record.get('plan') is not None or record.get('plan_error')]
        if plans:
            write_jsonl_atomic(self.root / 'plans.jsonl', plans)
        summary = {'schema_version': 1, 'run_id': self.manifest['run_id'], 'updated_at': now(),
                   'identity': self.manifest['identity'],
                   **summarize_evaluation(records, samples=self.manifest['samples'])}
        write_json_atomic(self.root / 'summary.json', summary)
        failures = [{'record_id': r['record_id'], 'example_id': r['example_id'], 'sample_index': r['sample_index'],
                     'status': r['status'], 'error': r.get('error'), 'judge_error': r.get('judge_error'),
                     'failed_steps': r.get('failed_steps', {}), 'capped_steps': r.get('capped_steps', [])}
                    for r in records if r.get('error') or r.get('judge_error') or r.get('capped_steps')]
        write_jsonl_atomic(self.root / 'failures.jsonl', failures)
        return summary

    def close(self) -> None:
        self.lock.close()


def new_record(row: dict, index: int, sample: int, model: str) -> dict:
    example = str(row.get('id', row.get('dataset_id', row.get('example_id', ''))))
    if not example.strip() or example == 'None':
        raise ValueError(f'Input row {index} is missing an ID')
    if not isinstance(row.get('question'), str) or not row['question'].strip():
        raise ValueError(f'Input row {index} is missing a non-empty question')
    return {'record_id': fingerprint([example, sample]), 'example_id': example,
            'benchmark_position': index, 'sample_index': sample, 'model_id': model,
            'question': row['question'], 'image': row.get('image'), 'is_multimodal': bool(row.get('image')),
            'gold_answer': row.get('answer', row.get('gold_answer')),
            'reference_answer': row.get('reference_answer'), 'subject': row.get('subject'),
            'answer_type': row.get('answer_type'), 'track': row.get('track'),
            'task_group_id': row.get('task_group_id'), 'status': 'pending', 'error': None,
            'response': None, 'correct': None, 'worker_outputs': {}, 'step_fingerprints': {},
            'failed_steps': {}, 'attempt_history': []}


def clear_judge(record: dict) -> None:
    for key in ('judge_model', 'judge_correct', 'judge_error', 'judge_reason', 'judge_response', 'judge_fingerprint', 'judge_score', 'judge_scoring', 'correct'):
        record[key] = None


def event(record: dict, stage: str, model_id: str, *, output: Any = None,
          error: Exception | None = None, latency: float | None = None, step_id: str | None = None) -> None:
    reason = error.reason if isinstance(error, WorkflowStepError) else type(error).__name__ if error else None
    usage = output.usage if output is not None else None
    record.setdefault('in_flight_requests', {}).pop(f'{stage}:{step_id}', None)
    record['attempt_history'].append({'timestamp': now(), 'stage': stage, 'model_id': model_id,
        'step_id': step_id, 'attempt': len(record['attempt_history']) + 1,
        'usage': usage, 'latency_ms': latency if latency is not None else getattr(output, 'latency_ms', getattr(error, 'elapsed_ms', None)),
        'finish_reason': getattr(output, 'finish_reason', None), 'failure_reason': reason,
        'error': str(error) if error else None, 'response': getattr(output, 'text', None)})


def begin_request(record: dict, store: RunStore, stage: str, model_id: str, step_id: str | None = None) -> None:
    key = f'{stage}:{step_id}'
    previous = record.setdefault('in_flight_requests', {}).get(key)
    if previous:
        record['attempt_history'].append({**previous, 'failure_reason': 'interrupted_request',
                                          'usage': None, 'latency_ms': None})
    record['in_flight_requests'][key] = {'timestamp': now(), 'stage': stage,
                                       'model_id': model_id, 'step_id': step_id}
    store.save(record)


def check_response(response: Any, step_id: str) -> None:
    if response.finish_reason in {'length', 'max_tokens'}:
        raise WorkflowStepError(step_id, 'capped')
    if response.finish_reason in {'error', 'content_filter'}:
        raise WorkflowStepError(step_id, response.finish_reason)
    if not response.text.strip():
        raise WorkflowStepError(step_id, 'empty_response')


async def generate_plan(record: dict, *, store: RunStore, registry: ModelRegistry, planner: str,
                        template: str, model_ids: list[str], args: argparse.Namespace) -> bool:
    spec = registry.get(planner)
    key = fingerprint([spec_identity(spec), template, workflow_response_format(model_ids), record['question'], record.get('image'), args.plan_max_tokens, args.plan_temperature])
    if record.get('plan_fingerprint') == key and record.get('plan') and not record.get('plan_error'):
        return True
    record.update(plan=None, plan_fingerprint=key, planner_model_id=planner, plan_error=None,
                  worker_outputs={}, step_fingerprints={}, response=None, status='planning')
    clear_judge(record)
    store.save(record)
    prompt = template.replace(QUESTION_MARKER, record['question'])
    messages = None
    if record.get('image'):
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': record['image']}}]}]
    for _ in range(args.attempts):
        started = time.perf_counter()
        response = None
        begin_request(record, store, 'planner', planner)
        try:
            response = await spec.client.generate(instruction=prompt, question='', context={},
                messages=messages, max_tokens=args.plan_max_tokens, temperature=args.plan_temperature,
                response_format=workflow_response_format(model_ids))
            response = replace(response, usage=_usage_with_estimated_cost(response.usage, spec))
            check_response(response, 'planner')
            plan = validate_plan(json.loads(response.text), set(model_ids))
            record.update(plan=plan, plan_error=None, error=None, status='planned')
            event(record, 'planner', planner, output=response, latency=(time.perf_counter()-started)*1000)
            store.save(record)
            return True
        except Exception as exc:
            record.update(plan_error=f'{type(exc).__name__}: {exc}', error=f'{type(exc).__name__}: {exc}', status='failed')
            event(record, 'planner', planner, output=response, error=exc, latency=(time.perf_counter()-started)*1000)
            store.save(record)
    return False


async def execute_workflow(record: dict, *, store: RunStore, registry: ModelRegistry, args: argparse.Namespace) -> None:
    task = Task.from_dict({**record['plan'], 'question': record['question'], 'image': record.get('image')})
    runner = Runner(registry, max_worker_tokens=args.max_tokens,
                    use_model_output_limits=args.max_tokens is None, worker_temperature=args.temperature)
    cached: dict[str, StepOutput] = {}
    def request_key(spec, request):
        tool_models = [spec_identity(registry.get(key)) for key in registry.model_ids()] if spec.supports_tools else []
        return fingerprint([spec_identity(spec), request, tool_models, DEFAULT_WORKER_PROMPT_PATH.read_text()])
    # Recompute request identities in order. If a dependency is invalidated, so are its descendants.
    for step in task.workflow:
        dependencies = set(step.access_list) & {s.step_id for s in task.workflow}
        previous = record.get('worker_outputs', {}).get(step.step_id)
        if previous and dependencies <= cached.keys():
            spec, request = runner.prepare_step(step, task, cached)
            key = request_key(spec, request)
            if record.get('step_fingerprints', {}).get(step.step_id) == key:
                output = StepOutput(**previous)
                from .runner import step_failure
                if not step_failure(output):
                    cached[step.step_id] = output
    if record.get('status') == 'completed' and len(cached) == len(task.workflow):
        return
    record.update(worker_outputs={key: value.model_dump(mode='json') for key, value in cached.items()},
                  response=None, status='running', error=None, failed_steps={}, capped_steps=[])
    clear_judge(record)
    store.save(record)
    started = time.perf_counter()
    previous_runtime = record.get('workflow_runtime_ms', 0) or 0

    async def call_checkpoint(step, spec, response, error, elapsed):
        if response:
            response = replace(response, usage=_usage_with_estimated_cost(response.usage, spec))
            try:
                if not response.tool_calls:
                    check_response(response, step.step_id)
            except WorkflowStepError as exc:
                error = exc
        event(record, 'worker', str(step.model_id), output=response, error=error,
              latency=elapsed, step_id=step.step_id)
        store.save(record)

    async def call_started(step, spec, request):
        begin_request(record, store, 'worker', str(step.model_id), step.step_id)

    async def tool_checkpoint(step, call):
        record['attempt_history'].append({'timestamp': now(), 'stage': 'tool',
            'model_id': call.model_id or call.name, 'step_id': step.step_id,
            'usage': call.usage, 'latency_ms': call.duration_ms,
            'failure_reason': 'tool_error' if call.is_error else None})
        store.save(record)

    runner.call_checkpoint = call_checkpoint
    runner.tool_checkpoint = tool_checkpoint
    runner.call_started = call_started

    async def checkpoint(step: Step, output: StepOutput | None, error: Exception | None) -> None:
        context_outputs = {key: StepOutput(**value) for key, value in record['worker_outputs'].items()}
        spec, request = runner.prepare_step(step, task, context_outputs)
        record['step_fingerprints'][step.step_id] = request_key(spec, request)
        if output is not None:
            record['worker_outputs'][step.step_id] = output.model_dump(mode='json')
        if error:
            record['failed_steps'][step.step_id] = {'reason': getattr(error, 'reason', type(error).__name__), 'error': str(error)}
            if getattr(error, 'reason', None) == 'capped':
                if step.step_id not in record['capped_steps']:
                    record['capped_steps'].append(step.step_id)
        else:
            record['failed_steps'].pop(step.step_id, None)
            if step.step_id in record['capped_steps']:
                record['capped_steps'].remove(step.step_id)
        record['workflow_runtime_ms'] = previous_runtime + (time.perf_counter()-started)*1000
        store.save(record)

    for _ in range(args.attempts):
        try:
            result = await runner.run(task, initial_outputs={key: StepOutput(**value) for key, value in record['worker_outputs'].items()},
                                      checkpoint=checkpoint, fail_on_incomplete=True)
            record.update(response=result.outputs[task.workflow[-1].step_id].text, status='completed', error=None,
                          workflow_peak_concurrency=result.observed_peak_concurrency)
            record['extracted_answer'] = extract_final_answer(record['response'])
            break
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}', status='failed')
            store.save(record)
    record['workflow_runtime_ms'] = previous_runtime + (time.perf_counter()-started)*1000
    totals = [usage_metrics(output.get('usage')) for output in record['worker_outputs'].values()]
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [item[field] for item in totals if item[field] is not None]
        record[field] = sum(values) if values else None
    costs = [item['cost_usd'] for item in totals if item['cost_usd'] is not None]
    record['estimated_cost_usd'] = sum(costs) if costs else None
    record['latency_ms'] = record['workflow_runtime_ms']
    store.save(record)


async def generate_model(record: dict, *, store: RunStore, registry: ModelRegistry, model: str, args: argparse.Namespace) -> None:
    spec = registry.get(model)
    key = fingerprint([spec_identity(spec), record['question'], record.get('image'), args.max_tokens, args.temperature, DEFAULT_WORKER_PROMPT_PATH.read_text()])
    if record.get('generation_fingerprint') == key and record.get('status') == 'completed':
        return
    record.update(generation_fingerprint=key, response=None, status='running', error=None)
    clear_judge(record)
    store.save(record)
    runner = Runner(registry, tool_registry=ToolRegistry(), max_worker_tokens=args.max_tokens,
                    use_model_output_limits=args.max_tokens is None, worker_temperature=args.temperature)
    step = Step(step_id='final', model_id=model, instruction=DEFAULT_INSTRUCTION, access_list=['question'])
    task = Task(task_type='other', difficulty='hard', question=record['question'], image=record.get('image'), workflow=[step])
    for _ in range(args.attempts):
        started = time.perf_counter()
        output = None
        begin_request(record, store, 'solver', model)
        try:
            output = await runner.run_step(step, task, {})
            check_response(output, 'final')
            record.update(response=output.text, extracted_answer=extract_final_answer(output.text), status='completed', error=None,
                          finish_reason=output.finish_reason, latency_ms=output.latency_ms,
                          **{key: value for key, value in usage_metrics(output.usage).items() if key in {'prompt_tokens', 'completion_tokens', 'total_tokens'}},
                          estimated_cost_usd=usage_metrics(output.usage)['cost_usd'])
            event(record, 'solver', model, output=output, latency=(time.perf_counter()-started)*1000)
            store.save(record)
            return
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}', status='failed', finish_reason=getattr(output, 'finish_reason', None))
            event(record, 'solver', model, output=output, error=exc, latency=(time.perf_counter()-started)*1000)
            store.save(record)


async def judge_completion(record: dict, *, store: RunStore, registry: ModelRegistry, judge: str, args: argparse.Namespace) -> None:
    if record.get('error') or not record.get('response'):
        return
    if record.get('gold_answer') is None:
        record.update(judge_error='Missing gold_answer: cannot judge this completion', judge_correct=None, correct=None)
        store.save(record)
        return
    spec = registry.get(judge)
    rubric = record.get('answer_type') == 'rubric10'
    instruction = RUBRIC_JUDGE_INSTRUCTION if rubric else JUDGE_INSTRUCTION
    key = fingerprint([spec_identity(spec), record['question'], record.get('gold_answer'), record.get('reference_answer'),
                       record['response'], args.judge_max_tokens, instruction])
    if record.get('judge_fingerprint') == key and isinstance(record.get('judge_correct'), bool) and not args.force:
        return
    clear_judge(record)
    record.update(judge_fingerprint=key, judge_model=judge)
    store.save(record)
    for _ in range(args.attempts):
        started = time.perf_counter()
        response = None
        begin_request(record, store, 'judge', judge)
        try:
            response = await spec.client.generate(instruction=instruction,
                question=build_judge_batch_question([('item', record)]), context={}, temperature=0,
                max_tokens=args.judge_max_tokens, response_format=build_judge_response_format(['item'], rubric=rubric))
            response = replace(response, usage=_usage_with_estimated_cost(response.usage, spec))
            check_response(response, 'judge')
            value, reason = (parse_rubric_judge_batch if rubric else parse_judge_batch)(response.text, ['item'])['item']
            correct = value >= 7 if rubric else value
            record.update(judge_correct=correct, correct=correct, judge_reason=reason, judge_response=response.text, judge_error=None)
            record.update(judge_score=value if rubric else None, judge_scoring='rubric10-v1' if rubric else 'boolean')
            event(record, 'judge', judge, output=response, latency=(time.perf_counter()-started)*1000)
            store.save(record)
            return
        except Exception as exc:
            record.update(judge_error=f'{type(exc).__name__}: {exc}', judge_correct=None, correct=None)
            event(record, 'judge', judge, output=response, error=exc, latency=(time.perf_counter()-started)*1000)
            store.save(record)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog='theo-evaluate', description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('plans', 'workflows', 'model', 'judge'):
        command = commands.add_parser(name)
        command.add_argument('--output-dir', type=Path, required=True)
        command.add_argument('--config', type=Path, required=True, help='Worker/planner/solver/judge registry YAML')
        command.add_argument('--concurrency', type=int, default=4)
        command.add_argument('--attempts', type=int, default=3, help='Recovery attempts per invocation; never independent samples')
        command.add_argument('--judge-model', help='Registry model ID; omit to generate without judging')
        command.add_argument('--judge-max-tokens', type=int, default=8192)
        command.add_argument('--force', action='store_true', help='Rejudge existing verdicts')
        if name != 'judge':
            command.add_argument('--samples', type=int, default=1, help='Independent samples per question; may increase on resume')
            command.add_argument('--max-tokens', type=int, help='Worker/solver cap; default: configured model output limit')
            command.add_argument('--temperature', type=float, default=.2)
        if name in ('plans', 'workflows'):
            command.add_argument('--planner-model', help='Registry model ID')
            command.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT_PATH)
            command.add_argument('--plan-max-tokens', type=int, default=8192)
            command.add_argument('--plan-temperature', type=float, default=.1)
        if name in ('plans', 'model'):
            command.add_argument('--input', type=Path, required=True, help='Question JSONL: id, question, answer, optional image')
        if name == 'workflows':
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument('--input', type=Path, help='Generate plans and execute them')
            source.add_argument('--plans', type=Path, help='Execute plans.jsonl or a plan run directory')
        if name == 'model':
            command.add_argument('--model', required=True, help='Exactly one registry model ID')
        if name == 'judge':
            command.add_argument('--results', type=Path, help='Import legacy completion JSONL into a new run; omit to judge this run')
    catalog = commands.add_parser('catalog')
    catalog.add_argument('--root', type=Path, default=Path('outputs'))
    catalog.add_argument('--output', type=Path, help='Optionally save the catalogue as JSON')
    aggregate = commands.add_parser('aggregate')
    aggregate.add_argument('--runs', type=Path, nargs='+', required=True)
    aggregate.add_argument('--output', type=Path, required=True)
    return result


async def run(args: argparse.Namespace, registry: ModelRegistry | None = None) -> dict:
    if args.command == 'catalog':
        entries = []
        for path in sorted(args.root.rglob('manifest.json')):
            manifest = json.loads(path.read_text())
            summary = json.loads(path.with_name('summary.json').read_text()) if path.with_name('summary.json').exists() else None
            entries.append({'path': str(path.parent.resolve()), 'manifest': manifest, 'summary': summary})
        for path in sorted(args.root.rglob('summary.json')):
            if not path.with_name('manifest.json').exists():
                entries.append({'path': str(path.parent.resolve()), 'legacy': True,
                                'summary': json.loads(path.read_text())})
        catalog = {'runs': entries}
        if args.output:
            write_json_atomic(args.output, catalog)
        return catalog
    if args.command == 'aggregate':
        records = []
        identity = None
        sample_offset = 0
        seen_roots = set()
        generation_settings = None
        cohort_ids = None
        judge_ids = None
        for root in args.runs:
            if root.resolve() in seen_roots:
                raise ValueError('Cannot aggregate the same run twice')
            seen_roots.add(root.resolve())
            manifest = json.loads((root / 'manifest.json').read_text())
            if identity is not None and manifest['identity'] != identity:
                raise ValueError('Aggregate runs must have identical input/model/mode identities')
            identity = manifest['identity']
            settings = {key: value for key, value in manifest['settings_history'][-1]['settings'].items()
                        if key in {'config_sha256', 'max_tokens', 'temperature', 'plan_max_tokens', 'plan_temperature'}}
            if generation_settings is not None and settings != generation_settings:
                raise ValueError('Aggregate runs have different generation settings')
            generation_settings = settings
            rows = read_jsonl(root / 'results.jsonl')
            if not rows:
                raise ValueError(f'Run has no results: {root}')
            ids = {row['example_id'] for row in rows}
            judges = {row.get('judge_model') for row in rows if row.get('judge_model')}
            if cohort_ids is not None and ids != cohort_ids:
                raise ValueError('Aggregate runs have different question sets')
            if judge_ids is not None and judges != judge_ids:
                raise ValueError('Aggregate runs have different judges')
            cohort_ids, judge_ids = ids, judges
            for row in rows:
                row['sample_index'] += sample_offset
                records.append(row)
            sample_offset += manifest['samples']
        summary = summarize_evaluation(records, samples=sample_offset)
        write_json_atomic(args.output, summary)
        return summary
    for key in ('concurrency', 'attempts', 'judge_max_tokens'):
        if getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive')
    if args.command != 'judge' and (args.samples <= 0 or args.max_tokens is not None and args.max_tokens <= 0):
        raise ValueError('samples and token limits must be positive')
    if args.command in ('plans', 'workflows') and args.plan_max_tokens <= 0:
        raise ValueError('plan-max-tokens must be positive')
    registry = registry or ModelRegistry.from_yaml_file(args.config)
    if args.judge_model:
        registry.get(args.judge_model)
    if args.command == 'judge' and not args.judge_model:
        raise ValueError('judge requires --judge-model')
    store = RunStore(args.output_dir)
    try:
        if args.command == 'judge' and not args.results:
            if store.manifest is None:
                raise ValueError('No run manifest; use --results to import existing completions')
            records = list(store.records.values())
        else:
            source = args.results if args.command == 'judge' else args.plans if args.command == 'workflows' and args.plans else args.input
            if source.is_dir():
                source = source / 'plans.jsonl'
            rows = read_jsonl(source)
            if not rows:
                raise ValueError('Input contains no records')
            if any(int(row.get('sample_index', 0)) < 0 for row in rows):
                raise ValueError('sample_index must be non-negative')
            if args.command == 'workflows' and args.plans:
                # Multiple stored independent plans are distinct samples, not last-record-wins retries.
                indexed = {}
                for row in rows:
                    key = (str(row.get('dataset_id', row.get('example_id'))), int(row.get('sample_index', 0)))
                    indexed[key] = row
                rows = list(indexed.values())
            ids = [str(row.get('id', row.get('dataset_id', row.get('example_id', '')))) for row in rows]
            if args.command == 'judge':
                # Judge-only imports are homogeneous cohorts; mixed model files must be split first.
                if len({str(row.get('model_id')) for row in rows}) > 1:
                    raise ValueError('Judge imports require one solver model per run')
                rows = list({(str(row.get('example_id', row.get('id'))), int(row.get('sample_index', 0))): row for row in rows}.values())
            if args.command not in ('judge', 'workflows') or args.command == 'workflows' and not args.plans:
                if len(set(ids)) != len(ids):
                    raise ValueError('Input IDs must be unique')
            model = args.model if args.command == 'model' else 'theo-conductor'
            if args.command == 'model':
                registry.get(model)
            identity = {'mode': args.command, 'input_sha256': fingerprint(rows), 'model': model,
                        'planner': getattr(args, 'planner_model', None)}
            samples = 1 if args.command == 'judge' else args.samples
            if args.command == 'judge':
                samples = 1 + max(int(row.get('sample_index', 0)) for row in rows)
            if args.command == 'workflows' and args.plans:
                source_samples = 1 + max(int(row.get('sample_index', 0)) for row in rows)
                if args.samples not in {1, source_samples}:
                    raise ValueError('--plans executes stored samples; generate additional plans to increase k')
                samples = source_samples
            settings = {'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                        **{key: value for key, value in vars(args).items() if key not in {'config', 'output_dir', 'input', 'plans', 'results'} and not isinstance(value, Path)}}
            store.initialize(identity, settings, samples=samples)
            records = []
            for index, row in enumerate(rows):
                indices = [int(row.get('sample_index', 0))] if args.command == 'judge' or args.command == 'workflows' and args.plans else range(samples)
                for sample in indices:
                    fresh = new_record(row, index, sample, model)
                    record = store.records.get(fresh['record_id'], fresh)
                    if args.command == 'workflows' and args.plans:
                        record.update(plan=row.get('plan'), plan_error=row.get('error'), planner_model_id=row.get('planner_model_id', row.get('conductor_model')))
                        source_history = [event for event in row.get('attempt_history', []) if event.get('stage') == 'planner']
                        if not source_history and row.get('conductor_performance'):
                            performance = row['conductor_performance']
                            source_history = [{'stage': 'planner', 'model_id': record['planner_model_id'],
                                               'usage': performance.get('usage'), 'latency_ms': performance.get('batch_latency_ms')}]
                        record['source_planner_history'] = source_history
                    if args.command == 'judge' and record is fresh:
                        record_id = fresh['record_id']
                        source_history = [event for event in row.get('attempt_history', []) if event.get('stage') != 'judge']
                        if not source_history:
                            source_history = [{'stage': 'solver', 'model_id': row.get('model_id'),
                                'usage': {key: row.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'estimated_cost_usd')},
                                'latency_ms': row.get('latency_ms')}]
                        record.update(row)
                        record.update(record_id=record_id, sample_index=sample,
                                      status='failed' if row.get('error') else 'completed', attempt_history=[],
                                      source_completion_history=source_history, in_flight_requests={})
                        clear_judge(record)
                    records.append(record)
                    store.save(record)
        template = model_ids = None
        if args.command == 'plans' or args.command == 'workflows' and not args.plans:
            if not args.planner_model:
                raise ValueError('Plan generation requires --planner-model')
            registry.get(args.planner_model)
            template, model_ids = load_prompt_template(args.prompt_file)
            for model_id in model_ids:
                registry.get(model_id)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def process(record: dict) -> None:
            async with semaphore:
                try:
                    if template is not None:
                        if not await generate_plan(record, store=store, registry=registry, planner=args.planner_model,
                                                   template=template, model_ids=model_ids, args=args):
                            return
                    if args.command == 'workflows':
                        if not record.get('plan') or record.get('plan_error'):
                            raise ValueError(f"Plan unavailable: {record.get('plan_error')}")
                        await execute_workflow(record, store=store, registry=registry, args=args)
                    elif args.command == 'model':
                        await generate_model(record, store=store, registry=registry, model=args.model, args=args)
                    if args.judge_model and args.command != 'plans':
                        await judge_completion(record, store=store, registry=registry, judge=args.judge_model, args=args)
                except Exception as exc:
                    record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    store.save(record)
                print(f"[{args.command}] {record['example_id']} sample={record['sample_index']}: {record['status']}", file=sys.stderr)
        tasks = [asyncio.create_task(process(record)) for record in records]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return store.export()
    finally:
        try:
            if store.manifest:
                store.export()
        finally:
            store.close()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    try:
        summary = asyncio.run(run(parser().parse_args(argv)))
        print(json.dumps(summary, indent=2))
        return 1 if summary.get('generation_failures') or summary.get('judge_errors') else 0
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
