import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from theo_conductor.evaluate import RunStore, parser, run
from theo_conductor.evaluation_stats import repeated_metrics, summarize_evaluation
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prompt import validate_plan
from theo_conductor.schema import ModelResponse, ModelSpec


class Client:
    def __init__(self):
        self.calls = Counter()
        self.fail_step = None
        self.cap_step = None
        self.bad_judge = False

    async def generate(self, instruction, **kwargs):
        if kwargs.get('response_format', {}).get('json_schema', {}).get('name') == 'conductor_workflow':
            self.calls['planner'] += 1
            return ModelResponse(text=json.dumps(plan()), usage={'input_tokens': 10, 'output_tokens': 5}, latency_ms=2)
        if 'expected_answer' in kwargs.get('question', '') or kwargs.get('response_format'):
            self.calls['judge'] += 1
            if self.bad_judge:
                raise RuntimeError('judge unavailable')
            return ModelResponse(text='[{"id":"item","correct":true,"reason":"matches"}]', usage={'input_tokens': 3, 'output_tokens': 2})
        name = instruction.split()[0]
        self.calls[name] += 1
        if self.fail_step == name:
            raise RuntimeError('temporary failure')
        return ModelResponse(text='FINAL: 4', finish_reason='length' if self.cap_step == name else 'stop',
                             usage={'input_tokens': 10, 'output_tokens': 5}, latency_ms=2)


def plan():
    return {'task_type': 'physics', 'difficulty': 'hard', 'workflow': [
        {'step_id': 'a', 'model_id': 'worker', 'instruction': 'first', 'access_list': ['question']},
        {'step_id': 'b', 'model_id': 'worker', 'instruction': 'second', 'access_list': ['a']},
        {'step_id': 'final', 'model_id': 'worker', 'instruction': 'last', 'access_list': ['b']}]}


def setup(tmp_path):
    client = Client()
    registry = ModelRegistry([ModelSpec(model_idx=name, client=client, cost_per_1m_input_tokens=1,
                                        cost_per_1m_output_tokens=2) for name in ('worker', 'planner', 'judge')])
    config = tmp_path / 'config.yaml'
    config.write_text('models: []')
    source = tmp_path / 'questions.jsonl'
    source.write_text(json.dumps({'id': 'q', 'question': '2+2?', 'answer': '4'}) + '\n')
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('- model_id="worker"\n<USER QUESTION>')
    return client, registry, config, source, prompt


def options(tmp_path, command, config, source, prompt, extra=()):
    argv = [command, '--config', str(config), '--output-dir', str(tmp_path / 'run'), '--attempts', '1']
    if command != 'judge':
        argv += ['--input', str(source)]
    if command in ('plans', 'workflows'):
        argv += ['--planner-model', 'planner', '--prompt-file', str(prompt)]
    if command == 'model':
        argv += ['--model', 'worker']
    return parser().parse_args(argv + list(extra))


def results(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'run/results.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('score,correct', [(6.5, False), (7, True), (10, True)])
def test_research_rubric_grading_and_resume(tmp_path, score, correct):
    client, registry, config, source, prompt = setup(tmp_path)
    source.write_text(json.dumps({'id': 'research', 'question': 'Research question',
                                 'answer': 'Secret ten-point rubric', 'answer_type': 'rubric10'}) + '\n')
    original = client.generate

    async def generate(instruction, **kwargs):
        if kwargs.get('response_format'):
            client.calls['judge'] += 1
            assert 'ten-point' in instruction
            assert 'Secret ten-point rubric' in kwargs['question']
            return ModelResponse(text=json.dumps([{'id': 'item', 'score': score, 'reason': 'criterion points'}]))
        assert 'Secret ten-point rubric' not in kwargs.get('question', '')
        return await original(instruction, **kwargs)

    client.generate = generate
    args = options(tmp_path, 'model', config, source, prompt, ['--judge-model', 'judge'])
    asyncio.run(run(args, registry))
    row = results(tmp_path)[0]
    assert row['judge_score'] == score
    assert row['correct'] is correct
    asyncio.run(run(args, registry))
    assert client.calls['judge'] == 1


def test_workflow_failure_preserves_steps_and_resumes_then_only_judges(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'workflows', config, source, prompt)
    client.fail_step = 'second'
    first = asyncio.run(run(args, registry))
    assert first['generation_failures'] == 1
    assert set(results(tmp_path)[0]['worker_outputs']) == {'a'}
    client.fail_step = None
    recovered = asyncio.run(run(args, registry))
    assert recovered['generation_failures'] == 0
    assert client.calls == {'planner': 1, 'first': 1, 'second': 2, 'last': 1}
    asyncio.run(run(args, registry))
    assert client.calls['last'] == 1
    judge_args = options(tmp_path, 'judge', config, source, prompt, ['--judge-model', 'judge'])
    summary = asyncio.run(run(judge_args, registry))
    assert summary['repeated_sampling']['pass_at_k'] == 1
    asyncio.run(run(judge_args, registry))
    assert client.calls['judge'] == 1
    assert summary['all_attempts']['total_tokens']['sum'] == 65


def test_capped_step_retried_but_successful_ancestor_reused(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'workflows', config, source, prompt)
    client.cap_step = 'second'
    summary = asyncio.run(run(args, registry))
    assert summary['failure_reasons'] == {'capped': 1}
    assert results(tmp_path)[0]['capped_steps'] == ['b']
    assert client.calls['last'] == 0
    client.cap_step = None
    asyncio.run(run(args, registry))
    assert client.calls['first'] == 1
    assert results(tmp_path)[0]['capped_steps'] == []
    # Changing the cap invalidates all requests affected by it.
    args.max_tokens = 999
    asyncio.run(run(args, registry))
    assert client.calls['planner'] == 1
    assert client.calls['first'] == 2
    assert client.calls['last'] == 2


def test_independent_samples_can_be_extended_without_repeating_completed_samples(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'model', config, source, prompt, ['--samples', '2', '--judge-model', 'judge'])
    summary = asyncio.run(run(args, registry))
    assert len(results(tmp_path)) == 2
    assert summary['repeated_sampling']['pass_at_k'] == 1
    calls = sum(client.calls.values())
    asyncio.run(run(args, registry))
    assert sum(client.calls.values()) == calls
    args.samples = 3
    asyncio.run(run(args, registry))
    assert len(results(tmp_path)) == 3
    assert sum(client.calls.values()) == calls + 2


def test_plan_only_never_executes_and_exports_consumable_plans(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'plans', config, source, prompt)
    asyncio.run(run(args, registry))
    assert client.calls == {'planner': 1}
    plans = tmp_path / 'run/plans.jsonl'
    row = json.loads(plans.read_text())
    assert row['dataset_id'] == 'q'
    args = parser().parse_args(['workflows', '--plans', str(plans), '--config', str(config),
                               '--output-dir', str(tmp_path/'execution'), '--attempts', '1'])
    asyncio.run(run(args, registry))
    assert client.calls['planner'] == 1
    assert client.calls['last'] == 1
    summary = json.loads((tmp_path/'execution/summary.json').read_text())
    assert summary['source_planner']['total_tokens']['sum'] == 15
    assert summary['all_attempts']['total_tokens']['sum'] == 45


def test_judge_error_remains_unknown_and_resumes_without_regeneration(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'model', config, source, prompt, ['--judge-model', 'judge'])
    client.bad_judge = True
    summary = asyncio.run(run(args, registry))
    assert summary['judge_errors'] == 1
    assert summary['repeated_sampling']['pass_at_k'] is None
    assert results(tmp_path)[0]['correct'] is None
    before = client.calls.copy()
    client.bad_judge = False
    asyncio.run(run(args, registry))
    assert client.calls['Solve'] == before['Solve']
    assert client.calls['judge'] == 2


def test_identity_mismatch_and_concurrent_writer_rejected(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'model', config, source, prompt)
    asyncio.run(run(args, registry))
    source.write_text(json.dumps({'id':'q', 'question':'changed', 'answer':'4'})+'\n')
    with pytest.raises(ValueError, match='different input'):
        asyncio.run(run(args, registry))
    store = RunStore(tmp_path / 'run')
    try:
        with pytest.raises(ValueError, match='Another process'):
            RunStore(tmp_path / 'run')
    finally:
        store.close()


def test_repeat_metrics_use_samples_not_retries_and_require_complete_cohorts():
    records = [{'example_id':'q', 'sample_index':i, 'status':'completed', 'judge_correct': outcome}
               for i, outcome in enumerate([True, False, False, True])]
    stats = repeated_metrics(records, 4)
    assert stats['mean_at_k'] == .5
    assert stats['pass_at_k'] == 1
    assert stats['by_k']['2']['pass_at_k'] == pytest.approx(5/6)
    assert repeated_metrics(records[:3], 4)['pass_at_k'] is None
    records[0]['judge_error'] = 'unavailable'
    assert repeated_metrics(records, 4)['pass_at_k'] is None


def test_validation_rejects_missing_instruction():
    invalid = plan()
    del invalid['workflow'][0]['instruction']
    with pytest.raises(ValueError, match='instruction'):
        validate_plan(invalid, {'worker'})


def test_interruption_keeps_step_checkpoint_and_recovers(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    original = client.generate

    async def interrupted(instruction, **kwargs):
        if instruction == 'second':
            raise asyncio.CancelledError()
        return await original(instruction, **kwargs)

    client.generate = interrupted
    args = options(tmp_path, 'workflows', config, source, prompt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run(args, registry))
    assert set(results(tmp_path)[0]['worker_outputs']) == {'a'}
    client.generate = original
    asyncio.run(run(args, registry))
    assert client.calls['first'] == 1
    assert client.calls['last'] == 1


def test_parallel_sibling_success_survives_failure(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    workflow = plan()
    workflow['workflow'][1]['access_list'] = ['question']
    workflow['workflow'][2]['access_list'] = ['a', 'b']
    source.write_text(json.dumps({'dataset_id':'q', 'question':'Q', 'gold_answer':'4', 'plan':workflow})+'\n')
    args = parser().parse_args(['workflows', '--plans', str(source), '--config', str(config),
                               '--output-dir', str(tmp_path/'run'), '--attempts', '1'])
    client.fail_step = 'second'
    asyncio.run(run(args, registry))
    assert set(results(tmp_path)[0]['worker_outputs']) == {'a'}
    client.fail_step = None
    asyncio.run(run(args, registry))
    assert client.calls['first'] == 1
    assert client.calls['second'] == 2


def test_catalog_and_aggregate_validate_run_cohorts(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    args = options(tmp_path, 'model', config, source, prompt, ['--samples', '2', '--judge-model', 'judge'])
    asyncio.run(run(args, registry))
    args.output_dir = tmp_path/'run2'
    asyncio.run(run(args, registry))
    catalog = asyncio.run(run(parser().parse_args(['catalog', '--root', str(tmp_path)])))
    assert len(catalog['runs']) == 2
    aggregate = parser().parse_args(['aggregate', '--runs', str(tmp_path/'run'), str(tmp_path/'run2'),
                                     '--output', str(tmp_path/'aggregate.json')])
    summary = asyncio.run(run(aggregate))
    assert summary['repeated_sampling']['independent_samples'] == 4
    assert summary['repeated_sampling']['pass_at_k'] == 1
    aggregate.runs = [tmp_path/'run', tmp_path/'run']
    with pytest.raises(ValueError, match='same run twice'):
        asyncio.run(run(aggregate))


def test_import_judge_completions_preserves_original_and_latest_result(tmp_path):
    client, registry, config, source, prompt = setup(tmp_path)
    source.write_text('\n'.join(json.dumps({'example_id':'q', 'question':'Q', 'gold_answer':'4',
                                          'model_id':'worker', 'response':text, 'prompt_tokens':10, 'completion_tokens':5})
                                for text in ('old', 'FINAL: 4'))+'\n')
    original = source.read_bytes()
    args = options(tmp_path, 'judge', config, source, prompt, ['--results', str(source), '--judge-model','judge'])
    summary = asyncio.run(run(args, registry))
    assert summary['records'] == 1
    assert summary['source_completions']['total_tokens']['sum'] == 15
    assert results(tmp_path)[0]['response'] == 'FINAL: 4'
    assert source.read_bytes() == original
    asyncio.run(run(args, registry))
    assert client.calls['judge'] == 1


def test_stats_report_unknown_usage_and_separate_price_estimates():
    record = {'example_id':'q', 'status':'completed', 'sample_index':0,
              'attempt_history':[{'stage':'solver', 'model_id':'m', 'usage':{'input_tokens':10, 'output_tokens':5,
                                                                       'estimated_cost_usd':.1, 'cost':.07}},
                                 {'stage':'solver', 'model_id':'m', 'usage':None, 'failure_reason':'RuntimeError'}]}
    summary = summarize_evaluation([record])
    assert summary['all_attempts']['total_tokens']['sum'] == 15
    assert summary['all_attempts']['cost_usd']['sum'] == .07
    assert summary['all_attempts']['estimated_cost_usd']['sum'] == .1
    assert summary['all_attempts']['total_tokens']['unknown'] == 1


def test_configured_conductor_handles_workflow_clarifications():
    from theo_conductor.tools import default_tool_registry
    registry = ModelRegistry([ModelSpec(model_idx='planner', client=Client())], conductor_model='planner')
    tool = default_tool_registry(registry)._tools['request_clarification']
    assert tool.handler._model()[0] == 'planner'
