import asyncio
import base64
import hashlib
import json
from collections import Counter

import pytest

from theo_conductor import evaluate, prlbench
from theo_conductor.models.registry import ModelRegistry
from theo_conductor.prlbench_comparison import parser, prepare, summarize
from theo_conductor.schema import ModelResponse, ModelSpec, ToolCallRecord
from theo_conductor.tools.openrouter_shell import OpenRouterShell


def paper(id='0'):
    return {'id': id, 'title': 'Research', 'introduction': 'Public context', 'subfield': 'amo',
            'tasks': [{'id': 1, 'prompt': 'Derive and compute'}, {'id': 2, 'prompt': 'Validate'}],
            'rubrics': [{'task_id': 2, 'rubrics': ['secret rubric 2'] * 3},
                        {'task_id': 1, 'rubrics': ['secret rubric 1'] * 7}]}


def test_paper_normalization_preserves_all_tasks_and_joins_rubrics_by_id():
    row = prlbench.normalize_papers([paper()])[0]
    assert row['id'] == 'prlbench-0'
    assert row['question'] == 'Research\n\nPublic context\n\nTask 1\nDerive and compute\n\nTask 2\nValidate'
    assert row['subtask_count'] == 2
    assert len(row['answer']) == 10
    assert row['answer'][0]['id'] == '1:1'
    assert row['answer'][-1]['id'] == '2:3'
    assert 'secret' not in row['question']
    with pytest.raises(ValueError, match='unique'):
        prlbench.normalize_papers([paper(), paper()])
    bad = paper()
    bad['rubrics'][0]['task_id'] = 3
    with pytest.raises(ValueError, match='alignment'):
        prlbench.normalize_papers([bad])
    for threshold in (0, 1.1, float('nan')):
        with pytest.raises(ValueError, match='threshold'):
            prlbench.normalize_papers([paper()], pass_threshold=threshold)


def test_criterion_verdicts_are_strict_and_score_all_criteria():
    criteria = prlbench.normalize_papers([paper()])[0]['answer']
    verdicts = [{'id': c['id'], 'correct': i < 7, 'reason': 'checked'} for i, c in enumerate(criteria)]
    score, details = prlbench.parse_verdict(json.dumps(verdicts), criteria)
    assert score == .7 and len(details) == 10
    for invalid in (verdicts[:-1], verdicts + [verdicts[0]],
                    [{**v, 'correct': 'true'} for v in verdicts],
                    [{**v, 'reason': ''} for v in verdicts],
                    [{**v, 'id': 'unknown'} for v in verdicts]):
        with pytest.raises(ValueError):
            prlbench.parse_verdict(json.dumps(invalid), criteria)


class ResearchClient:
    def __init__(self):
        self.base_url = 'https://openrouter.ai/api/v1'
        self.api_key = 'fake-key'
        self.model = 'fake/model'
        self.calls = Counter()
        self.bad_judge = False

    async def generate(self, **request):
        fmt = request.get('response_format') or {}
        kind = fmt.get('json_schema', {}).get('name')
        if kind == 'prlbench_criteria':
            self.calls['judge'] += 1
            assert 'execution_evidence' in request['question']
            assert 'computed 4' in request['question']
            assert 'secret rubric' in request['question']
            assert request['messages'][0]['role'] == 'system'
            ids = fmt['json_schema']['schema']['items']['properties']['id']['enum']
            if self.bad_judge:
                ids = ids[:-1]
            return ModelResponse(text=json.dumps([{'id': key, 'correct': i < 7, 'reason': 'evidence checked'}
                                                  for i, key in enumerate(ids)]), usage={'total_tokens': 5})
        assert 'secret rubric' not in json.dumps(request)
        if kind == 'conductor_workflow':
            self.calls['planner'] += 1
            assert 'openrouter:shell' in request['instruction']
            assert 'No external tools' not in request['instruction']
            return ModelResponse(text=json.dumps({'task_type': 'physics', 'difficulty': 'hard', 'workflow': [
                {'step_id': 'final', 'model_id': 'worker', 'instruction': 'Solve every task', 'access_list': ['question']}]}))
        self.calls['solver'] += 1
        return ModelResponse(text='All requested derivations and results. FINAL: 4', finish_reason='stop',
            usage={'total_tokens': 5, 'cost': .013, 'cost_details': {'server_tool_cost': .003, 'upstream_inference_cost': .01}},
            server_tool_calls=(ToolCallRecord(call_id='hosted-call', name='openrouter:shell', arguments={'commands': ['python3 -c print(4)']},
                model_id='worker', result={'output': [{'stdout': 'computed 4', 'stderr': '', 'outcome': {'type': 'exit', 'exit_code': 0}}],
                    'container_id': 'fixture', 'files': [], 'artifacts': [], 'plots_downloaded': True}),))



@pytest.mark.parametrize('mode', ['model', 'workflows'])
def test_four_independent_interpreter_attempts_judging_resume_and_judge_only(tmp_path, monkeypatch, mode):
    async def hosted(self, spec, request):
        assert 'secret rubric' not in json.dumps(request)
        return await spec.client.generate(**request)
    monkeypatch.setattr(OpenRouterShell, 'generate', hosted)
    client = ResearchClient()
    registry = ModelRegistry([ModelSpec(model_idx=key, client=client, supports_tools=True)
                              for key in ('worker', 'planner', 'judge')])
    config = tmp_path / 'config.yaml'
    config.write_text('models: []')
    source = tmp_path / 'questions.jsonl'
    source.write_text(json.dumps(prlbench.normalize_papers([paper()])[0]) + '\n')
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('- model_id="worker"\n## Available Worker Tools\nNo external tools\n## User Question\n<USER QUESTION>')
    argv = [mode, '--config', str(config), '--input', str(source), '--output-dir', str(tmp_path / 'run'),
            '--samples', '4', '--attempts', '1', '--judge-model', 'judge', '--openrouter-shell']
    argv += ['--model', 'worker'] if mode == 'model' else ['--planner-model', 'planner', '--prompt-file', str(prompt)]
    args = evaluate.parser().parse_args(argv)
    summary = asyncio.run(evaluate.run(args, registry))
    rows = [json.loads(line) for line in (tmp_path / 'run/results.jsonl').read_text().splitlines()]
    assert {r['sample_index'] for r in rows} == {0, 1, 2, 3}
    assert all(r['judge_score'] == .7 and r['judge_correct'] is True for r in rows)
    assert summary['repeated_sampling']['mean_at_k'] == 1
    assert summary['repeated_sampling']['pass_at_k'] == 1
    assert client.calls['solver'] == 4 and client.calls['judge'] == 4
    assert summary['stages']['solver' if mode == 'model' else 'worker']['total_tokens']['sum'] == 20
    assert summary['stages']['tool']['observations'] == 4
    costs = summary['stages']['solver' if mode == 'model' else 'worker']
    assert costs['reported_cost_usd']['sum'] == pytest.approx(.052)
    assert costs['server_tool_cost_usd']['sum'] == pytest.approx(.012)
    calls = client.calls.copy()
    asyncio.run(evaluate.run(args, registry))
    assert calls == client.calls
    # A behavior-affecting interpreter limit invalidates generations and grading.
    args.max_tokens = 240
    asyncio.run(evaluate.run(args, registry))
    assert client.calls['solver'] == 8 and client.calls['judge'] == 8
    judge_args = evaluate.parser().parse_args(['judge', '--config', str(config), '--output-dir', str(tmp_path / 'run'),
                                             '--judge-model', 'judge', '--attempts', '1', '--force'])
    original_hydrate = evaluate.hydrate_record
    async def failed_download(*args):
        raise RuntimeError('Files API temporarily unavailable')
    monkeypatch.setattr(evaluate, 'hydrate_record', failed_download)
    before = client.calls.copy()
    summary = asyncio.run(evaluate.run(judge_args, registry))
    assert client.calls == before
    assert summary['generation_failures'] == 0 and summary['judge_errors'] == 4
    monkeypatch.setattr(evaluate, 'hydrate_record', original_hydrate)
    client.bad_judge = True
    summary = asyncio.run(evaluate.run(judge_args, registry))
    assert client.calls['solver'] == 8
    assert summary['judge_errors'] == 4 and summary['repeated_sampling']['pass_at_k'] is None
    client.bad_judge = False
    summary = asyncio.run(evaluate.run(judge_args, registry))
    assert summary['judge_errors'] == 0


def test_judge_checks_png_evidence_hash_and_prevents_external_paths(tmp_path):
    data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l6kAAAAASUVORK5CYII=')
    root = tmp_path / 'artifacts'
    root.mkdir()
    (root / 'plot.png').write_bytes(data)
    artifact = {'path': 'artifacts/plot.png', 'name': 'plot.png', 'sha256': hashlib.sha256(data).hexdigest()}
    record = {'tool_calls': [{'name': 'python_execute', 'result': {'artifacts': [artifact]}}]}
    messages = prlbench.judge_messages(record, tmp_path, 'judge', 'question')
    assert messages[-1]['content'][-1]['image_url']['url'].startswith('data:image/png;base64,')
    artifact['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        prlbench.judge_messages(record, tmp_path, 'judge', 'question')
    artifact['path'] = '../outside.png'
    with pytest.raises(ValueError, match='outside'):
        prlbench.judge_messages(record, tmp_path, 'judge', 'question')


def test_comparison_summary_keeps_missing_slots_unknown_and_failures_in_denominator(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(paper()) + '\n')
    root = tmp_path / 'comparison'
    args = parser().parse_args(['prepare', '--source', str(source), '--output-dir', str(root), '--max-papers', '1'])
    manifest = prepare(args)
    assert manifest['samples'] == 4 and manifest['subset'] is True
    assert manifest['dataset_revision'] is None
    report = summarize(root)
    assert report['systems']['fugu']['pass_at_4'] is None
    assert report['systems']['fugu']['expected_slots'] == 4
    question = prlbench.normalize_papers([paper()])[0]
    rows = [{'example_id': question['id'], 'sample_index': i, 'question': question['question'], 'subject': 'amo',
             'gold_answer': question['answer'], 'answer_type': 'prlbench_criteria',
             'pass_threshold': .7, 'status': 'completed', 'judge_model': 'gemini-3.8-flash',
             'judge_scoring': prlbench.SCORING, 'judge_correct': i == 0, 'judge_score': .8 if i == 0 else .1}
            for i in range(4)]
    rows[3].update(status='failed', error='capped', judge_correct=None, judge_score=None)
    (root / 'fugu').mkdir()
    (root / 'fugu/manifest.json').write_text(json.dumps({
        'identity': {'mode': 'model', 'input_sha256': evaluate.fingerprint([question]),
                     'model': 'fugu-ultra-v2', 'planner': None}, 'samples': 4,
        'settings_history': [{'settings': {'config_sha256': manifest['files']['fugu.yaml']}}]}))
    result = root / 'fugu/results.jsonl'
    result.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    report = summarize(root)['systems']['fugu']
    assert report['mean_at_4'] == .25 and report['pass_at_4'] == 1
    assert report['mean_criterion_score_at_4_percent'] == 25
    result.write_text('\n'.join(json.dumps(r) for r in rows[:-1]) + '\n')
    assert summarize(root)['systems']['fugu']['pass_at_4'] is None
    result.write_text('\n'.join(json.dumps(r) for r in rows + [rows[0]]) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        summarize(root)
    (root / 'questions.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        summarize(root)
