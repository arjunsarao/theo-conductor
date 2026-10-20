import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from theo_conductor.schema import ModelSpec
from theo_conductor.tools import openrouter_shell as shell


def spec(base_url='https://openrouter.ai/api/v1'):
    return ModelSpec(model_idx='worker', supports_tools=True, client=SimpleNamespace(
        base_url=base_url, api_key='fixture-key', model='fixture/model', timeout=httpx.Timeout(30)))


def body(status='completed', reason=None):
    return {'object': 'response', 'status': status, 'incomplete_details': {'reason': reason},
            'usage': {'input_tokens': 10, 'output_tokens': 5, 'cost': .013,
                      'cost_details': {'server_tool_cost': .003, 'upstream_inference_cost': .01}},
            'output': [
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Running code'}]},
                {'type': 'openrouter:shell', 'status': 'completed', 'call_id': 'call-1', 'container_id': 'fixture',
                 'action': {'commands': ['python3 -c "print(4)"'], 'timeout_ms': 120000},
                 'output': [{'stdout': '4\n', 'stderr': '', 'outcome': {'type': 'exit', 'exit_code': 0}}],
                 'files': []},
                {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'FINAL: 4'}]}]}


def test_native_request_uses_responses_preserves_input_and_isolates_every_request(monkeypatch):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        assert str(request.url) == 'https://openrouter.ai/api/v1/responses'
        assert request.headers['Authorization'] == 'Bearer fixture-key'
        return httpx.Response(200, json=body())

    client = httpx.AsyncClient
    monkeypatch.setattr(shell.httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    args = {'instruction': 'Solve', 'question': 'public question', 'context': {'earlier': 'correction'},
            'max_tokens': 4096, 'temperature': .3}

    async def run():
        executor = shell.OpenRouterShell()
        for _ in range(4):
            output = await executor.generate(spec(), args)
            assert output.text == 'FINAL: 4' and output.finish_reason == 'stop'
            assert output.raw == body() and output.usage['cost'] == .013
            assert not output.tool_calls  # Already executed server-side.
            call = output.server_tool_calls[0]
            assert call.name == 'openrouter:shell' and not call.is_error
            assert call.result['output'][0]['stdout'] == '4\n'
            assert call.model_id == 'worker'
    asyncio.run(run())
    environments = [r['tools'][0]['parameters']['environment'] for r in requests]
    assert len({e['container_id'] for e in environments}) == 4
    assert all(e['network_policy'] == {'type': 'disabled'} for e in environments)
    assert all(r['tools'][0]['parameters']['engine'] == 'openrouter' for r in requests)
    assert requests[0]['max_output_tokens'] == 4096 and requests[0]['temperature'] == .3
    assert 'public question' in json.dumps(requests[0]['input'])
    assert 'correction' in json.dumps(requests[0]['input'])
    for url in ('https://eu.openrouter.ai/api/v1', 'http://openrouter.ai/api/v1', 'https://example.com/api/v1'):
        with pytest.raises(ValueError, match='global endpoint'):
            shell.OpenRouterShell().build_request(spec(url), args)


@pytest.mark.parametrize(('status', 'reason', 'finish'), [
    ('incomplete', 'max_output_tokens', 'max_tokens'), ('failed', None, 'error'), ('completed', None, 'stop')])
def test_status_and_failed_commands_preserve_evidence(status, reason, finish):
    response = body(status, reason)
    response['output'][1]['output'][0]['outcome'] = {'type': 'timeout'}
    parsed = shell.response_from_body(response, model_id='worker', latency_ms=12)
    assert parsed.finish_reason == finish and parsed.server_tool_calls[0].is_error
    response['output'].pop()
    assert shell.response_from_body(response, model_id='worker', latency_ms=12).text == ''


def test_image_input_conversion_preserves_worker_image():
    converted = shell.input_messages([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Solve'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}])
    assert converted[0]['content'][1] == {'type': 'input_image', 'image_url': 'data:image/png;base64,abc'}


def test_plot_download_retry_keeps_generation_and_previous_downloads_and_pages_empty_results(tmp_path, monkeypatch):
    png = b'\x89PNG\r\n\x1a\nfixture'
    fail = True
    downloads, pages, snapshots = [], [], []
    record = {'tool_calls': [shell.response_from_body(body(), model_id='worker', latency_ms=1).server_tool_calls[0].model_dump()]}

    def handle(request):
        if request.url.path.endswith('/files'):
            pages.append(request.url.params.get('after'))
            if 'after' not in request.url.params:
                return httpx.Response(200, json={'data': [], 'has_more': True, 'last_id': 'empty-page-cursor'})
            # The second plot was omitted from tool citations; listing discovers it.
            return httpx.Response(200, json={'data': [
                {'id': 'cfile_first', 'path': 'outputs/first.png'}, {'id': 'cfile_second', 'path': 'outputs/second.png'}],
                'has_more': False})
        downloads.append(request.url.path)
        if fail and 'cfile_second' in request.url.path:
            return httpx.Response(503)
        return httpx.Response(200, content=png)

    client = httpx.AsyncClient
    monkeypatch.setattr(shell.httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    registry = SimpleNamespace(get=lambda key: spec())
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(shell.hydrate_record(record, registry, tmp_path, lambda: snapshots.append(json.dumps(record))))
    result = record['tool_calls'][0]['result']
    assert len(result['artifacts']) == 1 and not result.get('plots_downloaded')
    fail = False
    asyncio.run(shell.hydrate_record(record, registry, tmp_path, lambda: snapshots.append(json.dumps(record))))
    assert result['plots_downloaded'] and len(result['artifacts']) == 2
    assert pages == [None, 'empty-page-cursor']
    assert sum('cfile_first' in d for d in downloads) == 1
    assert sum('cfile_second' in d for d in downloads) == 2
    assert all((tmp_path / artifact['path']).read_bytes() == png for artifact in result['artifacts'])
    prior = len(downloads)
    asyncio.run(shell.hydrate_record(record, registry, tmp_path, lambda: None))
    assert len(downloads) == prior


def test_native_transport_does_not_retry_paid_posts(monkeypatch):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(503)
    client = httpx.AsyncClient
    monkeypatch.setattr(shell.httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(shell.OpenRouterShell().generate(spec(), {'instruction': 'Solve', 'question': 'Q', 'context': {}}))
    assert len(calls) == 1


@pytest.mark.parametrize('legacy', [False, True])
def test_rejected_arguments_do_not_block_valid_plot_download(tmp_path, monkeypatch, legacy):
    rejection = {'type': 'openrouter:shell', 'status': 'failed', 'call_id': 'bad',
                 'action': {'commands': []}, 'error': json.dumps([
                     {'code': 'invalid_type', 'path': ['commands'], 'message': 'expected array'}])}
    response = body()
    response['output'].insert(1, rejection)
    parsed = shell.response_from_body(response, model_id='worker', latency_ms=1)
    calls = [c.model_dump() for c in parsed.server_tool_calls]
    record = {'worker_outputs': {'step': {'tool_calls': calls}},
              'attempt_history': [{'raw_response': response}] if legacy else []}
    if legacy:
        calls[0]['result'].pop('status')
        calls[0]['result'].pop('error')
    original = json.dumps(calls[0])
    requests = []
    png = b'\x89PNG\r\n\x1a\nfixture'

    def handle(request):
        requests.append(request.url.path)
        if request.url.path.endswith('/files'):
            return httpx.Response(200, json={'data': [{'id': 'plot', 'path': 'outputs/result.png'}], 'has_more': False})
        return httpx.Response(200, content=png)

    client = httpx.AsyncClient
    monkeypatch.setattr(shell.httpx, 'AsyncClient', lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    asyncio.run(shell.hydrate_record(record, SimpleNamespace(get=lambda key: spec()), tmp_path, lambda: None))
    assert json.dumps(calls[0]) == original  # Rejected call evidence survives.
    assert calls[1]['result']['plots_downloaded']
    assert len(calls[1]['result']['artifacts']) == 1
    assert len(requests) == 2 and all('/containers/fixture/' in p for p in requests)


@pytest.mark.parametrize('change', [
    {}, {'status': 'completed'}, {'output': [{'stdout': 'executed'}]},
    {'files': [{'id': 'plot'}]}, {'container_id': 'invalid/id'},
])
def test_unexplained_or_executed_missing_container_still_blocks_grading(tmp_path, change):
    result = {'output': [], 'files': [], 'artifacts': [], 'container_id': None}
    if change:
        result.update(status='failed', error='Failed to parse tool call arguments: invalid JSON')
        result.update(change)
    record = {'tool_calls': [{'name': 'openrouter:shell', 'is_error': True, 'result': result}]}
    with pytest.raises(ValueError, match='container ID'):
        asyncio.run(shell.hydrate_record(record, None, tmp_path, lambda: None))


@pytest.mark.parametrize('error', [
    'Failed to parse tool call arguments for "openrouter_shell": invalid JSON',
    json.dumps([{'code': 'too_big', 'path': ['commands', 0]}]),
])
def test_recognize_preexecution_argument_rejection(error):
    assert shell.rejected_before_execution({'status': 'failed', 'error': error})
