"""Native hosted shell requests and recoverable, local PNG evidence.

OpenRouter executes the entire tool loop. No model command is executed locally.
Artifact retrieval happens after generation is checkpointed, so a Files API
failure can be recovered by judge-only execution without another solver call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from ..models.openai_compat import build_message
from ..schema import ModelResponse, ToolCallRecord

DESCRIPTION = """Workers have OpenRouter's hosted openrouter:shell tool. Use shell
commands, Python 3 (python3), NumPy, pandas and Matplotlib for calculations.
There is no internet access. Do not assume SciPy, SymPy, mpmath, QuTiP or NetworkX
are installed. Commands in one worker request share a sandbox; each worker and
recovery request starts with a fresh isolated sandbox. No files pass between
workers: pass numerical results through permitted predecessor text.
Print important results and save requested plots as PNG under ~/outputs/.
The judge receives actual execution transcripts and downloaded PNG plots.
Provider command defaults are 120 seconds and 16,384 characters per output
stream; provider caps are 300 seconds and 65,536 characters. The provider
controls the internal tool loop; local --max-tool-rounds does not limit it.
"""
MAX_PLOT_BYTES = 32 * 1024 * 1024


def rejected_before_execution(item: dict) -> bool:
    """Recognize provider argument rejection, never a failed executed command."""
    if (item.get('status') != 'failed' or item.get('container_id') or
            item.get('output') or item.get('files')):
        return False
    error = item.get('error')
    if not isinstance(error, str):
        return False
    if error.startswith('Failed to parse tool call arguments') and 'invalid JSON' in error:
        return True
    try:
        issues = json.loads(error)
    except (ValueError, TypeError):
        return False
    return bool(isinstance(issues, list) and issues and all(
        isinstance(issue, dict) and issue.get('code') in {'invalid_type', 'too_big'}
        and isinstance(issue.get('path'), list) and issue['path'][:1] == ['commands']
        for issue in issues))


def validate_client(client) -> None:
    url = urlsplit(getattr(client, 'base_url', ''))
    if url.scheme != 'https' or url.netloc != 'openrouter.ai' or url.path.rstrip('/') != '/api/v1':
        raise ValueError('Hosted shell requires https://openrouter.ai/api/v1 (global endpoint)')
    if not getattr(client, 'api_key', None) or not getattr(client, 'model', None):
        raise ValueError('Hosted shell requires an authenticated OpenRouter model client')


def input_messages(messages: list[dict]) -> list[dict]:
    """Translate the existing worker message contract to Responses input."""
    result = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            content = [{'type': 'input_text', 'text': content}]
        else:
            parts = []
            for part in content:
                if part['type'] == 'text':
                    parts.append({'type': 'input_text', 'text': part['text']})
                elif part['type'] == 'image_url':
                    image = part['image_url']
                    parts.append({'type': 'input_image', 'image_url': image['url'],
                                  **({'detail': image['detail']} if 'detail' in image else {})})
                else:
                    raise ValueError(f'Unsupported hosted-shell input content: {part["type"]}')
            content = parts
        result.append({'role': message['role'], 'content': content})
    return result


def response_from_body(body: dict, *, model_id: str, latency_ms: float) -> ModelResponse:
    messages, calls = [], []
    for item in body.get('output') or []:
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            messages.append(''.join(part.get('text', '') for part in item.get('content', [])
                                    if part.get('type') == 'output_text'))
        elif item.get('type') == 'openrouter:shell':
            outputs = item.get('output') or []
            calls.append(ToolCallRecord(
                call_id=str(item.get('call_id') or item.get('id')),
                name='openrouter:shell', arguments=item.get('action') or {}, model_id=model_id,
                result={'output': outputs, 'container_id': item.get('container_id'),
                        'files': item.get('files') or [], 'artifacts': [],
                        'status': item.get('status'), 'error': item.get('error')},
                is_error=item.get('status') != 'completed' or any(
                    output.get('outcome', {}).get('type') != 'exit' or
                    output.get('outcome', {}).get('exit_code') != 0 for output in outputs)))
    status = body.get('status')
    reason = (body.get('incomplete_details') or {}).get('reason')
    finish = 'stop' if status == 'completed' else 'max_tokens' if reason == 'max_output_tokens' else 'error'
    # A pre-tool assistant preamble must not become a successful final response.
    final = messages[-1] if messages and (body.get('output') or [])[-1].get('type') == 'message' else ''
    return ModelResponse(text=final, raw=body, usage=body.get('usage'), latency_ms=latency_ms,
                         finish_reason=finish, server_tool_calls=tuple(calls))


class OpenRouterShell:
    def build_request(self, spec, request: dict) -> dict:
        validate_client(spec.client)
        if not spec.supports_tools:
            raise ValueError(f'Hosted shell requires supports_tools: {spec.model_idx}')
        messages = request.get('messages') or build_message(
            instruction=request['instruction'], question=request['question'], context=request['context'])
        messages = [*messages, {'role': 'user', 'content': DESCRIPTION}]
        return {'model': spec.client.model, 'input': input_messages(messages),
                'max_output_tokens': request.get('max_tokens') or 2048,
                'temperature': request.get('temperature', .2),
                'tools': [{'type': 'openrouter:shell', 'parameters': {
                    'engine': 'openrouter', 'environment': {
                        'type': 'container_reference', 'container_id': 'prl_' + uuid.uuid4().hex,
                        'network_policy': {'type': 'disabled'}}}}]}

    async def generate(self, spec, request: dict) -> ModelResponse:
        payload = self.build_request(spec, request)
        start = time.perf_counter()
        # No automatic POST retry: interrupted billing is unknown and recovery
        # belongs to the evaluator's declared per-slot attempts policy.
        async with httpx.AsyncClient(timeout=spec.client.timeout,
                headers={'Authorization': f'Bearer {spec.client.api_key}'}) as client:
            response = await client.post(spec.client.base_url + '/responses', json=payload)
            response.raise_for_status()
            body = response.json()
        return response_from_body(body, model_id=str(spec.model_idx or spec.name),
                                  latency_ms=(time.perf_counter()-start)*1000)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


async def hydrate_record(record: dict, registry, root: Path, checkpoint) -> None:
    """Download final PNGs, including files omitted from the 10-file citations.

Persist inventory and each completed download. Remote failures remain judge
errors; successful generation and successful downloads are retained on retry.
"""
    calls = ([call for output in record['worker_outputs'].values() for call in output.get('tool_calls', [])]
             if record.get('worker_outputs') else record.get('tool_calls', []))
    by_container = {}
    # Older checkpoints retain rejection details only in raw request history.
    rejected_calls = set()
    for attempt in record.get('attempt_history', []):
        for item in (attempt.get('raw_response') or {}).get('output', []):
            if item.get('type') == 'openrouter:shell' and rejected_before_execution(item):
                rejected_calls.add(str(item.get('call_id') or item.get('id')))
    for call in calls:
        if call.get('name') == 'openrouter:shell':
            result = call.get('result') or {}
            container = result.get('container_id')
            if not isinstance(container, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', container):
                if (container is None and call.get('is_error') and not result.get('output')
                        and not result.get('files') and not result.get('artifacts')
                        and (rejected_before_execution(result) or call.get('call_id') in rejected_calls)):
                    continue
                raise ValueError('Hosted shell result has no valid container ID')
            by_container[container] = call
    for container, call in by_container.items():
        result = call['result']
        if result.get('plots_downloaded'):
            continue
        source = registry.get(call['model_id']).client
        validate_client(source)
        base = source.base_url + '/containers/' + container + '/files'
        async with httpx.AsyncClient(timeout=source.timeout,
                headers={'Authorization': f'Bearer {source.api_key}'}) as client:
            if 'plot_inventory' not in result:
                inventory, cursor, cursors = [], None, set()
                while True:
                    page = await client.get(base, params={'limit': 1000, **({'after': cursor} if cursor else {})})
                    page.raise_for_status()
                    data = page.json()
                    inventory.extend(item for item in data.get('data', []) if item.get('path', '').lower().endswith('.png'))
                    if not data.get('has_more'):
                        break
                    cursor = data.get('last_id')
                    if not cursor or cursor in cursors or len(cursors) >= 1000:
                        raise ValueError('Invalid hosted shell file pagination')
                    cursors.add(cursor)
                result['plot_inventory'] = inventory
                checkpoint()
            for item in result['plot_inventory']:
                if any(artifact.get('file_id') == item['id'] for artifact in result['artifacts']):
                    continue
                content = bytearray()
                async with client.stream('GET', base + '/' + quote(item['id'], safe='') + '/content') as downloaded:
                    downloaded.raise_for_status()
                    async for chunk in downloaded.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_PLOT_BYTES:
                            raise ValueError('Hosted shell PNG exceeds 32 MiB evidence limit')
                if not content.startswith(b'\x89PNG\r\n\x1a\n'):
                    raise ValueError('Hosted shell PNG has invalid content')
                digest = hashlib.sha256(content).hexdigest()
                path = Path('artifacts') / (digest + '.png')
                _atomic_bytes(root / path, content)
                result['artifacts'].append({'name': item['path'], 'path': str(path),
                                           'sha256': digest, 'file_id': item['id']})
                checkpoint()
            result['plots_downloaded'] = True
            checkpoint()
