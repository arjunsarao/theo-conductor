"""Recover the ten historically capped Fugu slots, preserving source results."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from theo_conductor.diamond_benchmark import generate_answers, judge_answers, new_record
from theo_conductor.jsonio import read_jsonl, write_json_atomic, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[1]

async def run(args):
    questions = {r['id']: r for r in read_jsonl(ROOT / 'outputs/hle-diamond-physics/questions.jsonl')}
    groups = []
    slots = []
    for cohort, model in [('hle-diamond-fugu-max-mean4-22753', 'sakana/fugu-max'), ('hle-diamond-fugu-ultra-v2-mean4', 'sakana/fugu-ultra-v2')]:
        for repeat in range(1, 5):
            source = ROOT / 'outputs' / cohort / f'run-{repeat}' / 'results.jsonl'
            capped = [r for r in read_jsonl(source) if r.get('finish_reason') in {'length', 'max_tokens'}]
            if not capped:
                continue
            target = args.output_dir / cohort / f'run-{repeat}'
            groups.append((model, source, target, capped))
            for r in capped:
                assert r['question'] == questions[r['example_id']]['question']
                slots.append({'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'example_id': r['example_id'], 'repeat': repeat, 'model': model, 'original_finish_reason': r['finish_reason'], 'original_max_tokens': 32768, 'max_tokens': 65536})
    assert len(slots) == 10, f'Expected ten capped slots, found {len(slots)}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'policy': 'All saved capped Fugu responses; original slots retained; no selective correctness filtering', 'budget_provenance': '32768 from historical launcher defaults; original per-request budgets were not persisted; usage includes internal work and is not the caller budget', 'judge_model': 'google/gemini-3.8-flash', 'temperature': 0.0, 'slots': slots, 'worker_prompt_sha256': hashlib.sha256((ROOT / 'worker-prompt.txt').read_bytes()).hexdigest()}
    manifest_path = args.output_dir / 'recovery-manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Recovery manifest changed')
    write_json_atomic(manifest_path, manifest)
    print(f'Prepared {len(slots)} slots with max_tokens=65536', flush=True)
    if args.prepare_only:
        return
    api_key = os.environ['OPENROUTER_API_KEY']
    semaphore = asyncio.Semaphore(2)
    async def recover(group):
        model, source, target, capped = group
        async with semaphore:
            target.mkdir(parents=True, exist_ok=True)
            path = target / 'results.jsonl'
            rows = [questions[r['example_id']] for r in capped]
            records = read_jsonl(path) if path.exists() else [new_record(row, i, solver_model=model, solver_name=model) for i, row in enumerate(rows)]
            write_jsonl_atomic(target / 'original-capped-results.jsonl', capped)
            write_jsonl_atomic(target / 'questions.jsonl', rows)
            request_args = argparse.Namespace(solver_model=model, solver_name=model, solver_standard=True, solver_timeout_seconds=7200, solver_concurrency=2, max_tokens=65536, attempts=3, poll_seconds=30, judge_mode='sync', judge_concurrency=2, judge_max_tokens=8192)
            await generate_answers(args=request_args, rows=rows, records=records, results_path=path, api_key=api_key)
            await judge_answers(args=request_args, records=records, results_path=path, api_key=api_key)
            write_json_atomic(target / 'recovery-summary.json', {'attempts': len(records), 'generation_errors': sum(bool(r.get('error')) for r in records), 'still_capped': sum(r.get('finish_reason') in {'length','max_tokens'} for r in records), 'judge_errors': sum(bool(r.get('judge_error')) for r in records), 'correct': sum(r.get('judge_correct') is True for r in records)})
    await asyncio.gather(*(recover(g) for g in groups))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    asyncio.run(run(parser.parse_args()))
