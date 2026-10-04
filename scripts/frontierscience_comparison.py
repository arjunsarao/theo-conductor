"""Prepare a frozen FrontierScience physics comparison and report its two arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
import yaml

REPO = Path(__file__).resolve().parents[1]


def prepare(root: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    from unittest.mock import patch
    from datasets import Dataset
    from theo_conductor import data
    from theo_conductor.models.registry import ModelRegistry
    from theo_conductor.prompt import load_prompt_template
    import subprocess

    if root.exists():
        raise ValueError(f"Use a new comparison directory: {root}")
    load_dotenv(REPO / '.env')
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError('OPENROUTER_API_KEY is required')
    token = os.getenv('HF_TOKEN')
    revision = HfApi(token=token).dataset_info('openai/frontierscience').sha
    sources = {}
    for track in ('olympiad', 'research'):
        sources[track] = Path(hf_hub_download(
            'openai/frontierscience', f'{track}/test.jsonl', repo_type='dataset',
            revision=revision, token=token, cache_dir=REPO / 'outputs' / '.frontierscience-cache'))

    # Use the production normalizer with local files from one pinned revision.
    def pinned_load(name, **kwargs):
        track = kwargs['data_files']['test'].split('/')[0]
        return Dataset.from_list([json.loads(line) for line in sources[track].read_text().splitlines() if line.strip()])

    with patch.object(data, 'load_dataset', pinned_load):
        questions = list(data.load_frontierscience_physics_dataset())
    counts = dict(Counter(row['track'] for row in questions))
    if counts != {'olympiad': 50, 'research': 20}:
        raise ValueError(f'Unexpected physics cohort: {counts}')
    if len({row['id'] for row in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    root.mkdir(parents=True)
    (root / 'questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in questions))
    frontier = yaml.safe_load((REPO / 'configs/worker_pool_frontier.yaml').read_text())
    direct = yaml.safe_load((REPO / 'configs/fugu_ultra_v2_direct.yaml').read_text())
    judge = next(model for model in frontier['models'] if model['model_idx'] == 'gemini-3.7-flash')
    direct['models'].append(judge)
    for name, config in [('conductor', frontier), ('fugu', direct)]:
        (root / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ('conductor-prompt-refined.md', 'worker-prompt.txt'):
        shutil.copy2(REPO / name, root / name)
    template, model_ids = load_prompt_template(root / 'conductor-prompt-refined.md')
    if template.count('<USER QUESTION>') != 1:
        raise ValueError('Expected exactly one question marker')
    registry = ModelRegistry.from_yaml_file(root / 'conductor.yaml')
    for model_id in [*model_ids, frontier['conductor_model'], judge['model_idx']]:
        registry.get(model_id)
    ModelRegistry.from_yaml_file(root / 'fugu.yaml').get('fugu-ultra-v2')
    shutil.copytree(REPO / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.orig'))
    shutil.copy2(Path(__file__), root / 'comparison.py')
    shutil.copy2(REPO / 'scripts/benchmark_frontierscience_comparison.sbatch', root / 'launch.sbatch')
    for track, path in sources.items():
        shutil.copy2(path, root / f'{track}-source.jsonl')
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob('*') if path.is_file()}
    duplicate_groups = {}
    for row in questions:
        duplicate_groups.setdefault((row['track'], row['question']), []).append(row['id'])
    metadata = {
        'dataset': 'openai/frontierscience', 'selector': 'frontierscience-physics',
        'revision': revision, 'physics_questions': len(questions), 'track_counts': counts,
        'distinct_question_texts': len(duplicate_groups),
        'duplicate_question_groups': [ids for ids in duplicate_groups.values() if len(ids) > 1],
        'samples': 4, 'expected_slots_per_arm': len(questions) * 4,
        'planner': frontier['conductor_model'], 'judge': judge['model_idx'], 'solver': 'fugu-ultra-v2',
        'temperature': 0.2, 'plan_temperature': 0.1, 'plan_max_tokens': 8192,
        'judge_max_tokens': 8192, 'recovery_attempts_per_slot': 3, 'concurrency_per_arm': 4,
        'output_limits': 'Per-model max_output_tokens from frozen registries',
        'failure_policy': 'Exhausted solver/workflow failures count false; planner/judge errors stay unknown',
        'grading': 'Olympiad semantic short answer; Research rubric10 with score >= 7 correct',
        'duplicate_policy': 'Retain every original source row; disclose duplicates in reported comparisons',
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'file_sha256': hashes,
    }
    (root / 'comparison-manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: value for key, value in metadata.items() if key != 'file_sha256'}, indent=2))


def summarize(root: Path) -> None:
    manifest = json.loads((root / 'comparison-manifest.json').read_text())
    report = {'physics_questions': manifest['physics_questions'], 'samples_per_question': 4, 'arms': {}}
    for arm in ('conductor', 'fugu'):
        path = root / arm / 'summary.json'
        if not path.exists():
            report['arms'][arm] = {'status': 'pending'}
            continue
        summary = json.loads(path.read_text())
        metrics = summary['repeated_sampling']['by_k']['4']
        report['arms'][arm] = {'mean_at_4': metrics['mean_at_k'], 'pass_at_4': metrics['pass_at_k'],
                               'generation_failures': summary['generation_failures'],
                               'judge_errors': summary['judge_errors'],
                               'repeated_sampling': summary['repeated_sampling'],
                               'all_attempts': summary['all_attempts']}
    target = root / 'comparison-summary.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'summarize'])
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'summarize': summarize}[args.command](args.output_root.resolve())
