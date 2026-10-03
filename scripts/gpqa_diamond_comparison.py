"""Prepare a frozen GPQA Diamond physics comparison and report its two arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
import yaml

REPO = Path(__file__).resolve().parents[1]


def prepare(root: Path, seed: int = 42) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    if root.exists():
        raise ValueError(f"Use a new comparison directory: {root}")
    load_dotenv(REPO / '.env')
    token = os.getenv('HF_TOKEN')
    revision = HfApi(token=token).dataset_info('Idavidrein/gpqa').sha
    source = Path(hf_hub_download('Idavidrein/gpqa', 'gpqa_diamond.csv',
                                repo_type='dataset', revision=revision, token=token,
                                cache_dir=REPO / 'outputs' / '.gpqa-cache'))
    with source.open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 198:
        raise ValueError(f"Expected 198 Diamond rows, got {len(all_rows)}")
    physics = [row for row in all_rows if row['High-level domain'].strip().lower() == 'physics']
    if not physics:
        raise ValueError('No physics rows found')
    rng = random.Random(seed)
    questions = []
    for row in physics:
        options = [(True, row['Correct Answer'])] + [
            (False, row[f'Incorrect Answer {i}']) for i in range(1, 4)]
        rng.shuffle(options)
        answer = next(label for label, (correct, _) in zip('ABCD', options) if correct)
        question = row['Question'] + '\n\n' + '\n'.join(
            f'{label}. {text}' for label, (_, text) in zip('ABCD', options))
        question += '\n\nChoose exactly one option. End your answer with FINAL: <A, B, C, or D>.'
        questions.append({'id': f"gpqa-diamond-{row['Record ID']}", 'question': question,
                          'answer': answer, 'subject': 'Physics', 'answer_type': 'multipleChoice'})
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
    shutil.copytree(REPO / 'src', root / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.orig'))
    shutil.copy2(Path(__file__), root / 'comparison.py')
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob('*') if path.is_file()}
    metadata = {'dataset': 'Idavidrein/gpqa', 'config': 'gpqa_diamond', 'revision': revision,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'diamond_questions': len(all_rows), 'physics_questions': len(questions),
                'option_seed': seed, 'gold_option_distribution': dict(Counter(q['answer'] for q in questions)),
                'samples': 4, 'planner': frontier['conductor_model'], 'judge': judge['model_idx'],
                'solver': 'fugu-ultra-v2', 'file_sha256': hashes}
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
