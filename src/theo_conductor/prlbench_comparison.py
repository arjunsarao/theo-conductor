"""Prepare, run, and summarize a paired four-sample PRLBench comparison."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import subprocess
from pathlib import Path
from statistics import mean

import yaml
from dotenv import load_dotenv

from . import evaluate, prlbench
from .evaluation_stats import repeated_metrics, summarize_evaluation
from .jsonio import read_jsonl, write_json_atomic, write_jsonl_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_ID = 'gemini-3.8-flash'
JUDGE_SPEC = {
    'model_idx': JUDGE_ID, 'provider': 'openrouter', 'name': JUDGE_ID,
    'display_name': 'Gemini 3.8 Flash', 'context_length': 1048576,
    'max_output_tokens': 32768, 'supports_json': True, 'supports_tools': False,
    'cost_per_1m_input_tokens': .75, 'cost_per_1m_output_tokens': 3.75,
    'role': 'PRLBench criterion judge', 'client': {
        'type': 'openai_compatible', 'base_url': 'https://openrouter.ai/api/v1',
        'model': 'google/gemini-3.8-flash', 'timeout_seconds': 7200}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError('Comparison already prepared; reuse it with run or choose a new directory')
    if args.source:
        source = args.source
        revision = None
    else:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(repo_id=prlbench.DATASET_ID, repo_type='dataset',
                    filename='prl_bench.jsonl', revision=args.dataset_revision))
        revision = args.dataset_revision
    papers = read_jsonl(source)
    if len(papers) != 100 and not args.max_papers:
        raise ValueError('Full PRLBench comparison expects 100 papers; use --max-papers for a declared smoke subset')
    if args.max_papers is not None:
        if args.max_papers <= 0 or args.max_papers > len(papers):
            raise ValueError('max-papers must be positive and no larger than the source cohort')
        papers = papers[:args.max_papers]
    rows = prlbench.normalize_papers(papers, pass_threshold=args.pass_threshold)
    root.mkdir(parents=True, exist_ok=True)
    # Freeze dataset, registries and planner prompt for a reviewable run.
    (root / 'source.jsonl').write_bytes(source.read_bytes())
    write_jsonl_atomic(root / 'questions.jsonl', rows)
    for name, config in [('conductor', args.conductor_config), ('fugu', args.fugu_config)]:
        data = yaml.safe_load(config.read_text())
        if any(spec['model_idx'] == JUDGE_ID for spec in data['models']):
            raise ValueError('Source registry already defines the comparison judge')
        data['models'].append(JUDGE_SPEC)
        (root / f'{name}.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    (root / 'planner.md').write_bytes(args.prompt_file.read_bytes())
    # Validate registry IDs and the full worker catalogue without inference.
    conductor = evaluate.ModelRegistry.from_yaml_file(root / 'conductor.yaml')
    fugu = evaluate.ModelRegistry.from_yaml_file(root / 'fugu.yaml')
    conductor.get(args.planner_model)
    fugu.get('fugu-ultra-v2')
    _, ids = evaluate.load_prompt_template(root / 'planner.md')
    for key in ids:
        conductor.get(key)
    manifest = {
        'dataset': prlbench.DATASET_ID, 'dataset_revision': revision, 'source_sha256': sha(source),
        'source_kind': 'local' if args.source else 'huggingface',
        'samples': 4, 'unit': 'paper (all subtasks together)', 'papers': len(rows),
        'subtasks': sum(row['subtask_count'] for row in rows), 'subset': len(rows) != 100,
        'pass_threshold': args.pass_threshold, 'scoring': prlbench.SCORING,
        'weighting': 'Equal weight per public criterion within each paper; macro average over papers',
        'planner_model': args.planner_model, 'solver_model': 'fugu-ultra-v2', 'judge_model': JUDGE_ID,
        'interpreter': {'backend': 'openrouter:shell', 'engine': 'openrouter',
                        'network_policy': 'disabled', 'isolation': 'fresh container per worker request',
                        'packages': 'Provider-managed Python, NumPy, pandas, Matplotlib'},
        'created_at': evaluate.now(), 'files': {name: sha(root / name) for name in (
            'source.jsonl', 'questions.jsonl', 'conductor.yaml', 'fugu.yaml', 'planner.md')},
        'code_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                       capture_output=True, text=True).stdout.strip() or None,
        'implementation_sha256': {name: sha(REPO_ROOT / name) for name in (
            'src/theo_conductor/prlbench.py', 'src/theo_conductor/prlbench_comparison.py',
            'src/theo_conductor/evaluate.py', 'src/theo_conductor/runner.py',
            'src/theo_conductor/evaluation_stats.py',
            'src/theo_conductor/benchmark.py', 'src/theo_conductor/models/openai_compat.py',
            'src/theo_conductor/tools/openrouter_shell.py', 'src/theo_conductor/schema.py', 'worker-prompt.txt')},
        'limitations': ['Public criterion weights are unavailable; this is an adapted score.',
                        'pass@4 is an oracle metric; systems have different model/compute budgets.',
                        'Four samples and Gemini 3.8 judging differ from the paper protocol.']}
    write_json_atomic(root / 'comparison-manifest.json', manifest)
    return manifest


def load_manifest(root: Path, *, verify_code: bool = False) -> dict:
    manifest = json.loads((root / 'comparison-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Prepared comparison file changed: {name}; prepare a new comparison')
    if verify_code:
        for name, expected in manifest.get('implementation_sha256', {}).items():
            if sha(REPO_ROOT / name) != expected:
                raise ValueError(f'Prepared comparison implementation changed: {name}; prepare a new comparison')
    return manifest


async def run_comparison(args: argparse.Namespace) -> dict:
    root = args.output_dir.resolve()
    manifest = load_manifest(root, verify_code=True)
    common = ['--input', str(root / 'questions.jsonl'), '--samples', '4',
              '--concurrency', str(args.concurrency), '--attempts', str(args.attempts),
              '--temperature', str(args.temperature), '--judge-model', JUDGE_ID,
              '--judge-max-tokens', '32768', '--openrouter-shell']
    if args.max_tokens:
        common += ['--max-tokens', str(args.max_tokens)]
    commands = [
        ['workflows', '--config', str(root / 'conductor.yaml'), '--output-dir', str(root / 'conductor'),
         '--prompt-file', str(root / 'planner.md'), '--planner-model', manifest['planner_model'],
         '--plan-temperature', str(args.plan_temperature), *common],
        ['model', '--config', str(root / 'fugu.yaml'), '--output-dir', str(root / 'fugu'),
         '--model', manifest['solver_model'], *common]]
    # Independent system runs; a failure in one preserves completed sibling work.
    results = await asyncio.gather(*(evaluate.run(evaluate.parser().parse_args(cmd)) for cmd in commands),
                                   return_exceptions=True)
    errors = [f'{type(r).__name__}: {r}' for r in results if isinstance(r, BaseException)]
    report = summarize(root)
    if errors:
        raise ValueError('; '.join(errors))
    return report


def summarize(root: Path) -> dict:
    manifest = load_manifest(root)
    inputs = read_jsonl(root / 'questions.jsonl')
    expected = {row['id']: row for row in inputs}
    report = {'comparison': manifest, 'systems': {}}
    paired_settings = None
    for name in ('conductor', 'fugu'):
        path = root / name / 'results.jsonl'
        rows = read_jsonl(path) if path.exists() else []
        run_manifest_path = root / name / 'manifest.json'
        if rows:
            if not run_manifest_path.exists():
                raise ValueError(f'{name}: results have no managed run manifest')
            run_manifest = json.loads(run_manifest_path.read_text())
            expected_identity = {'mode': 'workflows' if name == 'conductor' else 'model',
                'input_sha256': evaluate.fingerprint(inputs),
                'model': 'theo-conductor' if name == 'conductor' else manifest['solver_model'],
                'planner': manifest['planner_model'] if name == 'conductor' else None}
            if run_manifest['identity'] != expected_identity or run_manifest['samples'] != 4:
                raise ValueError(f'{name}: incompatible managed run identity/sample count')
            settings = run_manifest['settings_history'][-1]['settings']
            if settings['config_sha256'] != manifest['files'][f'{name}.yaml']:
                raise ValueError(f'{name}: registry differs from prepared comparison')
            comparable = {key: settings.get(key) for key in (
                'temperature', 'max_tokens', 'openrouter_shell', 'max_tool_rounds', 'shell_runtime_sha256', 'judge_model', 'judge_max_tokens')}
            if paired_settings is not None and paired_settings != comparable:
                raise ValueError('Systems have different interpreter, generation or judge settings')
            paired_settings = comparable
        slots = {}
        for row in rows:
            key = (row['example_id'], row['sample_index'])
            if key in slots or key[0] not in expected or key[1] not in range(4):
                raise ValueError(f'{name}: unexpected or duplicate independent sample slot')
            source = expected[key[0]]
            if (row['question'] != source['question'] or row.get('gold_answer') != source['answer'] or
                row.get('subject') != source['subject'] or row.get('answer_type') != 'prlbench_criteria' or
                row.get('pass_threshold') != manifest['pass_threshold']):
                raise ValueError(f'{name}: incompatible input or scoring threshold')
            if row.get('judge_correct') is not None and (
                row.get('judge_model') != JUDGE_ID or row.get('judge_scoring') != prlbench.SCORING
            ):
                raise ValueError(f'{name}: incompatible judge/scoring policy')
            slots[key] = row
        complete_rows = [slots.get((key, sample), {
            'example_id': key, 'sample_index': sample, 'status': 'pending',
            'subject': source['subject'], 'judge_correct': None})
            for key, source in expected.items() for sample in range(4)]
        stats = summarize_evaluation(complete_rows, samples=4)
        score_values = []
        for row in complete_rows:
            if row.get('plan_error') or row.get('judge_error'):
                score_values.append(None)
            elif row.get('status') == 'failed' and row.get('error'):
                score_values.append(0.)
            elif row.get('judge_scoring') == prlbench.SCORING:
                score_values.append(row.get('judge_score'))
            else:
                score_values.append(None)
        metrics = stats['repeated_sampling']
        report['systems'][name] = {
            'mean_at_4': metrics['mean_at_k'], 'pass_at_4': metrics['pass_at_k'],
            'mean_criterion_score_at_4_percent': 100 * mean(score_values) if all(v is not None for v in score_values) else None,
            'observed_slots': len(rows), 'expected_slots': len(expected) * 4,
            'criterion_scores_known': sum(v is not None for v in score_values),
            'by_subfield': {subject: repeated_metrics([row for row in complete_rows if row.get('subject') == subject], 4)
                            for subject in sorted({row['subject'] for row in inputs})},
            'evaluation': stats, 'results_sha256': sha(path) if path.exists() else None}
    write_json_atomic(root / 'comparison-summary.json', report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'summarize'):
        command = commands.add_parser(name)
        command.add_argument('--output-dir', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--source', type=Path, help='Local original PRLBench JSONL; omit to download pinned HF revision')
            command.add_argument('--dataset-revision', default=prlbench.DATASET_REVISION)
            command.add_argument('--max-papers', type=int, help='Explicit first-N paper smoke subset')
            command.add_argument('--pass-threshold', type=float, default=.7)
            command.add_argument('--conductor-config', type=Path, default=REPO_ROOT / 'configs/worker_pool_frontier.yaml')
            command.add_argument('--fugu-config', type=Path, default=REPO_ROOT / 'configs/fugu_ultra_v2_direct.yaml')
            command.add_argument('--prompt-file', type=Path, default=evaluate.DEFAULT_PROMPT_PATH)
            command.add_argument('--planner-model', default='gpt-6.1-sol')
        if name == 'run':
            command.add_argument('--max-tokens', type=int)
            command.add_argument('--concurrency', type=int, default=4, help='Concurrent question/sample slots per system')
            command.add_argument('--attempts', type=int, default=3, help='Recovery attempts within each slot')
            command.add_argument('--temperature', type=float, default=.2)
            command.add_argument('--plan-temperature', type=float, default=.1)
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parser().parse_args(argv)
    try:
        report = prepare(args) if args.command == 'prepare' else asyncio.run(run_comparison(args)) if args.command == 'run' else summarize(args.output_dir)
        print(json.dumps(report, indent=2))
        if args.command == 'run' and any(system['mean_at_4'] is None or
                system['evaluation']['generation_failures'] or system['evaluation']['judge_errors']
                for system in report['systems'].values()):
            return 1
        return 0
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
