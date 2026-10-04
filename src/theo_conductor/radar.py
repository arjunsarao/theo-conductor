"""Compare saved system outcomes by HLE classification or GPQA-D subdomain."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import textwrap
from collections import defaultdict
from html import escape
from pathlib import Path

from .evaluation_stats import repeated_metrics
from .jsonio import read_jsonl, write_json_atomic


def result_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            if (path / 'results.jsonl').exists():
                files.append(path / 'results.jsonl')
            else:
                runs = sorted(path.glob('run-*/results.jsonl'))
                if not runs:
                    raise ValueError(f'No results.jsonl or run-*/results.jsonl in {path}')
                files.extend(runs)
        else:
            files.append(path)
    if len(set(p.resolve() for p in files)) != len(files):
        raise ValueError('The same results file was supplied more than once')
    return files


def load_system(paths: list[Path]) -> tuple[list[dict], int, list[Path]]:
    """Each file contributes independent slots; duplicate slots within it are errors."""
    files = result_files(paths)
    combined = []
    offset = 0
    expected_ids = None
    for path in files:
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f'Empty results file: {path}')
        ids = set()
        slots = set()
        indices = set()
        for row in rows:
            identifier = row.get('example_id', row.get('id'))
            sample = row.get('sample_index', 0)
            if identifier is None or str(identifier).strip() == '':
                raise ValueError(f'Missing question ID in {path}')
            if type(sample) is not int or sample < 0:
                raise ValueError(f'Invalid sample_index in {path}: {sample}')
            identifier = str(identifier)
            slot = (identifier, sample)
            if slot in slots:
                raise ValueError(f'Duplicate sample {slot} in {path}; supply a final snapshot')
            slots.add(slot)
            ids.add(identifier)
            indices.add(sample)
            # Preserve managed verdicts; legacy results use `correct` instead.
            correct = row.get('judge_correct', row.get('correct'))
            combined.append({**row, 'example_id': identifier,
                             'sample_index': sample + offset, 'judge_correct': correct,
                             'status': row.get('status', 'failed' if row.get('error') else 'completed')})
        if expected_ids is not None and ids != expected_ids:
            raise ValueError(f'Mismatched question set in {path}')
        expected_ids = ids
        count = max(indices) + 1
        if indices != set(range(count)):
            raise ValueError(f'Noncontiguous sample indices in {path}')
        # Managed manifests can expose trailing samples absent from the snapshot.
        manifest = path.parent / 'manifest.json'
        if manifest.exists():
            import json
            declared = json.loads(manifest.read_text()).get('samples')
            if declared is not None and declared != count:
                raise ValueError(f'{path}: manifest expects {declared} samples, found {count}')
        offset += count
    metrics = repeated_metrics(combined, offset)
    if metrics['mean_at_k'] is None:
        detail = metrics['by_k'][str(offset)]
        raise ValueError(f'Incomplete or unjudged cohort: {detail["complete_questions"]}/'
                         f'{detail["expected_questions"]} complete questions; recover/judge before plotting')
    return combined, offset, files


def load_categories(path: Path, dataset: str, tag: str = 'tag_1',
                    metadata_format: str | None = None) -> dict[str, tuple[str, ...]]:
    if metadata_format == 'csv' or (metadata_format is None and path.suffix.lower() == '.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_jsonl(path)
    categories = {}
    for row in rows:
        identifier = row.get('id', row.get('example_id'))
        if identifier is None and dataset == 'gpqa-d':
            raw_id = row.get('Record ID')
            if raw_id is not None:
                identifier = f'gpqa-diamond-{raw_id}'
        if identifier is None:
            raise ValueError(f'Missing question ID in {path}')
        identifier = str(identifier)
        keys = ('tag_1', 'tag_2', 'tag_3') if dataset == 'hle' and tag == 'all' else (tag,)
        if dataset == 'gpqa-d':
            values = [row.get('subdomain', row.get('Subdomain'))]
        else:
            values = [row.get(key) for key in keys]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f'Missing classification/subdomain for {identifier} in {path}')
        if identifier in categories:
            raise ValueError(f'Duplicate metadata ID {identifier} in {path}')
        categories[identifier] = tuple(dict.fromkeys(value.strip() for value in values))
    return categories


def comparison(systems: dict[str, list[Path]], categories: dict[str, tuple[str, ...]],
               k: int | None = None) -> dict:
    loaded = {label: load_system(paths) for label, paths in systems.items()}
    cohorts = [{row['example_id'] for row in rows} for rows, _, _ in loaded.values()]
    if any(ids != cohorts[0] for ids in cohorts[1:]):
        raise ValueError('Systems must contain exactly the same question IDs; no intersection filtering')
    missing = cohorts[0] - categories.keys()
    if missing:
        raise ValueError(f'{len(missing)} questions lack classification/subdomain metadata')
    counts = {samples for _, samples, _ in loaded.values()}
    if len(counts) != 1:
        raise ValueError('Systems must have the same independent sample count')
    samples = counts.pop()
    k = samples if k is None else k
    if not 1 <= k <= samples:
        raise ValueError(f'k must be between 1 and {samples}')
    grouped_ids = defaultdict(set)
    for identifier in sorted(cohorts[0]):
        for category in categories[identifier]:
            grouped_ids[category].add(identifier)
    report = {'k': k, 'samples': samples, 'questions': len(cohorts[0]),
              'question_ids': sorted(cohorts[0]),
              'categories': [{'label': label, 'questions': len(ids)}
                             for label, ids in sorted(grouped_ids.items())],
              'failure_policy': 'Exhausted generation failures count as false; missing plans and judge errors '
                                'remain unknown and block plotting. No question filtering.',
              'limitations': 'Configured-system comparison; budgets are not necessarily matched. '
                             'pass@k is an oracle metric. Small category counts can be unstable.',
              'systems': {}}
    for label, (rows, _, files) in loaded.items():
        def metrics(selected):
            return repeated_metrics(selected, samples)['by_k'][str(k)]
        report['systems'][label] = {
            'sources': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in files],
            'generation_failures': sum(bool(row.get('error')) for row in rows),
            'overall': metrics(rows),
            'categories': {category: metrics([r for r in rows if r['example_id'] in ids])
                           for category, ids in sorted(grouped_ids.items())}}
    return report


def radar_svg(report: dict, conductor: str, other: str, title: str) -> str:
    """Standalone SVG, with no plotting dependency or display server required."""
    categories = report['categories']
    if len(categories) < 3:
        raise ValueError('A radar chart requires at least three nonempty categories')
    if 'fugu' in other.lower():
        comparison_color = '#e10600'
    elif re.search(r'\bjev(?:[-\s]+router)?\b', other, re.IGNORECASE):
        comparison_color = '#e752bb'
    elif re.search(r'\b(?:openai|gpt|o\d+)\b', other, re.IGNORECASE):
        comparison_color = '#0fa37f'
    else:
        comparison_color = '#ca603b'
    colors = ('#282d36', comparison_color)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1100" '
             'viewBox="0 0 1800 1100" role="img">',
             f'<title>{escape(title)}</title>',
             '<rect width="1800" height="1100" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#282d36">']

    def text(x, y, content, size=16, anchor='middle', color='#566071', weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                     f'fill="{color}" font-weight="{weight}">{escape(content)}</text>')

    text(55, 55, title, 32, 'start', '#282d36', 'bold')
    text(55, 91, f'{report["questions"]} paired questions · {report["samples"]} independent samples per question',
         18, 'start')
    for panel, metric in enumerate(('mean_at_k', 'pass_at_k')):
        cx, cy, radius = 450 + panel * 900, 495, 275
        text(cx, 140, f'{"mean" if panel == 0 else "pass"}@{report["k"]}', 26, weight='bold')
        angles = [-math.pi / 2 + 2 * math.pi * i / len(categories) for i in range(len(categories))]

        def point(angle, value):
            return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

        for fraction in (.25, .5, .75, 1):
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius * fraction}" '
                         'fill="none" stroke="#dbe1eb"/>')
            text(cx + 10, cy - radius * fraction - 5, f'{fraction:.0%}', 12, 'start', '#8994a6')
        for category, angle in zip(categories, angles, strict=True):
            x, y = point(angle, 1)
            parts.append(f'<path d="M {cx} {cy} L {x} {y}" stroke="#dbe1eb"/>')
            lx, ly = point(angle, 1.14)
            cosine = math.cos(angle)
            anchor = 'start' if cosine > .15 else 'end' if cosine < -.15 else 'middle'
            lines = textwrap.wrap(category['label'], width=18) + [f'(n={category["questions"]})']
            for index, line in enumerate(lines):
                text(lx, ly + (index - (len(lines) - 1) / 2) * 18, line, 14, anchor)
        for index, label in enumerate((conductor, other)):
            values = [report['systems'][label]['categories'][c['label']][metric] for c in categories]
            points = [point(a, v) for a, v in zip(angles, values, strict=True)]
            coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
            dash = '' if index == 0 else ' stroke-dasharray="9 5"'
            parts.append(f'<polygon points="{coords}" fill="{colors[index]}" fill-opacity="0.10" '
                         f'stroke="{colors[index]}" stroke-width="3"{dash}/>')
            for x, y in points:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{colors[index]}"/>')
            overall = report['systems'][label]['overall'][metric]
            text(cx, 907 + 32 * index, f'{label}: {overall:.1%}', 18, color=colors[index], weight='bold')
        text(cx, 875, 'Overall', 15)
    text(900, 1000, f'mean@{report["k"]}: average sample accuracy. '
         f'pass@{report["k"]}: probability of at least one correct answer in k samples.', 16)
    text(900, 1028, 'Shared 0–100% scale; paired question cohort; category counts denote unique questions.', 14)
    text(900, 1055, 'Generation failures count as unsuccessful samples. Small category counts can be unstable.', 14)
    parts.extend(['</g>', '</svg>'])
    return '\n'.join(parts) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hle', 'gpqa-d'], required=True)
    parser.add_argument('--dataset-label', help='Chart cohort name, e.g. HLE-Diamond-Physics')
    parser.add_argument('--metadata', type=Path, required=True,
                        help='HLE classifications.jsonl or GPQA-D CSV/JSONL with subdomain metadata')
    parser.add_argument('--tag', choices=['tag_1', 'tag_2', 'tag_3', 'all'], default='tag_1')
    parser.add_argument('--metadata-format', choices=['csv', 'jsonl'],
                        help='Override format inferred from metadata filename')
    parser.add_argument('--system', nargs='+', action='append', required=True,
                        metavar='LABEL_OR_PATH', help='LABEL followed by result files or run directories; repeat per system')
    parser.add_argument('--conductor', default='Conductor', help='Label of the conductor system')
    parser.add_argument('--k', type=int, help='Defaults to all independent samples')
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--prefix', default=None, help='Output filename prefix (defaults to dataset)')
    args = parser.parse_args(argv)
    try:
        systems = {}
        for spec in args.system:
            if len(spec) < 2 or spec[0] in systems:
                raise ValueError('Each --system needs a unique label and at least one path')
            systems[spec[0]] = [Path(p) for p in spec[1:]]
        if args.conductor not in systems or len(systems) < 2:
            raise ValueError('Supply the conductor label and at least one comparison system')
        if args.dataset == 'gpqa-d' and args.tag != 'tag_1':
            raise ValueError('--tag applies only to HLE')
        categories = load_categories(args.metadata, args.dataset, args.tag, args.metadata_format)
        report = comparison(systems, categories, args.k)
        report['dataset'] = args.dataset
        report['dataset_label'] = args.dataset_label or args.dataset.upper()
        report['classification'] = args.tag if args.dataset == 'hle' else 'subdomain'
        report['metadata'] = {'path': str(args.metadata.resolve()),
                              'sha256': hashlib.sha256(args.metadata.read_bytes()).hexdigest()}
        if args.tag == 'all':
            report['limitations'] += ' HLE tag groups overlap; a question can appear on multiple axes.'
        def slug(value):
            return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-') or 'system'
        prefix = slug(args.prefix or args.dataset)
        outputs = {}
        for other in systems:
            if other != args.conductor:
                name = f'{prefix}-{slug(args.conductor)}-vs-{slug(other)}'
                if name in outputs:
                    raise ValueError('System labels produce colliding output filenames')
                outputs[name] = radar_svg(report, args.conductor, other,
                                         f'{args.conductor} vs {other} · {report["dataset_label"]}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, svg in outputs.items():
            path = args.output_dir / f'{name}.svg'
            path.write_text(svg, encoding='utf-8')
            print(path)
        write_json_atomic(args.output_dir / f'{prefix}-radar-metrics.json', report)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
