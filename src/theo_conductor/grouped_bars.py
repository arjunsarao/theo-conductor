"""Plot FrontierScience-physics mean@k/pass@k by Research and Olympiad track."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path

from .jsonio import read_jsonl, write_json_atomic
from .radar import comparison


def load_comparison(root: Path, k: int | None = None) -> dict:
    questions_path = root / 'questions.jsonl'
    questions = read_jsonl(questions_path)
    categories = {}
    texts = {}
    for row in questions:
        identifier = row.get('id')
        track = row.get('track')
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError('Every question must have a nonempty string ID')
        if identifier in categories:
            raise ValueError(f'Duplicate question ID: {identifier}')
        if track not in ('research', 'olympiad'):
            raise ValueError(f'Unknown FrontierScience track for {identifier}: {track}')
        if str(row.get('subject', '')).lower() != 'physics':
            raise ValueError(f'Non-physics question in cohort: {identifier}')
        categories[identifier] = (track.title(),)
        if row.get('question'):
            texts.setdefault((track, row['question']), []).append(identifier)
    report = comparison({'Conductor': [root / 'conductor'],
                         'Fugu Ultra v2': [root / 'fugu']}, categories, k)
    if set(report['question_ids']) != set(categories):
        raise ValueError('Results must cover every question in questions.jsonl')
    if {c['label'] for c in report['categories']} != {'Research', 'Olympiad'}:
        raise ValueError('Both Research and Olympiad tracks are required')
    report['dataset_label'] = 'FrontierScience-physics'
    report['metadata'] = {'path': str(questions_path.resolve()),
                          'sha256': hashlib.sha256(questions_path.read_bytes()).hexdigest()}
    report['duplicate_question_groups'] = [ids for ids in texts.values() if len(ids) > 1]
    report['limitations'] += ' Original source rows are retained, including repeated question texts.'
    manifest_path = root / 'comparison-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        report['comparison_manifest'] = {
            'path': str(manifest_path.resolve()),
            'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'settings': {key: value for key, value in manifest.items() if key != 'file_sha256'}}
    return report


def percentage(value: float) -> str:
    number = (Decimal(str(value)) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return f'{number:f}'.removesuffix('.0') + '%'


def grouped_bars_svg(report: dict) -> str:
    title = 'Conductor vs Fugu Ultra v2 · FrontierScience-physics'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1100" '
             'viewBox="0 0 1800 1100" role="img">',
             f'<title>{escape(title)}</title>',
             '<rect width="1800" height="1100" fill="white"/>',
             '<g font-family="Arial, sans-serif">']

    def text(x, y, value, size=24, color='#566071', anchor='middle', weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                     f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def rect(x, y, width, height, color, fill, dashed=False):
        dash = ' stroke-dasharray="17 10"' if dashed else ''
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                     f'fill="{fill}" stroke="{color}" stroke-width="3"{dash}/>')

    text(55, 65, title, 38, '#282d36', 'start', 'bold')
    text(55, 110, f'Research and Olympiad tracks · {report["samples"]} independent samples per question',
         24, anchor='start')
    for index, (label, color, fill) in enumerate((('Conductor', '#282d36', '#e9eaeb'),
                                                 ('Fugu Ultra v2', '#e10600', '#fce7e6'))):
        x = 75 + index * 260
        rect(x, 180, 40, 18, color, fill, index == 1)
        text(x + 60, 200, label, 24, anchor='start')
    left, right, top, bottom = 150, 1740, 280, 850
    height = bottom - top
    for value in (0, .25, .5, .75, 1):
        y = bottom - value * height
        parts.append(f'<path d="M {left} {y} H {right}" stroke="#dbe1eb" stroke-width="1.5"/>')
        text(left - 25, y + 9, f'{value:.0%}', 25, '#8994a6', 'end')
    parts.append(f'<path d="M 940 {top - 30} V {bottom}" stroke="#dbe1eb" stroke-width="1.5"/>')
    counts = {c['label']: c['questions'] for c in report['categories']}
    for track, centers in (('Research', (320, 655)), ('Olympiad', (1200, 1535))):
        for metric, center in zip(('mean_at_k', 'pass_at_k'), centers, strict=True):
            for index, (label, color, fill) in enumerate((('Conductor', '#282d36', '#e9eaeb'),
                                                         ('Fugu Ultra v2', '#e10600', '#fce7e6'))):
                value = report['systems'][label]['categories'][track][metric]
                x = center - 100 + index * 100
                y = bottom - value * height
                rect(x, y, 100, value * height, color, fill, index == 1)
                text(x + 50, y - 24, percentage(value), 26, color, weight='bold')
            text(center, bottom + 45, f'{"mean" if metric == "mean_at_k" else "pass"}@{report["k"]}', 27)
        center = sum(centers) / 2
        text(center, 972, track, 34, weight='bold')
        text(center, 1014, f'{counts[track]} questions', 24)
    note = 'Paired source-row cohort; generation failures count false. pass@k is an oracle metric.'
    if report.get('duplicate_question_groups'):
        note += ' Repeated source question texts retained.'
    text(900, 1070, note, 15)
    parts.extend(['</g>', '</svg>'])
    return '\n'.join(parts) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', required=True, type=Path,
                        help='Directory containing questions.jsonl and conductor/fugu run directories')
    parser.add_argument('--k', type=int, help='Defaults to all independent samples')
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--png', action='store_true', help='Also export PNG using installed ImageMagick')
    args = parser.parse_args(argv)
    try:
        converter = shutil.which('convert') if args.png else None
        if args.png and converter is None:
            raise ValueError('--png requires ImageMagick (convert); omit it to produce SVG only')
        report = load_comparison(args.comparison_root, args.k)
        svg = grouped_bars_svg(report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / 'frontierscience-physics-grouped-bars.svg'
        output.write_text(svg, encoding='utf-8')
        print(output)
        write_json_atomic(args.output_dir / 'frontierscience-physics-grouped-bars-metrics.json', report)
        if converter:
            png = output.with_suffix('.png')
            subprocess.run([converter, '-background', 'white', str(output.resolve()), str(png.resolve())],
                           check=True, capture_output=True, text=True)
            print(png)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
