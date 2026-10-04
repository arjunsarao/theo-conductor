import json
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest

from theo_conductor.radar import comparison, load_categories, load_system, main, radar_svg


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_managed_and_legacy_metrics_and_cli(tmp_path):
    managed = save(tmp_path / 'managed.jsonl', [
        {'example_id': q, 'sample_index': s, 'judge_correct': s == 0}
        for q in ('a', 'b', 'c') for s in range(4)])
    legacy = [save(tmp_path / f'run-{s}' / 'results.jsonl', [
        {'example_id': q, 'correct': s < 2} for q in ('a', 'b', 'c')]) for s in range(4)]
    cats = {q: (q,) for q in ('a', 'b', 'c')}
    report = comparison({'Conductor': [managed], 'Other': legacy}, cats, k=2)
    assert report['systems']['Conductor']['overall']['mean_at_k'] == .25
    assert report['systems']['Conductor']['overall']['pass_at_k'] == .5
    assert report['systems']['Other']['overall']['pass_at_k'] == pytest.approx(1 - 1 / math.comb(4, 2))
    metadata = save(tmp_path / 'metadata.jsonl', [{'id': q, 'tag_1': q} for q in cats])
    out = tmp_path / 'figures'
    assert main(['--dataset', 'hle', '--metadata', str(metadata), '--k', '2',
                 '--system', 'Conductor', str(managed), '--system', 'Other',
                 *map(str, legacy), '--output-dir', str(out)]) == 0
    root = ElementTree.parse(out / 'hle-conductor-vs-other.svg').getroot()
    assert root.tag.endswith('svg')
    assert 'mean@2' in ''.join(root.itertext())
    assert json.loads((out / 'hle-radar-metrics.json').read_text())['questions'] == 3


@pytest.mark.parametrize('bad', [
    {'judge_correct': None}, {'judge_correct': True, 'judge_error': 'timeout'},
    {'judge_correct': False, 'plan_error': 'invalid'},
])
def test_unknowns_block_plots(tmp_path, bad):
    path = save(tmp_path / 'results.jsonl', [{'example_id': 'q', **bad}])
    with pytest.raises(ValueError, match='Incomplete or unjudged'):
        load_system([path])


def test_failures_remain_in_denominator(tmp_path):
    path = save(tmp_path / 'results.jsonl', [
        {'example_id': 'q', 'sample_index': 0, 'status': 'failed', 'error': 'exhausted'},
        {'example_id': 'q', 'sample_index': 1, 'judge_correct': True}])
    report = comparison({'Conductor': [path], 'Other': [path]}, {'q': ('A',)})
    assert report['systems']['Conductor']['overall']['mean_at_k'] == .5
    assert report['systems']['Conductor']['generation_failures'] == 1


def test_duplicate_missing_and_unpaired_slots(tmp_path):
    row = {'example_id': 'q', 'judge_correct': True}
    duplicate = save(tmp_path / 'dup.jsonl', [row, row])
    with pytest.raises(ValueError, match='Duplicate sample'):
        load_system([duplicate])
    missing = save(tmp_path / 'missing.jsonl', [row, {**row, 'example_id': 'r', 'sample_index': 1}])
    with pytest.raises(ValueError, match='Incomplete'):
        load_system([missing])
    first = save(tmp_path / 'first.jsonl', [row])
    second = save(tmp_path / 'second.jsonl', [{**row, 'example_id': 'r'}])
    with pytest.raises(ValueError, match='same question IDs'):
        comparison({'Conductor': [first], 'Other': [second]}, {'q': ('A',), 'r': ('B',)})
    (tmp_path / 'manifest.json').write_text('{"samples": 4}')
    with pytest.raises(ValueError, match='manifest expects'):
        load_system([first])


def test_categories(tmp_path):
    hle = save(tmp_path / 'hle.jsonl', [{'id': 'q', 'tag_1': 'A', 'tag_2': 'B', 'tag_3': 'C'}])
    assert load_categories(hle, 'hle') == {'q': ('A',)}
    assert load_categories(hle, 'hle', 'all') == {'q': ('A', 'B', 'C')}
    gpqa = tmp_path / 'gpqa.csv'
    gpqa.write_text('Record ID,Subdomain\n123,Quantum mechanics\n')
    assert load_categories(gpqa, 'gpqa-d') == {'gpqa-diamond-123': ('Quantum mechanics',)}
    normalized = save(tmp_path / 'gpqa.jsonl', [{'id': 'gpqa-diamond-123', 'subdomain': 'Quantum mechanics'}])
    assert load_categories(normalized, 'gpqa-d') == load_categories(gpqa, 'gpqa-d')


def test_category_denominators_overlap_and_escaping(tmp_path):
    path = save(tmp_path / 'results.jsonl', [
        {'example_id': 'a', 'correct': True}, {'example_id': 'b', 'correct': False}])
    report = comparison({'Conductor': [path], 'A & B': [path]},
                        {'a': ('X & Y', 'Z'), 'b': ('Z', '<Other>')})
    assert report['systems']['Conductor']['categories']['Z']['mean_at_k'] == .5
    assert report['systems']['Conductor']['categories']['X & Y']['mean_at_k'] == 1
    svg = radar_svg(report, 'Conductor', 'A & B', '<Comparison>')
    ElementTree.fromstring(svg)
    assert '&lt;Comparison&gt;' in svg
    assert 'A &amp; B' in svg


def test_unequal_samples_and_missing_metadata(tmp_path):
    one = save(tmp_path / 'one.jsonl', [{'example_id': 'q', 'correct': True}])
    two = save(tmp_path / 'two.jsonl', [
        {'example_id': 'q', 'sample_index': s, 'correct': True} for s in range(2)])
    with pytest.raises(ValueError, match='same independent sample'):
        comparison({'Conductor': [one], 'Other': [two]}, {'q': ('A',)})
    with pytest.raises(ValueError, match='lack classification'):
        comparison({'Conductor': [one], 'Other': [one]}, {})
