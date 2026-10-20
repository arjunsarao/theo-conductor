import json
from xml.etree import ElementTree

import pytest

from theo_conductor.grouped_bars import grouped_bars_svg, load_comparison, main, percentage


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def cohort(tmp_path):
    questions = [
        {'id': 'r1', 'track': 'research', 'subject': 'physics', 'question': 'Repeated'},
        {'id': 'r2', 'track': 'research', 'subject': 'physics', 'question': 'Repeated'},
        {'id': 'o1', 'track': 'olympiad', 'subject': 'physics', 'question': 'Distinct'},
    ]
    write_jsonl(tmp_path / 'questions.jsonl', questions)
    for arm in ('conductor', 'fugu'):
        write_jsonl(tmp_path / arm / 'results.jsonl', [
            {'example_id': q['id'], 'sample_index': sample,
             'judge_correct': q['id'] == 'r1' and sample == 0}
            for q in questions for sample in range(4)])
    return tmp_path


def test_track_metrics_and_source_row_denominators(cohort, tmp_path):
    report = load_comparison(cohort)
    research = report['systems']['Conductor']['categories']['Research']
    assert research['mean_at_k'] == .125
    assert research['pass_at_k'] == .5
    assert research['expected_questions'] == 2
    assert report['duplicate_question_groups'] == [['r1', 'r2']]
    assert report['systems']['Conductor']['categories']['Olympiad']['mean_at_k'] == 0
    root = ElementTree.fromstring(grouped_bars_svg(report))
    labels = ''.join(root.itertext())
    assert 'Research' in labels and 'Olympiad' in labels
    assert '12.5%' in labels and '50%' in labels
    out = tmp_path / 'figures'
    assert main(['--comparison-root', str(cohort), '--output-dir', str(out)]) == 0
    assert (out / 'frontierscience-physics-grouped-bars.svg').exists()
    saved = json.loads((out / 'frontierscience-physics-grouped-bars-metrics.json').read_text())
    assert saved['metadata']['sha256'] == report['metadata']['sha256']


def test_missing_whole_question_rejected(cohort):
    for arm in ('conductor', 'fugu'):
        path = cohort / arm / 'results.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write_jsonl(path, [r for r in rows if r['example_id'] != 'r2'])
    with pytest.raises(ValueError, match='cover every question'):
        load_comparison(cohort)


def test_nonphysics_cohort_rejected(cohort):
    write_jsonl(cohort / 'questions.jsonl', [
        {'id': 'q', 'track': 'research', 'subject': 'chemistry'}])
    with pytest.raises(ValueError, match='Non-physics'):
        load_comparison(cohort)


def test_percentage_rounding():
    assert percentage(.6625) == '66.3%'
    assert percentage(.5875) == '58.8%'
    assert percentage(.7) == '70%'
