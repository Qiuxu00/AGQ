"""Recompute released aggregate results without models, weights, or GPU dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_quality(quality: dict, tasks: list[dict]) -> dict:
    rows = quality['rows']
    ids = {row['id'] for row in rows}
    require(len(rows) == len(ids) == 10, 'Expected ten unique configurations')
    require(quality['episodes'] == 6000, 'Expected 6000 evaluated episodes')
    require(quality['weight_format'] == 'original_packed', 'Quality format must remain explicit')
    require(len(tasks) == 400, 'Expected 400 configuration/task records')
    suites = set(quality['suites'])
    require(suites == {'spatial', 'object', 'goal', 'long'}, 'Unexpected suites')
    totals = defaultdict(lambda: Counter(episodes=0, successes=0))
    seen = set()
    for task in tasks:
        config, suite = task['configuration'], task['suite']
        index, successes, episodes = (int(task[k]) for k in ('task_index', 'successes', 'episodes'))
        require(config in ids and suite in suites, 'Unknown configuration or suite')
        key = config, suite, index
        require(key not in seen and 0 <= index < 10, 'Duplicated or invalid task index')
        seen.add(key)
        require(episodes == 15 and 0 <= successes <= episodes, 'Invalid task counts')
        totals[config, suite].update(episodes=episodes, successes=successes)
    for row in rows:
        require(set(row['by_suite']) == suites, 'A configuration is missing suites')
        for suite in suites:
            require(dict(totals[row['id'], suite]) == row['by_suite'][suite], 'Task/suite count mismatch')
            require(row['by_suite'][suite]['episodes'] == 150, 'Expected 150 episodes per suite')
        count = sum(v['successes'] for v in row['by_suite'].values())
        require(count == row['successes'] and row['episodes'] == 600, 'Suite/overall count mismatch')
        require(math.isclose(row['success_percent'], count / 6, abs_tol=1e-10), 'Incorrect success percentage')
    require(sum(r['episodes'] for r in rows) == quality['episodes'], 'Dataset denominator mismatch')
    return {'configurations': len(ids), 'episodes': 6000, 'task_records': len(tasks)}


def validate_performance(performance: dict) -> dict:
    require(performance['bytes_per_gb'] == 10**9, 'GB uses decimal bytes')
    require(performance['inference_calls'] == 10, 'Expected ten fixed-input timing calls')
    rows = performance['rows']
    require(len(rows) == 8, 'Unexpected performance table size')
    baseline = next(row for row in rows if row['configuration'] == 'bf16')
    seen = set()
    for row in rows:
        key = row['configuration'], row['format']
        require(key not in seen, 'Repeated performance configuration/format')
        seen.add(key)
        require(row['model_bytes'] > 0 and row['peak_allocated_bytes'] > 0 and row['latency_ms'] > 0,
                'Performance quantities must be positive')
        require(math.isclose(row['compression_ratio'], baseline['model_bytes'] / row['model_bytes'], rel_tol=1e-10),
                'Whole-model compression ratio mismatch')
        if row['format'] == 'grouped_huffman':
            require(row['implementation'] == 'CODEBOOK003', 'Main coded table uses CODEBOOK003')
            require(row['strict_closed_loop_pairing'] is False, 'Preserve strict-pairing limitation')
    alternatives = performance['alternative_codec_measurements']
    require(len(alternatives) == 4, 'Expected both codecs at W2 and W1.5*')
    require({(r['configuration'], r['implementation']) for r in alternatives}
            == {(c, k) for c in ('w2', 'w15') for k in ('CODEBOOK001', 'CODEBOOK003')},
            'Alternative-codec grid is incomplete')
    for row in rows:
        for alt in alternatives:
            if row['configuration'] == alt['configuration'] and row['implementation'] == alt['implementation']:
                require(row['peak_allocated_bytes'] == alt['peak_allocated_bytes'], 'Codec peak-memory mismatch')
                require(math.isclose(row['latency_ms'], alt['latency_ms'], abs_tol=1e-9), 'Codec latency mismatch')
    return {'performance_rows': len(rows), 'codec_alternatives': len(alternatives)}


def verify(root: Path) -> dict:
    data = root / 'results'
    quality = read_json(data / 'quality.json')
    with (data / 'tasks.csv').open(encoding='utf-8', newline='') as f:
        tasks = list(csv.DictReader(f))
    report = validate_quality(quality, tasks)
    report.update(validate_performance(read_json(data / 'performance.json')))
    allocations = read_json(data / 'allocations.json')
    require(allocations['bitwidth_order'] == [1, 2, 3, 4], 'Bitwidth order changed')
    for row in allocations['rows']:
        require(sum(row['vision']) == 204 and sum(row['language']) == 224, 'Wrong backbone matrix count')
        require(row['combined'] == [v + l for v, l in zip(row['vision'], row['language'])], 'Matrix count mismatch')
    literature = read_json(data / 'published_baselines.json')
    require(len(literature['rows']) == 11, 'Literature selection changed')
    require(next(r for r in literature['rows'] if r[0] == 'FP16')[-1] == 96.9, 'FP16 reference changed')
    require(not any(r[0] in {'AWQ', 'GPTQ'} and r[1] == 2.5 for r in literature['rows']),
            'Do not invent missing W2.5 references')
    cases = read_json(root / 'examples/cases/catalog.json')['cases']
    ids = {r['id'] for r in quality['rows']}
    require(len(cases) == 13 and len({c['id'] for c in cases}) == 13, 'Expected thirteen unique cases')
    require(all(set(c['configurations']) <= ids for c in cases), 'Unknown case configuration')
    manifest = read_json(data / 'manifest.json')
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), 'Manifest path escapes repository')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Checksum mismatch: {name}')
    report.update(cases=len(cases), allocation_rows=len(allocations['rows']), checksums=len(manifest['sha256']), passed=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == '__main__':
    main()
