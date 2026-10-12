"""Failures which would corrupt the paper's denominator or join incompatible measurements."""
import copy
import csv
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('verify_results', ROOT / 'scripts/verify_results.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.quality = json.loads((ROOT / 'results/quality.json').read_text(encoding='utf-8'))
        self.performance = json.loads((ROOT / 'results/performance.json').read_text(encoding='utf-8'))
        with (ROOT / 'results/tasks.csv').open(encoding='utf-8', newline='') as stream:
            self.tasks = list(csv.DictReader(stream))

    def test_release_recomputes(self):
        self.assertTrue(VERIFIER.verify(ROOT)['passed'])

    def test_rejects_old_denominator(self):
        self.tasks[0]['episodes'] = '60'
        with self.assertRaisesRegex(ValueError, 'task counts'):
            VERIFIER.validate_quality(self.quality, self.tasks)

    def test_rejects_duplicate_task(self):
        self.tasks[1] = copy.deepcopy(self.tasks[0])
        with self.assertRaisesRegex(ValueError, 'Duplicated'):
            VERIFIER.validate_quality(self.quality, self.tasks)

    def test_rejects_unmatched_aggregate(self):
        self.tasks[0]['successes'] = str((int(self.tasks[0]['successes']) + 1) % 16)
        with self.assertRaisesRegex(ValueError, 'Task/suite'):
            VERIFIER.validate_quality(self.quality, self.tasks)

    def test_rejects_combined_memory_and_latency(self):
        coded = next(r for r in self.performance['rows'] if r['configuration'] == 'w15' and r['format'] == 'grouped_huffman')
        coded['peak_allocated_bytes'] = 2784405504
        with self.assertRaisesRegex(ValueError, 'Codec peak-memory'):
            VERIFIER.validate_performance(self.performance)

    def test_preserves_coded_pairing_limit(self):
        coded = next(r for r in self.performance['rows'] if r['format'] == 'grouped_huffman')
        coded['strict_closed_loop_pairing'] = True
        with self.assertRaisesRegex(ValueError, 'strict-pairing'):
            VERIFIER.validate_performance(self.performance)


if __name__ == '__main__':
    unittest.main()
