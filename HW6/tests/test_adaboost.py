"""Hand-derived regression cases for the worksheet's two-dataset semantics."""
import math
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from src import adaboost
from src import build_report


class AdaBoostTests(unittest.TestCase):
    def test_latex_output_has_no_escaped_control_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(build_report, 'ROOT', Path(directory)):
                build_report.build_report(adaboost.calculate())
            for path in Path(directory).rglob('*.tex'):
                controls = [ord(c) for c in path.read_text() if ord(c) < 32 and c != '\n']
                self.assertEqual(controls, [], path.name)

    def test_sample_multiplicity_and_selected_stumps(self):
        result = adaboost.calculate()
        self.assertEqual([r['sample_errors'] for r in result['rounds']], [
            dict(zip('ABCDEF', [2, 8, 5, 5, 3, 7])),
            dict(zip('ABCDEF', [5, 5, 6, 4, 1, 9])),
            dict(zip('ABCDEF', [3, 7, 9, 1, 6, 4])),
        ])
        self.assertEqual([r['rule'] for r in result['rounds']], ['A', 'E', 'D'])

    def test_weighted_errors_use_original_data(self):
        rounds = adaboost.calculate()['rounds']
        for r, epsilon, ratio in zip(rounds, [1/5, 3/16, 5/26], [4, 13/3, 21/5]):
            self.assertAlmostEqual(r['epsilon'], epsilon)
            self.assertAlmostEqual(r['alpha'], math.log(ratio)/2)
        self.assertEqual(rounds[0]['wrong_ids'], [8, 9])
        self.assertEqual(rounds[1]['wrong_ids'], [5, 6, 7])
        self.assertEqual(rounds[2]['original_error_count'], 5)
        self.assertIsNone(rounds[2]['weights_after'])

    def test_normalization_and_weight_propagation(self):
        rounds = adaboost.calculate()['rounds']
        expected = [[1/16]*7 + [1/4, 1/4, 1/16],
                    [1/26]*4 + [1/6]*3 + [2/13, 2/13, 1/26]]
        for i, weights in enumerate(expected):
            for actual, want in zip(rounds[i]['weights_after'], weights):
                self.assertAlmostEqual(actual, want)
            self.assertAlmostEqual(sum(rounds[i]['weights_after']), 1)
            self.assertEqual(rounds[i+1]['weights_before'], rounds[i]['weights_after'])

    def test_prediction_and_boundary(self):
        for rule in adaboost.RULES.values():
            self.assertEqual(rule.predict(rule.split), rule.left)
            self.assertEqual(rule.predict(rule.split+0.01), rule.right)
        predictions = adaboost.calculate()['predictions']
        self.assertEqual([p['votes'] for p in predictions],
                         [[-1, -1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]])
        self.assertEqual([p['prediction'] for p in predictions], [-1, 1, -1, 1])
        for p, ratio in zip(predictions, [63/260, 252/65, 20/91, 260/63]):
            self.assertAlmostEqual(p['score'], math.log(ratio)/2)


if __name__ == '__main__':
    unittest.main()
