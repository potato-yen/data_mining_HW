"""Rebuild the fixed-sample AdaBoost worksheet, without random resampling."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X = tuple(range(1, 11))
Y = (-1, -1, -1, -1, 1, 1, 1, -1, -1, 1)
SAMPLES = (
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (1, 3, 4, 5, 8, 8, 9, 9, 9, 10),
    (1, 5, 5, 6, 6, 7, 7, 8, 9, 9),
)


@dataclass(frozen=True)
class Stump:
    split: float
    left: int
    right: int

    def predict(self, x: float) -> int:
        return self.left if x <= self.split else self.right


RULES = {
    name: Stump(split, left, -left)
    for name, split, left in zip('ABCDEF', (4.5, 4.5, 7.5, 7.5, 9.5, 9.5),
                                (-1, 1, -1, 1, -1, 1))
}


def run_round(weights: list[float], sample_ids: tuple[int, ...], *,
              update: bool) -> dict:
    """Select on the multiset; evaluate and update on the original ten rows."""
    errors = {
        name: sum(rule.predict(X[i-1]) != Y[i-1] for i in sample_ids)
        for name, rule in RULES.items()
    }
    name = min(errors, key=errors.get)  # Stable A–F tie breaking.
    rule = RULES[name]
    predictions = [rule.predict(x) for x in X]
    wrong = [i for i, (p, y) in enumerate(zip(predictions, Y)) if p != y]
    epsilon = math.fsum(weights[i] for i in wrong)
    if not 0 < epsilon < 1:
        raise ValueError('The worksheet requires a finite classifier weight: 0 < epsilon < 1')
    alpha = 0.5 * math.log((1-epsilon)/epsilon)
    unnormalized = normalizer = after = None
    if update:
        unnormalized = [w * math.exp(-alpha*y*p)
                        for w, y, p in zip(weights, Y, predictions)]
        normalizer = math.fsum(unnormalized)
        after = [u/normalizer for u in unnormalized]
        assert math.isclose(math.fsum(after), 1, abs_tol=1e-12)
    return dict(sample_ids=list(sample_ids), sample_errors=errors, rule=name,
                stump=asdict(rule), weights_before=weights.copy(),
                original_predictions=predictions, wrong_ids=[i+1 for i in wrong],
                original_error_count=len(wrong), epsilon=epsilon, alpha=alpha,
                unnormalized_weights=unnormalized, normalizer=normalizer,
                weights_after=after)


def calculate() -> dict:
    weights = [0.1]*10
    rounds = []
    for t, sample in enumerate(SAMPLES, 1):
        result = run_round(weights, sample, update=t < 3)
        result['round'] = t
        rounds.append(result)
        if result['weights_after'] is not None:
            weights = result['weights_after']
    predictions = []
    for x in (2.5, 5.5, 8.5, 9.8):
        votes = [RULES[r['rule']].predict(x) for r in rounds]
        score = math.fsum(r['alpha']*vote for r, vote in zip(rounds, votes))
        predictions.append(dict(x=x, votes=votes, score=score,
                                prediction=(score > 0)-(score < 0)))
    return dict(data=[dict(id=i, x=x, y=y) for i, x, y in zip(X, X, Y)],
                rules={name: asdict(rule) for name, rule in RULES.items()},
                rounds=rounds, predictions=predictions)


def write_csv(path: Path, headers: list[str], rows: list) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    result = calculate()
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    source = ROOT / 'AdaBoost_學習單.docx'
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'adaboost.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    write_csv(output / 'candidate_errors.csv', ['round', 'rule', 'split', 'left', 'right', 'sample_errors'],
              [[r['round'], name, rule.split, rule.left, rule.right, r['sample_errors'][name]]
               for r in result['rounds'] for name, rule in RULES.items()])
    write_csv(output / 'round_summary.csv', ['round', 'rule', 'epsilon', 'alpha', 'wrong_ids', 'original_error_count'],
              [[r['round'], r['rule'], r['epsilon'], r['alpha'], ' '.join(map(str, r['wrong_ids'])),
                r['original_error_count']] for r in result['rounds']])
    write_csv(output / 'weights.csv', ['round', 'id', 'x', 'y', 'prediction', 'weight_before', 'u', 'weight_after'],
              [[r['round'], i+1, X[i], Y[i], r['original_predictions'][i], r['weights_before'][i],
                r['unnormalized_weights'][i], r['weights_after'][i]]
               for r in result['rounds'][:2] for i in range(10)])
    write_csv(output / 'predictions.csv', ['x', 'h1', 'h2', 'h3', 'score', 'prediction'],
              [[p['x'], *p['votes'], p['score'], p['prediction']] for p in result['predictions']])
    lines = [f"Round {r['round']}: {r['sample_errors']}; h={r['rule']}; "
             f"epsilon={r['epsilon']:.12f}; alpha={r['alpha']:.12f}" for r in result['rounds']]
    lines += [f"x={p['x']}: S={p['score']:.12f}; H={p['prediction']:+d}" for p in result['predictions']]
    summary = '\n'.join(lines)+'\n'
    (output / 'console_summary.txt').write_text(summary)
    print(summary, end='')
    # Import only when authoring, so the calculation stays separately testable.
    from build_report import build_report
    build_report(result)


if __name__ == '__main__':
    main()
