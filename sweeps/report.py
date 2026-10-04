"""Summarize a sweep's runs into a Markdown report.

Usage::

    python sweeps/report.py runs/pilot              # ranked table of finished runs
    python sweeps/report.py runs/main --matrix      # plus the per-cell matrix

For each run with an ``eval.json`` it reports the mean evaluation ``win × n`` (win rate
relative to the fair share; 1.0 = as good as the field) over all evaluated cells and
lineups, the per-lineup means, the last training iterations' ``win_rel`` and the
training throughput.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any, Dict, List


def load_run(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, 'eval.json'), encoding='utf-8') as fh:
        results = [r for r in json.load(fh) if not r.get('skipped')]
    metrics: List[Dict[str, float]] = []
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding='utf-8') as fh:
            metrics = [json.loads(line) for line in fh if line.strip()]
    tail = metrics[-10:]
    lineups = sorted({r['lineup'] for r in results})
    return {
        'name': os.path.basename(run_dir),
        'mean': mean(r['win_rel'] for r in results) if results else 0.0,
        'by_lineup': {k: mean(r['win_rel'] for r in results if r['lineup'] == k) for k in lineups},
        'cells': {(r['cell'], r['lineup']): r['win_rel'] for r in results},
        'train_tail': mean(m['win_rel'] for m in tail) if tail and 'win_rel' in tail[0] else None,
        'steps_per_sec': mean(m['steps_per_sec'] for m in metrics) if metrics else None,
        'iterations': len(metrics),
    }


def report(out: str, matrix: bool = False) -> str:
    runs = []
    for name in sorted(os.listdir(out)):
        run_dir = os.path.join(out, name)
        if os.path.exists(os.path.join(run_dir, 'eval.json')):
            runs.append(load_run(run_dir))
    runs.sort(key=lambda r: -r['mean'])
    lineups = sorted({k for r in runs for k in r['by_lineup']})
    head = ['run', 'eval win × n'] + [f'vs {k}' for k in lineups] + ['train win_rel (last 10)', 'steps/s', 'iters']
    lines = [f'## {out}', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in runs:
        tail = f"{r['train_tail']:.2f}" if r['train_tail'] is not None else '–'
        sps = f"{r['steps_per_sec']:.0f}" if r['steps_per_sec'] else '–'
        cols = [r['name'], f"{r['mean']:.2f}"] + [f"{r['by_lineup'].get(k, 0):.2f}" for k in lineups]
        lines.append('| ' + ' | '.join(cols + [tail, sps, str(r['iterations'])]) + ' |')
    if matrix and runs:
        cells = sorted({c for r in runs for c, _ in r['cells']})
        lines += [
            '',
            '### win × n per cell (mean over lineups)',
            '',
            '| run | ' + ' | '.join(cells) + ' |',
            '|' + '---|' * (len(cells) + 1),
        ]
        for r in runs:
            row = []
            for cell in cells:
                values = [v for (c, _k), v in r['cells'].items() if c == cell]
                row.append(f'{mean(values):.2f}' if values else '–')
            lines.append(f"| {r['name']} | " + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def cli() -> None:
    parser = argparse.ArgumentParser(description='Summarize a training sweep.')
    parser.add_argument('out', help='sweep output directory, e.g. runs/pilot')
    parser.add_argument('--matrix', action='store_true', help='also print the per-cell matrix')
    args = parser.parse_args()
    print(report(args.out, args.matrix))


if __name__ == '__main__':
    cli()
