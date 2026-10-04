"""Generate the training-sweep specs (pilot / main / league) as JSON.

Usage::

    python sweeps/generate.py                       # defaults
    python sweeps/generate.py --lr 1e-4 --entropy 0.003   # after the pilot picked them

Then run a spec with ``wonders train sweep sweeps/<name>.json`` (resumable; ``--dry-run``
lists the runs). All tracks use the universal vocabulary, so every model can be
evaluated on every (expansions, players) cell.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS: Dict[str, List[str]] = {
    'base': ['base'],
    'leaders': ['base', 'leaders'],
    'cities': ['base', 'cities'],
    'leaders+cities': ['base', 'leaders', 'cities'],
}
PLAYERS = [3, 4, 5, 6, 7]
OPPONENT_POOL = ['heuristic', 'board', 'board-focus', 'greedy', 'builder', 'civic', 'scientist']
EVAL_ALL = {'cells': 'all', 'games': 400, 'lineups': ['heuristic', 'board']}


def teacher_tables(combo: str, n: int) -> List[Tuple[List[str], List[str]]]:
    """Imitation tables at one cell: all board-aware teachers, and teachers among mixed opponents."""
    mixed = (['board', 'heuristic', 'greedy', 'civic', 'scientist', 'builder', 'militarist'])[:n]
    return [(COMBOS[combo], ['board'] * n), (COMBOS[combo], mixed)]


def imitation(name: str, cells: Sequence[Tuple[str, int]], games: int, hidden: int = 256, layers: int = 2) -> Dict:
    tables = [t for combo, n in cells for t in teacher_tables(combo, n)]
    return {
        'name': name,
        'kind': 'imitate',
        'universal': True,
        'teachers': ['board', 'board-focus'],
        'games': games,
        'tables': tables,
        'hidden': hidden,
        'layers': layers,
        'epochs': 12,
    }


def ppo(
    name: str,
    init: str,
    cells: Sequence[Tuple[str, int]],
    steps: int,
    seed: int,
    hp: Dict,
    envs: int,
    per_worker: int,
    eval_spec: Dict = EVAL_ALL,
    **extra: Any,
) -> Dict:
    return {
        'name': name,
        'kind': 'ppo',
        'init': init,
        'steps': steps,
        'ppo': {
            'tables': [[COMBOS[c], n] for c, n in cells],
            'opponent_pool': OPPONENT_POOL,
            'universal': True,
            'n_envs': envs,
            'envs_per_worker': per_worker,
            'rollout_steps': hp.get('rollout_steps', 64),
            'lr': hp['lr'],
            'entropy_coef': hp['entropy_coef'],
            'seed': seed,
            **extra,
        },
        'eval': eval_spec,
    }


def pilot() -> Dict:
    cells = {'base-3p': [('base', 3)], 'lc-5p': [('leaders+cities', 5)]}
    runs: List[Dict] = []
    for key, cell in cells.items():
        runs.append(imitation(f'imit-{key}', cell, games=3000 if key == 'base-3p' else 1500))
        run = ppo(
            f'ppo-{key}',
            f'imit-{key}',
            cell,
            500_000,
            0,
            {'lr': 1e-4, 'entropy_coef': 0.003},
            24,
            4,
            eval_spec={'cells': f"{cell[0][0]}/{cell[0][1]}p", 'games': 600, 'lineups': ['heuristic', 'board']},
        )
        run['grid'] = {'lr': [3e-5, 1e-4, 3e-4], 'entropy_coef': [0.001, 0.003, 0.01]}
        runs.append(run)
    return {'out': 'runs/pilot', 'runs': runs}


def main_sweep(hp: Dict) -> Dict:
    runs: List[Dict] = []
    every = [(c, n) for c in COMBOS for n in PLAYERS]
    # Track C — universal models (all 20 cells mixed), two sizes x 3 seeds.
    for hidden, layers in ((256, 2), (512, 3)):
        imit = f'imit-C-{hidden}x{layers}'
        runs.append(imitation(imit, every, games=12_000, hidden=hidden, layers=layers))
        for seed in (0, 1, 2):
            runs.append(ppo(f'ppo-C-{hidden}x{layers}-s{seed}', imit, every, 20_000_000, seed, hp, 64, 4))
    # Track B — one model per expansion combination, all player counts mixed.
    for combo in COMBOS:
        cells = [(combo, n) for n in PLAYERS]
        imit = f'imit-B-{combo}'
        runs.append(imitation(imit, cells, games=6000))
        for seed in (0, 1):
            runs.append(ppo(f'ppo-B-{combo}-s{seed}', imit, cells, 6_000_000, seed, hp, 48, 4))
    # Track A — specialists, one (expansions, players) cell each.
    for combo, n in every:
        key = f'{combo}-{n}p'
        imit = f'imit-A-{key}'
        runs.append(imitation(imit, [(combo, n)], games=max(800, 9000 // n)))
        for seed in (0, 1):
            runs.append(ppo(f'ppo-A-{key}-s{seed}', imit, [(combo, n)], 2_000_000, seed, hp, 32, 4))
    return {'out': 'runs/main', 'runs': runs}


def league_sweep(hp: Dict, base_run: str = 'ppo-C-256x2-s0') -> Dict:
    every = [(c, n) for c in COMBOS for n in PLAYERS]
    run = ppo(
        f'league-{base_run}', base_run, every, 10_000_000, 7, hp, 64, 4, league=True, snapshot_every=50, league_size=6
    )
    return {'out': 'runs/main', 'runs': [run]}


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy', type=float, default=0.003)
    parser.add_argument('--rollout', type=int, default=64)
    args = parser.parse_args()
    hp = {'lr': args.lr, 'entropy_coef': args.entropy, 'rollout_steps': args.rollout}
    for name, spec in (('pilot', pilot()), ('main', main_sweep(hp)), ('league', league_sweep(hp))):
        with open(os.path.join(HERE, f'{name}.json'), 'w', encoding='utf-8') as fh:
            json.dump(spec, fh, indent=1)
            fh.write("\n")
        print(f"{name}.json: {len(spec['runs'])} run entries")


if __name__ == '__main__':
    cli()
