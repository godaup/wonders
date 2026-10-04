"""Evaluate a checkpoint across expansion combinations and player counts.

Each *cell* is ``(expansions, n_players)``. For every cell and reference lineup the
checkpoint plays seeded benchmark games (:func:`~wonders.agents.bench.run_bench`, seats
rotating) and we report its win rate relative to the fair share ``1/n`` — 1.0 means as
good as the field, 2.0 twice the fair share — with a 95% interval, plus mean rank and
score. Cells whose cards the model's vocabulary doesn't cover are skipped.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wonders.agents.bench import AGENTS, run_bench
from wonders.env.vocab import vocab_for
from wonders.ml.model import load_checkpoint
from wonders.ml.ppo import opponent_spec

Cell = Tuple[Tuple[str, ...], int]

COMBOS: Tuple[Tuple[str, ...], ...] = (
    ('base',),
    ('base', 'leaders'),
    ('base', 'cities'),
    ('base', 'leaders', 'cities'),
)
PLAYER_COUNTS: Tuple[int, ...] = (3, 4, 5, 6, 7)
ALL_CELLS: Tuple[Cell, ...] = tuple((combo, n) for combo in COMBOS for n in PLAYER_COUNTS)


def reference_lineup(kind: str, n_players: int) -> List[str]:
    """Opponents for the model in an ``n_players`` game: all heuristic, or one board agent + heuristics."""
    if kind == 'heuristic':
        return ['heuristic'] * (n_players - 1)
    if kind == 'board':
        return ['board'] + ['heuristic'] * (n_players - 2)
    if kind == 'mixed':
        pool = ['board', 'heuristic', 'greedy', 'builder', 'civic', 'scientist']
        return pool[: n_players - 1]
    raise ValueError(f'unknown reference lineup {kind!r}')


def cell_key(cell: Cell) -> str:
    expansions, n_players = cell
    extras = [e for e in expansions if e != 'base']
    return f"{'+'.join(extras) or 'base'}/{n_players}p"


def evaluate(
    checkpoint: str,
    cells: Sequence[Cell] = ALL_CELLS,
    lineups: Sequence[str] = ('heuristic', 'board'),
    games: int = 600,
    seed: int = 4242,
    processes: int = 1,
) -> List[Dict[str, Any]]:
    """Benchmark ``checkpoint`` in every cell against every reference lineup."""
    _model, card_names, _meta = load_checkpoint(checkpoint)
    spec = opponent_spec(f'neural:{checkpoint}')
    results: List[Dict[str, Any]] = []
    for cell in cells:
        expansions, n_players = cell
        if not set(vocab_for(tuple(expansions)).card_names) <= set(card_names):
            results.append({'cell': cell_key(cell), 'lineup': None, 'skipped': 'vocabulary does not cover this cell'})
            continue
        for kind in lineups:
            seats = [spec] + [AGENTS[name] for name in reference_lineup(kind, n_players)]
            r = run_bench(seats, n_games=games, seed=seed, expansions=tuple(expansions), processes=processes)
            s = r.stats[spec.name]
            lo, hi = s.win_interval
            results.append(
                {
                    'cell': cell_key(cell),
                    'lineup': kind,
                    'games': s.games,
                    'win_rate': s.win_rate,
                    'win_rel': s.win_rate * n_players,
                    'win_rel_ci': [lo * n_players, hi * n_players],
                    'mean_rank': s.mean_rank,
                    'mean_score': s.mean_score,
                }
            )
    return results


def summary_table(results: Sequence[Dict[str, Any]], title: Optional[str] = None) -> str:
    """A Markdown table of :func:`evaluate` results (win rate relative to fair share)."""
    lines = [f'### {title}', ''] if title else []
    lines += ['| cell | lineup | win × n (95% CI) | win % | mean rank |', '|---|---|---|---|---|']
    for r in results:
        if r.get('skipped'):
            lines.append(f"| {r['cell']} | – | n/a ({r['skipped']}) | – | – |")
            continue
        lo, hi = r['win_rel_ci']
        lines.append(
            f"| {r['cell']} | {r['lineup']} | {r['win_rel']:.2f} ({lo:.2f}–{hi:.2f}) "
            f"| {100 * r['win_rate']:.1f} | {r['mean_rank']:.2f} |"
        )
    return '\n'.join(lines)


def parse_cells(text: str) -> List[Cell]:
    """``'all'``, or comma-separated ``combo/Np`` keys such as ``base/3p,leaders+cities/5p``."""
    if text == 'all':
        return list(ALL_CELLS)
    cells: List[Cell] = []
    for item in text.split(','):
        combo, _, players = item.strip().partition('/')
        expansions = ('base',) if combo == 'base' else ('base', *combo.split('+'))
        cells.append((expansions, int(players.rstrip('p'))))
    return cells


def default_label(checkpoint: str) -> str:
    return os.path.splitext(os.path.basename(checkpoint))[0]
