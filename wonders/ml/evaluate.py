"""Evaluate a checkpoint across expansion combinations and player counts.

Each *cell* is ``(expansions, n_players)``. For every cell and reference lineup the
checkpoint plays seeded benchmark games (:func:`~wonders.agents.bench.run_bench`, seats
rotating) and we report its win rate relative to the fair share ``1/n`` — 1.0 means as
good as the field, 2.0 twice the fair share — with a 95% interval, plus mean rank and
score. Cells whose cards the model's vocabulary doesn't cover are skipped.

:func:`head_to_head` seats several checkpoints at the *same* table (filled up with a
reference agent) and also reports, per pair, how often one finishes above the other —
far more sensitive than win rates once the models beat the scripted agents easily.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders.agents.bench import AGENTS, AgentSpec, run_bench, wilson_interval
from wonders.env.vocab import vocab_for
from wonders.ml.agent import NeuralAgent
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


def _covers(card_names: Sequence[str], expansions: Sequence[str]) -> bool:
    return set(vocab_for(tuple(expansions)).card_names) <= set(card_names)


def _pair_summary(counts: Dict[Tuple[str, str], List[int]]) -> List[Dict[str, Any]]:
    out = []
    for (a, b), (above, n) in counts.items():
        lo, hi = wilson_interval(above, n)
        out.append({'a': a, 'b': b, 'games': n, 'a_above_b': above / n, 'ci': [lo, hi]})
    return out


def neural_spec(label: str, checkpoint: str, **kwargs: Any) -> AgentSpec:
    """A picklable :class:`NeuralAgent` spec under a display ``label`` (extra kwargs are passed on)."""
    return AgentSpec.of(label, NeuralAgent, checkpoint=os.path.abspath(checkpoint), name=label, **kwargs)


def head_to_head(
    models: Sequence[Tuple[str, str]],
    cells: Sequence[Cell] = ALL_CELLS,
    games: int = 400,
    seed: int = 4242,
    filler: str = 'heuristic',
    processes: int = 1,
    seating: str = 'shuffle',
) -> Dict[str, Any]:
    """Play ``models`` (``(label, checkpoint)`` pairs) at the same tables.

    Only models whose vocabulary covers a cell's cards play in it; see :func:`compare_agents`
    for the result layout.
    """
    labels = [label for label, _path in models]
    if len(set(labels)) != len(labels):
        raise ValueError('give every model a distinct label')
    vocab = {label: load_checkpoint(path)[1] for label, path in models}
    return compare_agents(
        [neural_spec(label, path) for label, path in models],
        cells=cells,
        games=games,
        seed=seed,
        filler=filler,
        processes=processes,
        seating=seating,
        eligible=lambda cell: [label for label in labels if _covers(vocab[label], cell[0])],
    )


def compare_agents(
    specs: Sequence[AgentSpec],
    cells: Sequence[Cell] = ALL_CELLS,
    games: int = 400,
    seed: int = 4242,
    filler: str = 'heuristic',
    processes: int = 1,
    eligible: Optional[Callable[[Cell], Sequence[str]]] = None,
    seating: str = 'shuffle',
) -> Dict[str, Any]:
    """Seat ``specs`` (distinct names) at the same tables, one seat each.

    ``seating`` is passed to :func:`~wonders.agents.bench.run_bench`. Keep the default
    ``'shuffle'`` (every game seated at random): with ``'rotate'`` the agents keep their
    relative order, so the first-listed model always picks from a hand before the next one
    in Ages I and III — worth ~7 points of "ranked above" between two trained models.
    ``filler`` agents take the remaining seats; ``eligible(cell)`` may restrict which specs
    play a cell (cells with fewer than two eligible agents, or more than seats, are
    skipped). Returns per-cell results — each agent's win rate × players with a 95%
    interval, and per pair ``a > b`` the share of games where ``a`` ranked above ``b`` —
    plus those pair counts pooled over all cells (``'pairs'``) and per expansion
    combination (``'pairs_by_combo'``).
    """
    by_name = {spec.name: spec for spec in specs}
    labels = list(by_name)
    if len(labels) != len(specs):
        raise ValueError('give every agent a distinct name')
    if filler in by_name:
        raise ValueError(f'the filler {filler!r} must not also be compared')
    pooled: Dict[Tuple[str, str], List[int]] = {}
    by_combo: Dict[str, Dict[Tuple[str, str], List[int]]] = {}
    cell_results: List[Dict[str, Any]] = []
    for cell in cells:
        expansions, n_players = cell
        present = list(eligible(cell)) if eligible is not None else labels
        if len(present) < 2 or len(present) > n_players:
            cell_results.append({'cell': cell_key(cell), 'skipped': f'{len(present)} eligible models'})
            continue
        seats = [by_name[label] for label in present] + [AGENTS[filler]] * (n_players - len(present))
        r = run_bench(
            seats, n_games=games, seed=seed, expansions=tuple(expansions), processes=processes, seating=seating
        )
        entry: Dict[str, Any] = {'cell': cell_key(cell), 'games': games, 'models': {}, 'pairs': {}}
        for label in present:
            s = r.stats[label]
            lo, hi = s.win_interval
            entry['models'][label] = {
                'win_rel': s.win_rate * n_players,
                'win_rel_ci': [lo * n_players, hi * n_players],
                'mean_score': s.mean_score,
                'mean_rank': s.mean_rank,
            }
        combo = cell_key(cell).split('/')[0]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                above = sum(dict(game)[a] < dict(game)[b] for game in r.ranks)  # each model sits once per game
                entry['pairs'][f'{a} > {b}'] = above / games
                for counts in (
                    pooled.setdefault((a, b), [0, 0]),
                    by_combo.setdefault(combo, {}).setdefault((a, b), [0, 0]),
                ):
                    counts[0] += above
                    counts[1] += games
        cell_results.append(entry)
    return {
        'models': labels,
        'filler': filler,
        'seating': seating,
        'cells': cell_results,
        'pairs': _pair_summary(pooled),
        'pairs_by_combo': {combo: _pair_summary(counts) for combo, counts in by_combo.items()},
    }


def _pair_rows(pairs: Sequence[Dict[str, Any]], prefix: str = '') -> List[str]:
    rows = []
    for p in pairs:
        lo, hi = p['ci']
        share = f"{100 * p['a_above_b']:.1f}% ({100 * lo:.1f}–{100 * hi:.1f})"
        rows.append(f"| {prefix}{p['a']} vs {p['b']} | {p['games']} | {share} |")
    return rows


def head_to_head_table(result: Dict[str, Any], title: Optional[str] = None) -> str:
    """Markdown summary of :func:`head_to_head`: pooled pairs, pairs per combination, per-cell win × n."""
    lines = [f'### {title}', ''] if title else []
    lines += [f"Filler seats: {result['filler']}. 'A above B' = share of games where A ranked higher.", '']
    lines += ['| pair | games | A above B (95% CI) |', '|---|---|---|', *_pair_rows(result['pairs'])]
    lines += ['', '| combination: pair | games | A above B (95% CI) |', '|---|---|---|']
    for combo, pairs in result['pairs_by_combo'].items():
        lines += _pair_rows(pairs, f'{combo}: ')
    labels = result['models']
    lines += ['', '| cell | ' + ' | '.join(f'{m} win × n' for m in labels) + ' |', '|---' * (len(labels) + 1) + '|']
    for c in result['cells']:
        models = c.get('models', {})
        cols = [f"{models[m]['win_rel']:.2f}" if m in models else '–' for m in labels]
        lines.append(f"| {c['cell']} | " + ' | '.join(cols) + ' |')
    return '\n'.join(lines)
