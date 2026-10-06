"""Data-driven values of the Leaders, for the scripted agents' draft and recruitment.

Most leaders pay off through cards a city builds *later* (``Tally`` points per card, resource
discounts, build income), so their worth cannot be read off the current position. Instead
:func:`estimate_leader_values` plays games in which every leader is drafted and recruited at
random (board agents otherwise) and regresses each player's final score margin (own score
minus the opponents' mean) on the leaders they recruited, with one intercept per table type.
Because recruitment is random, each coefficient estimates what recruiting that leader is worth,
in victory points of margin, cost included.

:data:`LEADER_VALUES` holds the result used by :class:`~wonders.agents.heuristic.HeuristicAgent`;
regenerate it with ``python -m wonders.agents.leader_values``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

# The estimation code imports the agents (which import this table) lazily, inside the functions.

Table = Tuple[Tuple[str, ...], int]

TABLES: Tuple[Table, ...] = tuple(
    (combo, n) for combo in (('base', 'leaders'), ('base', 'leaders', 'cities')) for n in (3, 4, 5, 6, 7)
)

# Estimated with estimate_leader_values(games=4000 per table, seed=0; 40,000 games, ~7-12k
# recruitments per leader, standard errors 0.10-0.13): VP of final score margin from recruiting
# the leader rather than using that recruitment for a wonder stage / 3 coins.
LEADER_VALUES: Dict[str, float] = {
    'Alexander': -0.54,
    'Amytis': 2.27,
    'Archimedes': -1.72,
    'Aristotle': -1.99,
    'Bilkis': 0.24,
    'Caesar': 2.36,
    'Cleopatra': 1.70,
    'Croesus': 0.35,
    'Euclid': 1.20,
    'Hammurabi': -0.99,
    'Hannibal': 0.16,
    'Hatshepsut': 0.37,
    'Hiram': -0.65,
    'Hypatia': -0.79,
    'Imhotep': -2.11,
    'Justinian': 1.03,
    'Leonidas': -1.15,
    'Maecenas': -1.64,
    'Midas': 2.81,
    'Nebuchadnezzar': -0.01,
    'Nefertiti': 0.74,
    'Nero': -1.13,
    'Pericles': 0.73,
    'Phidias': 0.14,
    'Plato': -0.11,
    'Praxiteles': -0.45,
    'Ptolemy': 1.31,
    'Pythagoras': 0.96,
    'Ramses': -0.23,
    'Sappho': -0.50,
    'Solomon': 1.34,
    'Tomyris': -0.91,
    'Varro': 0.34,
    'Vitruvius': -0.60,
    'Xenophon': -0.91,
    'Zenobia': 0.38,
}


def _game(job: Tuple[int, int, Tuple[str, ...], int]) -> List[Tuple[int, List[str], float]]:
    """One game of randomly-leadered board agents: per seat (table id, recruited leaders, margin)."""
    from wonders.agents.ablation import leader_ablation
    from wonders.agents.bench import AGENTS, _content
    from wonders.agents.match import play_match
    from wonders.cards.base import Color
    from wonders.engine.record import PlayerSetup

    index, seed, expansions, n = job
    content = _content(expansions)
    rng = random.Random(seed * 1_000_003 + index)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    spec = leader_ablation(AGENTS['board'], 'random')
    agents = [spec.build(random.Random(rng.random())) for _ in range(n)]
    game, _record = play_match(content, setups, agents, rng=rng, expansions=expansions)
    totals = [game.score(i).total for i in range(n)]
    out = []
    for i, player in enumerate(game.players):
        margin = totals[i] - (sum(totals) - totals[i]) / (n - 1)
        leaders = [c.name for c in player.cards if c.color is Color.WHITE]
        out.append((index, leaders, margin))
    return out


def estimate_leader_values(
    games: int = 4000,
    tables: Sequence[Table] = TABLES,
    seed: int = 0,
    processes: int = 1,
    ridge: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Per leader: ``value`` (VP of margin from recruiting it), its standard error and sample count."""
    import numpy as np

    from wonders._mp import get_context, init_worker

    jobs = [(g, seed + t, tuple(exp), n) for t, (exp, n) in enumerate(tables) for g in range(games)]
    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            results = pool.map(_game, jobs, chunksize=max(1, len(jobs) // (processes * 16)))
    else:
        results = [_game(job) for job in jobs]
    table_ids = {(exp, n): t for t, (exp, n) in enumerate(tables)}
    leaders = sorted({name for game in results for _i, names, _m in game for name in names})
    column = {name: k for k, name in enumerate(leaders)}
    rows, targets = [], []
    for job, game in zip(jobs, results):
        t = table_ids[(job[2], job[3])]
        for _index, names, margin in game:
            x = np.zeros(len(leaders) + len(tables))
            for name in names:
                x[column[name]] += 1.0
            x[len(leaders) + t] = 1.0  # per-table intercept
            rows.append(x)
            targets.append(margin)
    X, y = np.array(rows), np.array(targets)
    penalty = np.diag([ridge] * len(leaders) + [0.0] * len(tables))
    gram = X.T @ X + penalty
    beta = np.linalg.solve(gram, X.T @ y)
    residual = y - X @ beta
    sigma2 = float(residual @ residual) / max(1, len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(gram)) * sigma2)
    counts = X[:, : len(leaders)].sum(axis=0)
    return {
        name: {'value': float(beta[k]), 'se': float(se[k]), 'recruited': int(counts[k])} for name, k in column.items()
    }


if __name__ == '__main__':  # pragma: no cover - regeneration script
    import json

    values = estimate_leader_values(processes=10)
    for name, v in sorted(values.items(), key=lambda kv: -kv[1]['value']):
        print(f"{name:16s} {v['value']:+6.2f} ± {v['se']:.2f}  (recruited {v['recruited']})")
    print(json.dumps({k: round(v['value'], 2) for k, v in sorted(values.items())}, indent=1))
