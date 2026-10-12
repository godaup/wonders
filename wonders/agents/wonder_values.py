"""Data-driven strength of the wonder board sides, per table type (expansions and player count).

A side's strength is the expected victory points of final score margin (own score minus the
opponents' mean) from playing it instead of an average side of the same table. It depends on the
policy that plays the boards, so every seat is played by the same agent (``board`` by default:
:class:`~wonders.agents.board.BoardAwareAgent` values the board powers explicitly).

:func:`estimate_wonder_values` deals boards and sides at random and plays every deal ``rotations``
times with the same decks and the board assignment rotated around the table (common random numbers:
a deal's luck is shared by every side dealt in it). It then fits

    margin_i = sum_s c_is * (beta_s + delta_st) + e_i

where ``c_is`` is 1 for the seat's own side and ``-1/(n-1)`` for each opponent's, ``beta`` is a side's
main effect and ``delta`` its deviation at table ``t``. The opponent terms keep a side's estimate from
being diluted by the boards it happened to face. ``delta`` is shrunk towards 0 with a ridge penalty
``sigma^2 / tau^2`` (partial pooling); with ``pooling='auto'`` the spread ``tau`` of the table
deviations is estimated from the data (one empirical Bayes step), so a side keeps a table-specific
value only where the data show one. Values are centred on each table's sides (a constant is not
identified: margins sum to 0), and standard errors are cluster-robust by deal (the seats of a game
and the rotations of a deal share their noise).

Regenerate the report with ``python -m wonders.agents.wonder_values`` (``--help`` for the options).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

# The estimation code imports the agents and numpy lazily, inside the functions.

Table = Tuple[Tuple[str, ...], int]
Side = str  # '<wonder> <A|B>'


def side_name(wonder: str, side: str) -> Side:
    """The key of a board side, e.g. ``'The Pyramids of Giza B'``."""
    return f'{wonder} {side}'


def table_name(table: Table) -> str:
    """A table's label, e.g. ``'base+leaders/5p'``."""
    expansions, n = table
    return f"{'+'.join(expansions)}/{n}p"


def tables_for(combos: Sequence[Sequence[str]], players: Sequence[int] = (3, 4, 5, 6, 7)) -> Tuple[Table, ...]:
    """Every combination of an expansion set and a player count."""
    return tuple((tuple(combo), n) for combo in combos for n in players)


BASE_TABLES: Tuple[Table, ...] = tables_for([('base',)])
TABLES: Tuple[Table, ...] = tables_for(
    [
        ('base',),
        ('base', 'leaders'),
        ('base', 'cities'),
        ('base', 'leaders', 'cities'),
        ('base', 'wonderpack'),
        ('base', 'leaders', 'cities', 'wonderpack'),
    ]
)  # every bundled board (Abu Simbel needs Leaders) and the expansions' effect on the base boards

# One game of a deal: per seat (side, final score margin).
Seat = Tuple[Side, float]


def _deal(job: Tuple[int, int, Tuple[str, ...], int, int, str]) -> List[List[Seat]]:
    """Deal ``index``: the same decks played once per rotation of the board assignment."""
    from wonders.agents.bench import AGENTS, _content
    from wonders.agents.match import play_match
    from wonders.engine.record import PlayerSetup

    index, seed, expansions, n, rotations, agent = job
    content = _content(expansions)
    rng = random.Random(seed * 1_000_003 + index)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    boards = [(w, rng.choice('AB')) for w in rng.sample(names, n)]
    game_seed = rng.random()
    games = []
    for r in range(rotations):
        shift = r * n // rotations
        dealt = boards[shift:] + boards[:shift]
        setups = [PlayerSetup(f'p{i}', w, s) for i, (w, s) in enumerate(dealt)]
        game_rng = random.Random(game_seed)  # same decks (and agent seeds) in every rotation
        agents = [AGENTS[agent].build(random.Random(game_rng.random())) for _ in range(n)]
        game, _record = play_match(content, setups, agents, rng=game_rng, expansions=expansions)
        totals = [game.score(i).total for i in range(n)]
        mean = sum(totals) / n
        # own minus the opponents' mean = (own - table mean) * n / (n - 1)
        games.append([(side_name(w, s), (totals[i] - mean) * n / (n - 1)) for i, (w, s) in enumerate(dealt)])
    return games


@dataclass
class SideValue:
    """A side's estimated strength: VP of margin vs the average side, its standard error, games played."""

    value: float
    se: float
    games: int


@dataclass
class WonderValues:
    """The fit of :func:`estimate_wonder_values`."""

    main: Dict[Side, SideValue]  # pooled over the tables (each side's average deviation)
    tables: Dict[str, Dict[Side, SideValue]]  # per table: main effect + the table's deviation
    sigma: float  # residual SD of a seat's margin (VP)
    tau: float  # spread of the table deviations (VP) used for the pooling
    games: int
    agent: str
    rotations: int

    def as_dict(self) -> Dict:
        """JSON-ready form."""

        def sides(values: Dict[Side, SideValue]) -> Dict:
            return {s: {'value': round(v.value, 3), 'se': round(v.se, 3), 'games': v.games} for s, v in values.items()}

        return {
            'agent': self.agent,
            'games': self.games,
            'rotations': self.rotations,
            'sigma': round(self.sigma, 3),
            'tau': round(self.tau, 3),
            'main': sides(self.main),
            'tables': {t: sides(v) for t, v in self.tables.items()},
        }

    def markdown(self) -> str:
        """The main effects and every table as Markdown tables (sides sorted by main effect)."""
        order = sorted(self.main, key=lambda s: -self.main[s].value)
        lines = [
            f'Agent `{self.agent}`, {self.games} games ({self.rotations} rotations per deal); '
            f'residual SD {self.sigma:.1f} VP, table spread tau {self.tau:.2f} VP.',
            '',
            '| side | main | ' + ' | '.join(self.tables) + ' |',
            '|---|---:|' + '---:|' * len(self.tables),
        ]
        for s in order:
            cells = [f'{self.main[s].value:+.2f} ± {self.main[s].se:.2f}']
            for values in self.tables.values():
                v = values.get(s)
                cells.append('' if v is None else f'{v.value:+.2f} ± {v.se:.2f}')
            lines.append(f'| {s} | ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)


def estimate_wonder_values(
    games: int = 2000,
    tables: Sequence[Table] = BASE_TABLES,
    seed: int = 0,
    rotations: int = 2,
    agent: str = 'board',
    pooling: Union[str, float] = 'auto',
    processes: int = 1,
) -> WonderValues:
    """Estimate every board side's strength per table (see the module docstring).

    Args:
        games: Games per table (``games // rotations`` deals, each played ``rotations`` times).
        tables: ``(expansions, players)`` table types.
        seed: Seed of the deals (table ``t`` uses ``seed + t``).
        rotations: Games per deal, with the board assignment rotated around the table.
        agent: :data:`~wonders.agents.bench.AGENTS` id that plays every seat.
        pooling: Ridge penalty on the table deviations, or ``'auto'`` (``sigma^2 / tau^2`` estimated).
        processes: Worker processes.
    """
    from wonders._mp import get_context, init_worker

    if not 1 <= rotations <= min(n for _e, n in tables):
        raise ValueError('rotations must be between 1 and the smallest player count')
    deals = max(1, games // rotations)
    jobs = [(d, seed + t, tuple(exp), n, rotations, agent) for t, (exp, n) in enumerate(tables) for d in range(deals)]
    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            results = pool.map(_deal, jobs, chunksize=max(1, len(jobs) // (processes * 16)))
    else:
        results = [_deal(job) for job in jobs]
    table_of = [t for t in range(len(tables)) for _d in range(deals)]
    return fit_wonder_values(list(zip(table_of, results)), tables, pooling, agent, rotations)


def fit_wonder_values(
    deals: Sequence[Tuple[int, List[List[Seat]]]],
    tables: Sequence[Table],
    pooling: Union[str, float] = 'auto',
    agent: str = '',
    rotations: int = 1,
) -> WonderValues:
    """Fit the model of the module docstring to ``deals``: (table index, games of (side, margin) per seat)."""
    import numpy as np

    sides = sorted({s for _t, deal in deals for game in deal for s, _m in game})
    k = len(sides)
    column = {s: i for i, s in enumerate(sides)}
    n_tables = len(tables)
    p = k * (1 + n_tables)  # main effects, then one block of deviations per table
    # Sparse design: a row has 2n nonzeros (own +1, opponents -1/(n-1), main and table block).
    rows: List[Tuple['np.ndarray', 'np.ndarray', float, int]] = []  # (indices, coefficients, margin, deal)
    gram0, Xy = np.zeros((p, p)), np.zeros(p)
    counts = np.zeros((n_tables, k), dtype=int)  # games per side and table
    for cluster, (t, deal) in enumerate(deals):
        for game in deal:
            n = len(game)
            cols = np.array([column[s] for s, _m in game])
            idx = np.concatenate([cols, k * (1 + t) + cols])
            for i, (_own, margin) in enumerate(game):
                coef = np.full(n, -1.0 / (n - 1))
                coef[i] = 1.0
                coef2 = np.concatenate([coef, coef])
                gram0[np.ix_(idx, idx)] += np.outer(coef2, coef2)
                Xy[idx] += coef2 * margin
                rows.append((idx, coef2, margin, cluster))
            counts[t, cols] += 1

    def residuals(beta: 'np.ndarray') -> 'np.ndarray':
        return np.array([margin - coef @ beta[idx] for idx, coef, margin, _c in rows])

    def fit(lam_dev: float) -> Tuple['np.ndarray', 'np.ndarray', float]:
        # A weak penalty on the main effects only identifies the constant (margins sum to 0).
        gram = gram0 + np.diag([1.0] * k + [lam_dev] * (k * n_tables))
        beta = np.linalg.solve(gram, Xy)
        resid = residuals(beta)
        return beta, gram, float(resid @ resid) / max(1, len(rows) - p)

    def table_contrasts(t: int) -> Tuple['np.ndarray', 'np.ndarray']:
        """The sides dealt at table ``t`` and the rows giving their values there, centred on the table."""
        present = np.flatnonzero(counts[t])
        W = np.zeros((len(present), p))
        W[np.arange(len(present)), present] = 1.0
        W[np.arange(len(present)), k * (1 + t) + present] = 1.0
        return present, W - W.mean(axis=0)

    if pooling == 'auto' and n_tables > 1:
        # Empirical Bayes (method of moments): fit every table freely, then tau^2 = the spread of a
        # side's table values around its mean minus the part the sampling noise explains.
        beta, gram, sigma2 = fit(1e-3)
        inv = np.linalg.inv(gram)
        theta, var = np.full((n_tables, k), np.nan), np.full((n_tables, k), np.nan)
        for t in range(n_tables):
            present, C = table_contrasts(t)
            theta[t, present] = C @ beta
            var[t, present] = np.einsum('ij,jk,ik->i', C, inv, C) * sigma2
        spread, noise, m = 0.0, 0.0, 0
        for i in range(k):
            dealt = np.flatnonzero(counts[:, i])
            if len(dealt) < 2:
                continue
            dev = theta[dealt, i] - theta[dealt, i].mean()
            spread += float(dev @ dev) / (len(dealt) - 1) * len(dealt)  # unbiased variance, per value
            noise += float(var[dealt, i].sum())
            m += len(dealt)
        tau2 = max((spread - noise) / max(m, 1), 1e-4)
        lam_dev = sigma2 / tau2
    elif pooling == 'auto':
        lam_dev = 1e6  # one table: its deviations and the main effects coincide
    else:
        lam_dev = float(pooling)
    beta, gram, sigma2 = fit(lam_dev)
    tau = (sigma2 / lam_dev) ** 0.5

    # Cluster-robust (sandwich) covariance, clustered by deal.
    inv = np.linalg.inv(gram)
    resid = residuals(beta)
    scores = np.zeros((len(deals), p))
    for (idx, coef, _m, cluster), e in zip(rows, resid):
        scores[cluster, idx] += coef * e
    cov = inv @ (scores.T @ scores) @ inv

    def estimates(C: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        return C @ beta, np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', C, cov, C), 0.0))

    per_table: Dict[str, Dict[Side, SideValue]] = {}
    for t, table in enumerate(tables):
        present, C = table_contrasts(t)
        values, ses = estimates(C)
        per_table[table_name(table)] = {
            sides[i]: SideValue(float(v), float(se), int(counts[t, i])) for i, v, se in zip(present, values, ses)
        }
    # Main effect: a side's mean value over the tables it was dealt at, centred on all sides.
    W = np.zeros((k, p))
    for i in range(k):
        W[i, i] = 1.0
        dealt = np.flatnonzero(counts[:, i])
        W[i, k * (1 + dealt) + i] = 1.0 / len(dealt)
    values, ses = estimates(W - W.mean(axis=0))
    main = {s: SideValue(float(values[i]), float(ses[i]), int(counts[:, i].sum())) for i, s in enumerate(sides)}
    n_games = sum(len(deal) for _t, deal in deals)
    return WonderValues(main, per_table, sigma2**0.5, tau, n_games, agent, rotations)


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - regeneration script
    """Command line: estimate and print the Markdown report (and optionally save the JSON)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0] if __doc__ else None)
    parser.add_argument('-e', '--expansions', action='append', help="combo like 'base,leaders' (repeatable)")
    parser.add_argument('--all', action='store_true', help='every table of TABLES')
    parser.add_argument('-n', '--players', default='3,4,5,6,7', help='player counts, comma separated')
    parser.add_argument('-g', '--games', type=int, default=2000, help='games per table')
    parser.add_argument('-r', '--rotations', type=int, default=2)
    parser.add_argument('-a', '--agent', default='board')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-p', '--processes', type=int, default=1)
    parser.add_argument('-o', '--output', help='write the fit as JSON')
    args = parser.parse_args(argv)
    combos = [tuple(c.split(',')) for c in args.expansions] if args.expansions else [('base',)]
    tables = TABLES if args.all else tables_for(combos, [int(n) for n in args.players.split(',')])
    values = estimate_wonder_values(args.games, tables, args.seed, args.rotations, args.agent, processes=args.processes)
    print(values.markdown())
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(values.as_dict(), f, indent=1)


if __name__ == '__main__':  # pragma: no cover - regeneration script
    main()
