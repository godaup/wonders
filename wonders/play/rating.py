"""Elo ratings: fixed ratings for the agents, and the human's rating derived from their games.

A game of 3–7 players is scored as one pairwise result against every opponent — ranked above
(score, then coins) is a win, an exact tie a draw — with the usual Elo expectation
``1 / (1 + 10 ** ((opponent - player) / 400))``. The agents never change their rating:
:data:`AGENT_ELO` was fitted by maximum likelihood on that same pairwise model over thousands of
simulated games between random lineups of agents (all expansion combinations, 3–7 players), so a
human's rating is on the same scale. :func:`player_history` replays the human's finished games in
order and returns the rating after each of them.

Regenerate the table with ``python -m wonders.play.rating`` (needs PyTorch for the network).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Fitted with calibrate(games=20000, seed=0) — about 5,500 games per agent, standard errors ~10 points;
# ``random`` is anchored at 1000.
AGENT_ELO: Dict[str, int] = {
    'neural': 2165,
    'adaptive': 1914,
    'rival': 1899,
    'board': 1885,
    'board-focus': 1880,
    'heuristic': 1727,
    'scientist': 1651,
    'greedy': 1631,
    'civic': 1623,
    'builder': 1599,
    'economist': 1562,
    'opportunist': 1561,
    'table': 1550,
    'balanced': 1542,
    'militarist': 1510,
    'blocker': 1279,
    'random': 1000,
    'hoarder': 732,
}

ANCHOR = ('random', 1000.0)
DEFAULT_AGENT_ELO = 1500  # agents without a fitted rating (extensions, extra checkpoints)
START = 1500  # a new player's rating
K_PROVISIONAL, K = 48.0, 24.0  # rating change per game: larger for the first PROVISIONAL games
PROVISIONAL = 10


def agent_elo(agent: str) -> int:
    """The fixed rating of an agent id (``neural`` is the bundled network)."""
    return AGENT_ELO.get(agent, DEFAULT_AGENT_ELO)


def expected(rating: float, opponent: float) -> float:
    """Expected score (win = 1, draw = ½) against one opponent."""
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def pairwise(places: Sequence[int], seat: int) -> List[Tuple[int, float]]:
    """``(opponent seat, result)`` for ``seat``: 1 ranked above, ½ tied, 0 below.

    ``places`` gives each seat's place (1 = first); tied seats share a place.
    """
    return [
        (other, 1.0 if places[seat] < places[other] else 0.5 if places[seat] == places[other] else 0.0)
        for other in range(len(places))
        if other != seat
    ]


def places(totals: Sequence[float], coins: Sequence[float]) -> List[int]:
    """Each seat's place by score, then coins; exact ties share the better place."""
    keys = [(t, c) for t, c in zip(totals, coins)]
    return [1 + sum(other > key for other in keys) for key in keys]


@dataclass(frozen=True)
class RatedGame:
    """One finished game, as the rating needs it."""

    id: str
    places: Tuple[int, ...]
    seat: int  # the human
    opponents: Tuple[str, ...]  # agent id per seat ('' for the human)


@dataclass(frozen=True)
class RatingPoint:
    """The human's rating after one game."""

    id: str
    before: float
    after: float
    games: int  # rated games so far, this one included


def update(rating: float, game: RatedGame, games_played: int) -> float:
    """The rating after ``game``; the K-factor is shared between the opponents."""
    results = pairwise(game.places, game.seat)
    k = (K_PROVISIONAL if games_played < PROVISIONAL else K) / len(results)
    return rating + k * sum(score - expected(rating, agent_elo(game.opponents[o])) for o, score in results)


def rated_game(saved: Dict[str, Any]) -> Optional[RatedGame]:
    """A finished saved game (:meth:`~wonders.play.session.HumanSession.saved`) as a rated game."""
    result = saved.get('result')
    if not saved.get('finished') or not result:
        return None
    agents = tuple('' if p['agent'] == 'human' else str(p['agent']) for p in saved['players'])
    return RatedGame(str(saved['id']), tuple(result['places']), int(saved['human']), agents)


def player_history(games: Iterable[RatedGame], start: float = START) -> List[RatingPoint]:
    """The human's rating after each game (``games`` in the order they were played)."""
    rating, out = start, []
    for n, game in enumerate(games):
        after = update(rating, game, n)
        out.append(RatingPoint(game.id, rating, after, n + 1))
        rating = after
    return out


# -- calibration -----------------------------------------------------------------------------------
COMBOS: Tuple[Tuple[str, ...], ...] = (
    ('base',),
    ('base', 'leaders'),
    ('base', 'cities'),
    ('base', 'leaders', 'cities'),
)


def _pool(neural: bool) -> List[str]:
    from wonders.agents.bench import AGENTS

    return [*AGENTS, 'neural'] if neural else list(AGENTS)


def _game(job: Tuple[int, int, bool]) -> Tuple[List[str], List[int]]:
    """One game between a random lineup: (agent per seat, place per seat)."""
    from wonders.agents.base import Agent
    from wonders.agents.bench import AGENTS, _content
    from wonders.agents.match import play_match
    from wonders.engine.record import PlayerSetup

    index, seed, neural = job
    rng = random.Random(seed * 1_000_003 + index)
    expansions = rng.choice(COMBOS)
    lineup = rng.sample(_pool(neural), rng.randint(3, 7))
    content = _content(expansions)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [PlayerSetup(a, w, rng.choice('AB')) for a, w in zip(lineup, rng.sample(names, len(lineup)))]
    agents: List[Agent] = []
    for name in lineup:
        agent_rng = random.Random(rng.random())
        if name == 'neural':
            from wonders.ml import NeuralAgent, pretrained_path

            agents.append(NeuralAgent(pretrained_path(), rng=agent_rng, name='neural'))
        else:
            agents.append(AGENTS[name].build(agent_rng))
    game, _record = play_match(content, setups, agents, rng=rng, expansions=expansions)
    totals = [game.score(i).total for i in range(len(lineup))]
    return lineup, places(totals, [p.coins for p in game.players])


def fit(results: Sequence[Tuple[Sequence[str], Sequence[int]]], iterations: int = 500) -> Dict[str, float]:
    """Maximum-likelihood Elo ratings (Bradley–Terry with draws as half wins), anchored at :data:`ANCHOR`."""
    wins: Dict[str, float] = {}
    meetings: Dict[Tuple[str, str], int] = {}
    for lineup, place in results:
        for seat, name in enumerate(lineup):
            wins.setdefault(name, 0.0)
            for other, score in pairwise(place, seat):
                wins[name] += score
                pair = (name, lineup[other])
                meetings[pair] = meetings.get(pair, 0) + 1
    opponents: Dict[str, List[Tuple[str, int]]] = {name: [] for name in wins}
    for (a, b), n in meetings.items():
        opponents[a].append((b, n))
    # A weak prior keeps agents that never win finite: one win and one loss against a fixed reference.
    for name in wins:
        wins[name] += 1.0
        opponents[name].append(('', 2))
    strength = {name: 1.0 for name in wins}
    for _ in range(iterations):  # the minorization-maximization update (Hunter 2004)
        fixed = {**strength, '': 1.0}
        strength = {
            name: wins[name] / sum(n / (fixed[name] + fixed[b]) for b, n in opponents[name]) for name in strength
        }
    anchor, value = ANCHOR
    base = math.log10(strength[anchor]) if anchor in strength else 0.0
    return {name: value + 400.0 * (math.log10(s) - base) for name, s in strength.items()}


def calibrate(
    games: int = 20000, seed: int = 0, processes: int = 1, neural: bool = True
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Play ``games`` random-lineup games; return the fitted ratings and each agent's game count."""
    from wonders._mp import get_context, init_worker

    jobs = [(g, seed, neural) for g in range(games)]
    if processes > 1:
        with get_context('spawn' if neural else None).Pool(processes, initializer=init_worker) as pool:
            results = pool.map(_game, jobs, chunksize=max(1, games // (processes * 16)))
    else:
        results = [_game(job) for job in jobs]
    counts: Dict[str, int] = {}
    for lineup, _place in results:
        for name in lineup:
            counts[name] = counts.get(name, 0) + 1
    return fit(results), counts


def _main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - regeneration script
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Fit the agents\' Elo ratings by simulation.')
    parser.add_argument('--games', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--processes', type=int, default=10)
    parser.add_argument('--no-neural', action='store_true')
    args = parser.parse_args(argv)
    ratings, counts = calibrate(args.games, args.seed, args.processes, not args.no_neural)
    for name, value in sorted(ratings.items(), key=lambda kv: -kv[1]):
        print(f'{name:14s} {value:7.1f}  ({counts[name]} games)')
    print(json.dumps({k: round(v) for k, v in sorted(ratings.items(), key=lambda kv: -kv[1])}, indent=4))


if __name__ == '__main__':  # pragma: no cover
    _main()
