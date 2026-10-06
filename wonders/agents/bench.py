"""Benchmark agents against each other over many seeded games.

:func:`run_bench` plays a *lineup* (one agent per seat) for ``n_games`` games and
reports, per agent, the win rate with a 95% Wilson confidence interval, the mean
final score and rank, and the mean of every scoring category. To cancel seat and
wonder bias, the lineup rotates one seat per game and each game deals fresh random
wonders. Each game's randomness derives only from ``(seed, game index)``, so a
benchmark is reproducible and gives identical results with any number of processes.

Agents are described by picklable :class:`AgentSpec` objects (so games can run in
worker processes); :data:`AGENTS` names the built-in ones.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from wonders._mp import get_context, init_worker
from wonders.agents.adaptive import AdaptiveAgent, RivalAgent
from wonders.agents.base import Agent
from wonders.agents.baselines import (
    BuilderAgent,
    CivicAgent,
    DiscardAgent,
    EconomistAgent,
    GreedyAgent,
    MilitaristAgent,
    RandomAgent,
    ScientistAgent,
)
from wonders.agents.board import BoardAwareAgent, BoardFocusAgent
from wonders.agents.heuristic import HeuristicAgent
from wonders.agents.match import play_match
from wonders.agents.strategies import BalancedAgent, BlockerAgent, OpportunistAgent, TableAgent
from wonders.cards.content import Content, load_content
from wonders.engine.record import PlayerSetup

MIN_PLAYERS, MAX_PLAYERS = 3, 7  # the base deck has cards for 3-7 players

CATEGORIES = ('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science', 'leaders', 'cities')


@dataclass(frozen=True)
class AgentSpec:
    """A picklable recipe for an agent.

    ``seeded`` agents receive the per-game ``rng`` as their ``rng`` argument, so
    stochastic agents stay reproducible. ``kwargs`` are passed through (e.g.
    ``HeuristicAgent`` weights).
    """

    name: str
    cls: Type[Agent]
    kwargs: Tuple[Tuple[str, Any], ...] = ()
    seeded: bool = False

    @classmethod
    def of(cls, spec_name: str, agent_cls: Type[Agent], /, seeded: bool = False, **kwargs: Any) -> 'AgentSpec':
        return cls(spec_name, agent_cls, tuple(sorted(kwargs.items())), seeded)

    def build(self, rng: random.Random) -> Agent:
        kwargs = dict(self.kwargs)
        if self.seeded:
            kwargs['rng'] = rng
        return self.cls(**kwargs)  # type: ignore[call-arg]


AGENTS: Dict[str, AgentSpec] = {
    spec.name: spec
    for spec in (
        AgentSpec.of('random', RandomAgent, seeded=True),
        AgentSpec.of('greedy', GreedyAgent),
        AgentSpec.of('hoarder', DiscardAgent),
        AgentSpec.of('militarist', MilitaristAgent),
        AgentSpec.of('scientist', ScientistAgent),
        AgentSpec.of('civic', CivicAgent),
        AgentSpec.of('economist', EconomistAgent),
        AgentSpec.of('builder', BuilderAgent),
        AgentSpec.of('heuristic', HeuristicAgent),
        AgentSpec.of('opportunist', OpportunistAgent, seeded=True),
        AgentSpec.of('blocker', BlockerAgent),
        AgentSpec.of('table', TableAgent),
        AgentSpec.of('balanced', BalancedAgent),
        AgentSpec.of('board', BoardAwareAgent),
        AgentSpec.of('board-focus', BoardFocusAgent),
        AgentSpec.of('adaptive', AdaptiveAgent),
        AgentSpec.of('rival', RivalAgent),
    )
}


def wilson_interval(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a win rate (well-behaved near 0 and 1)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class AgentStats:
    """Aggregated results of one agent (summed over every seat it occupied)."""

    name: str
    games: int = 0
    wins: int = 0
    score_total: int = 0
    rank_total: int = 0
    categories: Dict[str, int] = field(default_factory=lambda: {c: 0 for c in CATEGORIES})

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def win_interval(self) -> Tuple[float, float]:
        return wilson_interval(self.wins, self.games)

    @property
    def mean_score(self) -> float:
        return self.score_total / self.games if self.games else 0.0

    @property
    def mean_rank(self) -> float:
        return self.rank_total / self.games if self.games else 0.0

    def mean_category(self, category: str) -> float:
        return self.categories[category] / self.games if self.games else 0.0


@dataclass
class BenchResult:
    n_games: int
    n_players: int
    lineup: List[str]
    stats: Dict[str, AgentStats]
    ranks: List[List[Tuple[str, int]]] = field(default_factory=list)  # per game: (agent name, rank) per seat

    def table(self) -> str:
        """A fixed-width text report, best win rate first."""
        cats = ' '.join(f'{c[:5]:>5}' for c in CATEGORIES)
        lines = [
            f'{self.n_games} games, {self.n_players} players, lineup: {", ".join(self.lineup)}',
            f'{"agent":<14} {"seats":>5} {"win%":>6} {"95% CI":>13} {"score":>6} {"rank":>5}  {cats}',
        ]
        for s in sorted(self.stats.values(), key=lambda s: -s.win_rate):
            lo, hi = s.win_interval
            cat_means = ' '.join(f'{s.mean_category(c):5.1f}' for c in CATEGORIES)
            lines.append(
                f'{s.name:<14} {s.games:>5} {100 * s.win_rate:6.1f} {100 * lo:5.1f}-{100 * hi:5.1f}%'
                f' {s.mean_score:6.1f} {s.mean_rank:5.2f}  {cat_means}'
            )
        return '\n'.join(lines)


Seat = Union[str, AgentSpec]
# 'rotate': the lineup's order rotates one seat per game (neighbours stay fixed);
# 'shuffle': a random seating per game (neighbours vary too).
SEATINGS = ('rotate', 'shuffle')

# Per-process content cache (loading JSON once per worker, not once per game).
_CONTENT: Dict[Tuple[str, ...], Content] = {}


def _content(expansions: Tuple[str, ...]) -> Content:
    if expansions not in _CONTENT:
        _CONTENT[expansions] = load_content(*expansions)
    return _CONTENT[expansions]


def _play_one(
    job: Tuple[int, int, Tuple[AgentSpec, ...], Tuple[str, ...], str, str],
) -> List[Tuple[str, int, int, Dict]]:
    """Play game ``index``; return per-seat (agent name, rank, won, score breakdown)."""
    index, seed, lineup, expansions, sides, seating = job
    content = _content(expansions)
    rng = random.Random(seed * 1_000_003 + index)
    n = len(lineup)
    if seating == 'shuffle':  # random seating (own rng, so the deal is the same as with rotation)
        seats = list(lineup)
        random.Random(seed * 7_919 + index).shuffle(seats)
    else:
        seats = [lineup[(i + index) % n] for i in range(n)]  # rotate one seat per game
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [
        PlayerSetup(spec.name, wonder, rng.choice('AB') if sides == 'random' else sides)
        for spec, wonder in zip(seats, rng.sample(names, n))
    ]
    agents = [spec.build(random.Random(rng.random())) for spec in seats]
    game, _record = play_match(content, setups, agents, rng=rng, expansions=expansions)
    breakdowns = [game.score(i).as_dict() for i in range(n)]
    winner = game.winner()
    order = sorted(range(n), key=lambda i: (-breakdowns[i]['total'], -game.players[i].coins))
    rank = {seat: r + 1 for r, seat in enumerate(order)}
    return [(seats[i].name, rank[i], int(i == winner), breakdowns[i]) for i in range(n)]


def run_bench(
    lineup: Sequence[Seat],
    n_games: int = 200,
    seed: int = 0,
    expansions: Sequence[str] = ('base',),
    sides: str = 'random',
    processes: int = 1,
    context: Optional[str] = None,
    seating: str = 'rotate',
) -> BenchResult:
    """Play ``lineup`` (one agent per seat; names from :data:`AGENTS` or specs) ``n_games`` times.

    The same agent may fill several seats; its stats are pooled. With
    ``processes > 1`` games run in a multiprocessing pool (results are identical).
    """
    specs = tuple(AGENTS[s] if isinstance(s, str) else s for s in lineup)
    if not MIN_PLAYERS <= len(specs) <= MAX_PLAYERS:
        raise ValueError(f'need {MIN_PLAYERS}-{MAX_PLAYERS} seats, got {len(specs)}')
    if seating not in SEATINGS:
        raise ValueError(f'seating must be one of {SEATINGS}, got {seating!r}')
    jobs = [(g, seed, specs, tuple(expansions), sides, seating) for g in range(n_games)]
    if processes > 1:
        with get_context(context).Pool(processes, initializer=init_worker) as pool:
            results = pool.map(_play_one, jobs, chunksize=max(1, n_games // (processes * 8)))
    else:
        results = [_play_one(job) for job in jobs]
    stats: Dict[str, AgentStats] = {spec.name: AgentStats(spec.name) for spec in specs}
    ranks: List[List[Tuple[str, int]]] = []
    for game in results:
        ranks.append([(name, rank) for name, rank, _won, _breakdown in game])
        for name, rank, won, breakdown in game:
            s = stats[name]
            s.games += 1
            s.wins += won
            s.rank_total += rank
            s.score_total += breakdown['total']
            for c in CATEGORIES:
                s.categories[c] += breakdown[c]
    return BenchResult(n_games, len(specs), [spec.name for spec in specs], stats, ranks)
