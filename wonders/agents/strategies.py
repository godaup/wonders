"""Characterful strategy agents for varied self-play and as opponents.

All read only the :class:`~wonders.engine.observation.Observation`, so they respect
partial observability (opponents' hands are hidden; their tableaus are public).

* :class:`OpportunistAgent` — commits to a focus, then switches to a fresh one every
  few turns.
* :class:`BlockerAgent` — denies the neighbour who will receive this hand the card
  they most want, by taking it (building) or trashing it (discarding).
* :class:`TableAgent` — builds the best affordable card by a fixed value lookup table.
* :class:`BalancedAgent` — builds toward an even spread of card colours.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from wonders.agents.base import Agent
from wonders.agents.baselines import _builds, _card_points, _first
from wonders.cards.base import Card
from wonders.cards.content import load_content
from wonders.effects.library import (
    Coins,
    Debt,
    Diplomacy,
    Military,
    MultiProduction,
    Production,
    Science,
    ScienceWild,
    Tally,
    Trade,
    VictoryPoints,
)
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import CardView, CityView, Observation

# Named colour focuses an agent can pursue.
_ARCHETYPES: List[Tuple[str, frozenset]] = [
    ('military', frozenset({'red'})),
    ('science', frozenset({'green'})),
    ('civic', frozenset({'blue'})),
    ('economy', frozenset({'yellow', 'brown', 'grey'})),
    ('guild', frozenset({'purple'})),
]


class OpportunistAgent(Agent):
    """Pursues one colour focus, switching to a *different* one every ``switch_every`` turns."""

    name = 'opportunist'

    def __init__(self, rng: Optional[random.Random] = None, switch_every: int = 3):
        self.rng = rng or random.Random()
        self.switch_every = max(1, switch_every)
        self._moves = 0
        self._focus = self.rng.choice(_ARCHETYPES)[1]

    def _reroll(self) -> None:
        choices = [c for name, c in _ARCHETYPES if c != self._focus]
        self._focus = self.rng.choice(choices)

    def decide(self, obs: Observation) -> Action:
        if self._moves and self._moves % self.switch_every == 0:
            self._reroll()
        self._moves += 1
        views = {c.name: c for c in obs.hand}
        builds = _builds(obs)
        preferred = [a for a in builds if views[a.card.name].color in self._focus]
        pool = preferred or builds
        if pool:
            return max(pool, key=lambda a: _card_points(views[a.card.name]))
        return _first(obs, ActionKind.WONDER) or _first(obs, ActionKind.DISCARD)


def _num(value: object) -> float:
    """Effect-summary values are loosely typed; read numbers safely."""
    return float(value) if isinstance(value, (int, float)) else 0.0


def _appeal_to(card: CardView, neighbour: CityView) -> float:
    """Rough estimate of how much ``neighbour`` would want ``card`` (for denial)."""
    appeal = float(neighbour.cards_by_color.get(card.color, 0))  # fits their current focus
    for e in card.effects:
        appeal += _num(e.get('victory_points')) * 0.5
        appeal += _num(e.get('shields')) * 1.5  # escalates a military neighbour
        if e.get('science'):
            appeal += 3.0 + sum(neighbour.science.values())  # science compounds hard
        if e.get('science_wild'):
            appeal += 3.0
    return appeal


class BlockerAgent(Agent):
    """Aggressive: deny the downstream neighbour their most-wanted card.

    Ages I and III pass hands left, Age II passes right, so whichever neighbour will
    receive this hand is the target. The agent takes that neighbour's most-wanted card
    if it can build it, otherwise discards it to remove it from the game entirely.
    """

    name = 'blocker'

    def decide(self, obs: Observation) -> Action:
        views = {c.name: c for c in obs.hand}
        target = obs.left if obs.age in (1, 3) else obs.right
        best_name = max(views, key=lambda n: _appeal_to(views[n], target))
        build = next(
            (a for a in obs.legal_actions if a.kind is ActionKind.BUILD and a.card.name == best_name),
            None,
        )
        if build is not None:
            return build  # take it for yourself — they never see it
        return next(
            a for a in obs.legal_actions if a.kind is ActionKind.DISCARD and a.card.name == best_name
        )  # trash it so nobody gets it


def _rate(card: Card) -> float:
    """A fixed, state-independent power rating for a card (for the lookup table)."""
    value = 0.0
    for e in card.effects:
        if isinstance(e, VictoryPoints):
            value += e.points
        elif isinstance(e, Military):
            value += e.shields_count * 1.5
        elif isinstance(e, (Science, ScienceWild)):
            value += 3.5
        elif isinstance(e, Coins):
            value += e.amount * 0.33
        elif isinstance(e, (Production, MultiProduction)):
            value += sum(1.0 for _ in e.production())
        elif isinstance(e, Trade):
            value += 1.0
        elif isinstance(e, Tally):
            value += e.vp_each * 2.0 + e.coins_each * 0.66
        elif isinstance(e, Debt):
            value += e.amount * 0.5
        elif isinstance(e, Diplomacy):
            value += e.count * 1.5
    value += len(card.chain_enables) * 0.7  # tempo: future free builds
    return value


@lru_cache(maxsize=None)
def default_card_values() -> Dict[str, float]:
    """A lookup table of fixed card values across all card-based expansions."""
    content = load_content('base', 'leaders', 'cities')
    return {card.name: _rate(card) for card in content.cards}


class TableAgent(Agent):
    """Builds the highest-rated affordable card by a fixed value lookup table."""

    name = 'table'

    def __init__(self, values: Optional[Dict[str, float]] = None):
        self.values = values if values is not None else default_card_values()

    def decide(self, obs: Observation) -> Action:
        builds = _builds(obs)
        if builds:
            return max(builds, key=lambda a: self.values.get(a.card.name, 0.0))
        return _first(obs, ActionKind.WONDER) or _first(obs, ActionKind.DISCARD)


class BalancedAgent(Agent):
    """Builds toward an even spread of card colours (fills the colour it has least of)."""

    name = 'balanced'

    def decide(self, obs: Observation) -> Action:
        views = {c.name: c for c in obs.hand}
        builds = _builds(obs)
        if builds:
            counts = obs.me.cards_by_color

            def key(a: Action) -> tuple:
                cv = views[a.card.name]
                return (-counts.get(cv.color, 0), _card_points(cv))

            return max(builds, key=key)
        return _first(obs, ActionKind.WONDER) or _first(obs, ActionKind.DISCARD)
