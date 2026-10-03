"""Baseline agents used for testing, self-play data, and as opponents."""

from __future__ import annotations

import random
from typing import Dict, Set

from wonders.agents.base import Agent
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import CardView, Observation


def _card_points(card: CardView) -> tuple:
    """Immediate victory points + shields, tie-broken toward the cheaper build."""
    points = 0
    for effect in card.effects:
        for key in ('victory_points', 'shields'):
            value = effect.get(key, 0)
            if isinstance(value, int):
                points += value
    return (points, -(card.build_coin_cost or 0))


# A free build (Olympia) is only worth spending on a card costing at least this many coins.
FREE_BUILD_MIN_COST = 2


def _builds(obs: Observation):
    """One build action per buildable card, using a free build where it pays off.

    A FREE action replaces the paid BUILD of the same card when the card is otherwise
    unaffordable or would cost at least ``FREE_BUILD_MIN_COST`` coins. During a
    discard-pile build every legal action is FREE, so they are all kept.
    """
    views = {c.name: c for c in obs.hand}
    paid = {a.card.name: a for a in obs.legal_actions if a.kind is ActionKind.BUILD}
    for a in obs.legal_actions:
        if a.kind is ActionKind.FREE:
            cost = views[a.card.name].build_coin_cost
            if obs.decision == 'discard_build' or cost is None or cost >= FREE_BUILD_MIN_COST:
                paid[a.card.name] = a
    return list(paid.values())


def _first(obs: Observation, kind: ActionKind):
    return next((a for a in obs.legal_actions if a.kind is kind), None)


def _fallback(obs: Observation) -> Action:
    """Advance the wonder, else discard, else (discard-pile build) the first option."""
    return _first(obs, ActionKind.WONDER) or _first(obs, ActionKind.DISCARD) or obs.legal_actions[0]


class RandomAgent(Agent):
    """Picks a uniformly random legal action."""

    def __init__(self, rng: random.Random, name: str = 'random'):
        self.rng = rng
        self.name = name

    def decide(self, obs: Observation) -> Action:
        return self.rng.choice(obs.legal_actions)


class DiscardAgent(Agent):
    """Trivial floor: always take the 3 coins. Useful as a control opponent."""

    name = 'hoarder'

    def decide(self, obs: Observation) -> Action:
        return _first(obs, ActionKind.DISCARD) or obs.legal_actions[0]


class GreedyAgent(Agent):
    """Myopic: build the highest immediate value card, else advance the wonder, else discard.

    Value is the card's immediate victory points plus shields, breaking ties toward
    the cheaper build.
    """

    name = 'greedy'

    def decide(self, obs: Observation) -> Action:
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        builds = _builds(obs)
        if builds:
            return max(builds, key=lambda a: _card_points(views[a.card.name]))
        return _fallback(obs)


class ArchetypeAgent(Agent):
    """Builds toward a preferred set of card colours, with sensible fallbacks.

    Prefers building a card of one of ``colors`` (best by immediate points); if none
    are available it optionally advances its wonder, then builds the best other card,
    then finally discards. A cheap way to get diverse, characterful opponents for
    self-play.
    """

    def __init__(self, name: str, colors: Set[str], prefer_wonder: bool = False):
        self.name = name
        self.colors = colors
        self.prefer_wonder = prefer_wonder

    def decide(self, obs: Observation) -> Action:
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        builds = _builds(obs)
        preferred = [a for a in builds if views[a.card.name].color in self.colors]
        if preferred:
            return max(preferred, key=lambda a: _card_points(views[a.card.name]))
        if self.prefer_wonder:
            wonder = _first(obs, ActionKind.WONDER)
            if wonder is not None:
                return wonder
        if builds:
            return max(builds, key=lambda a: _card_points(views[a.card.name]))
        return _fallback(obs)


class MilitaristAgent(ArchetypeAgent):
    def __init__(self) -> None:
        super().__init__('militarist', {'red'})


class ScientistAgent(ArchetypeAgent):
    def __init__(self) -> None:
        super().__init__('scientist', {'green'})


class CivicAgent(ArchetypeAgent):
    def __init__(self) -> None:
        super().__init__('civic', {'blue'})


class EconomistAgent(ArchetypeAgent):
    def __init__(self) -> None:
        super().__init__('economist', {'yellow', 'brown', 'grey'})


class BuilderAgent(ArchetypeAgent):
    """Rushes its wonder, filling in with the best card otherwise."""

    def __init__(self) -> None:
        super().__init__('builder', set(), prefer_wonder=True)
