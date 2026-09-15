"""Baseline agents used for testing, self-play data, and as opponents."""

from __future__ import annotations

import random
from typing import Dict

from wonders.agents.base import Agent
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import CardView, Observation


class RandomAgent(Agent):
    """Picks a uniformly random legal action."""

    def __init__(self, rng: random.Random, name: str = 'random'):
        self.rng = rng
        self.name = name

    def decide(self, obs: Observation) -> Action:
        return self.rng.choice(obs.legal_actions)


class GreedyAgent(Agent):
    """Myopic: build the highest immediate value card, else advance the wonder, else discard.

    Value is the card's immediate victory points plus shields (from the effect
    summaries in the observation), breaking ties toward the cheaper build.
    """

    name = 'greedy'

    def decide(self, obs: Observation) -> Action:
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        builds = [a for a in obs.legal_actions if a.kind is ActionKind.BUILD]
        if builds:
            return max(builds, key=lambda a: self._value(views[a.card.name]))
        wonders = [a for a in obs.legal_actions if a.kind is ActionKind.WONDER]
        if wonders:
            return wonders[0]
        return next(a for a in obs.legal_actions if a.kind is ActionKind.DISCARD)

    @staticmethod
    def _value(card: CardView):
        points = 0
        for effect in card.effects:
            for key in ('victory_points', 'shields'):
                value = effect.get(key, 0)
                if isinstance(value, int):
                    points += value
        return (points, -(card.build_coin_cost or 0))
