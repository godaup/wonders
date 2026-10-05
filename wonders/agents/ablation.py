"""Agents that replace part of another agent's play — for measuring what a decision is worth.

:class:`LeaderAblation` keeps an agent's regular turns but swaps its Leaders decisions
(the pre-game draft and/or each Age's recruitment) for random or cost-proxy choices.
Pairing the wrapped agent against the original at the same tables
(:func:`~wonders.ml.evaluate.compare_agents`) shows how much those decisions matter.
"""

from __future__ import annotations

import random
from typing import List, Optional

from wonders.agents.base import Agent
from wonders.agents.bench import AgentSpec
from wonders.cards.base import Card
from wonders.engine.actions import Action
from wonders.engine.observation import Observation

LEADER_MODES = ('random-draft', 'random', 'cost')


class LeaderAblation(Agent):
    """``inner`` plays every regular turn; Leaders decisions follow ``mode``.

    Modes: ``'random-draft'`` (random draft picks, ``inner`` recruits), ``'random'``
    (random draft picks and random recruitment plays) and ``'cost'`` (the base
    :class:`~wonders.agents.base.Agent` hooks: keep / recruit the costliest leader).

    Args:
        inner: Spec of the agent being ablated (built with a child of ``rng``).
        mode: One of :data:`LEADER_MODES`.
        rng: Random source for the random choices.
        name: Display name.
    """

    def __init__(self, inner: AgentSpec, mode: str = 'random', rng: Optional[random.Random] = None, name: str = ''):
        if mode not in LEADER_MODES:
            raise ValueError(f'mode must be one of {LEADER_MODES}, got {mode!r}')
        self.rng = rng or random.Random()
        self.inner = inner.build(random.Random(self.rng.random()))
        self.mode = mode
        self.name = name or f'{inner.name}[{mode}]'

    def decide(self, obs: Observation) -> Action:
        return self.inner.decide(obs)

    def act(self, obs: Observation, options: List[Card]) -> Action:
        if obs.decision == 'draft':
            if self.mode == 'cost':
                return Action.build(Agent.draft_leader(self, options))
            return Action.build(self.rng.choice(options))
        if obs.decision == 'recruit' and self.mode != 'random-draft':
            if self.mode == 'random':
                return self.rng.choice(obs.legal_actions)
            action = Agent.recruit(self, options, obs)
            if action is not None:
                return action
        return self.inner.act(obs, options)


def leader_ablation(inner: AgentSpec, mode: str) -> AgentSpec:
    """A picklable spec of ``inner`` with its Leaders decisions replaced (named ``inner[mode]``)."""
    name = f'{inner.name}[{mode}]'
    return AgentSpec.of(name, LeaderAblation, seeded=True, inner=inner, mode=mode, name=name)
