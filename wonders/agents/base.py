"""The agent interface.

Every agent — scripted, search-based, or a neural policy — implements the same
:meth:`Agent.decide` contract: given an :class:`~wonders.engine.observation.Observation`
(the full state a seat may see), return one of its ``legal_actions``. This shared
interface is what lets different agents be swapped into a match or played against
each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from wonders.agents import naval
from wonders.cards.base import Card
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import Observation


class Agent(ABC):
    name: str = 'agent'

    @abstractmethod
    def decide(self, obs: Observation) -> Action:
        """Return the action to take for ``obs.seat`` (must be a legal action)."""
        raise NotImplementedError

    def act(self, obs: Observation, options: List[Card]) -> Action:
        """Answer any decision point: routes Leaders draft / recruitment to their hooks.

        ``options`` are the cards being chosen from (``obs.hand`` as cards). Regular
        turns and wonder-power decisions go to :meth:`decide`.
        """
        if obs.decision == 'draft':
            return Action.build(self.draft_leader(options))
        if obs.decision == 'entomb_leader':
            return Action.build(self.entomb_leader(options, obs))
        if obs.decision == 'explore':
            return Action.build(self.explore(options, obs))
        if obs.decision == 'advance_fleet':
            return Action.build(self.advance_fleet(options, obs))
        if obs.decision == 'recruit':
            action = self.recruit(options, obs)
            if action is not None:
                return self.naval(obs, action)
        return self.naval(obs, self.decide(obs))

    # -- Armada hooks (sane defaults; see wonders.agents.naval) --
    def naval(self, obs: Observation, action: Action) -> Action:
        """Add a Naval Construction to ``action`` (or sell for the yellow fleet) when it is worth it."""
        return naval.with_naval(obs, action)

    def explore(self, options: List[Card], obs: Observation) -> Card:
        """Keep one of the islands offered. Default: the one worth most by a rough value."""
        return naval.best_island(options, obs)

    def advance_fleet(self, options: List[Card], obs: Observation) -> Card:
        """Pick the fleet to advance for free (``options`` are fleet cards). Default: the most valuable step."""
        return naval.best_fleet(options, obs)

    # -- Leaders expansion hooks (sane defaults; override for smarter play) --
    def entomb_leader(self, options: List[Card], obs: Observation) -> Card:
        """Abu Simbel: pick one of your recruited leaders to entomb (it scores twice its cost).

        Default: the costliest (the most points).
        """
        return max(options, key=lambda c: c.cost.coins)

    def draft_leader(self, options: List[Card]) -> Card:
        """Pick one leader to keep from ``options`` during the pre-game draft.

        Default: keep the costliest leader (a rough value proxy).
        """
        return max(options, key=lambda c: c.cost.coins)

    def recruit(self, leaders: List[Card], obs: Observation) -> Optional[Action]:
        """Play one leader at the start of an Age (recruit / wonder / discard).

        ``obs`` is built with the leader hand as its hand, so ``obs.legal_actions``
        are the plays available. Default: recruit the costliest affordable leader,
        else discard the cheapest to bank the coins. ``None`` only if no leaders.
        """
        if not leaders:
            return None
        builds = [a for a in obs.legal_actions if a.kind is ActionKind.BUILD]
        if builds:
            return max(builds, key=lambda a: a.card.cost.coins)
        discards = [a for a in obs.legal_actions if a.kind is ActionKind.DISCARD]
        return min(discards, key=lambda a: a.card.cost.coins)
