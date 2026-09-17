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

from wonders.cards.base import Card
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import Observation


class Agent(ABC):
    name: str = 'agent'

    @abstractmethod
    def decide(self, obs: Observation) -> Action:
        """Return the action to take for ``obs.seat`` (must be a legal action)."""
        raise NotImplementedError

    # -- Leaders expansion hooks (sane defaults; override for smarter play) --
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
