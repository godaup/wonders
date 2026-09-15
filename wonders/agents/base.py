"""The agent interface.

Every agent — scripted, search-based, or a neural policy — implements the same
:meth:`Agent.decide` contract: given an :class:`~wonders.engine.observation.Observation`
(the full state a seat may see), return one of its ``legal_actions``. This shared
interface is what lets different agents be swapped into a match or played against
each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from wonders.engine.actions import Action
from wonders.engine.observation import Observation


class Agent(ABC):
    name: str = 'agent'

    @abstractmethod
    def decide(self, obs: Observation) -> Action:
        """Return the action to take for ``obs.seat`` (must be a legal action)."""
        raise NotImplementedError
