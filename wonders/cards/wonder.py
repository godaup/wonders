"""Wonder boards.

Each player owns one double-sided wonder board: a starting resource produced every
turn for free, and an ordered list of stages. Stages are built left-to-right; each
has a :class:`~wonders.cards.base.Cost` and grants effects (VP, coins, shields,
science, production, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from wonders.cards.base import Cost, Ressource
from wonders.effects.base import Effect


@dataclass
class WonderStage:
    cost: Cost
    effects: Tuple[Effect, ...] = ()

    def __post_init__(self) -> None:
        self.effects = tuple(self.effects)


@dataclass
class Wonder:
    name: str
    starting_resource: Ressource
    stages: Tuple[WonderStage, ...]
    side: str = 'A'

    def __post_init__(self) -> None:
        self.stages = tuple(self.stages)

    @property
    def starting_production(self) -> frozenset:
        return frozenset({self.starting_resource})
