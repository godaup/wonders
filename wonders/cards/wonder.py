"""Wonder boards.

Each player owns one double-sided wonder board: a starting resource produced every
turn for free (Rome produces none), optional passive board powers (Rome's leader
discounts), and an ordered list of stages. Stages are built left-to-right; each
has a :class:`~wonders.cards.base.Cost` and grants effects (VP, coins, shields,
science, production, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from wonders.cards.base import Cost, Resource
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
    starting_resource: Optional[Resource]
    stages: Tuple[WonderStage, ...]
    side: str = 'A'
    effects: Tuple[Effect, ...] = ()  # passive powers of the board itself, active from the start

    def __post_init__(self) -> None:
        self.stages = tuple(self.stages)
        self.effects = tuple(self.effects)

    @property
    def starting_production(self) -> frozenset:
        return frozenset({self.starting_resource}) if self.starting_resource is not None else frozenset()
