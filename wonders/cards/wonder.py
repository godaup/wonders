"""Wonder boards.

Each player owns one double-sided wonder board: a starting resource produced every
turn for free (Rome and the Manneken Pis produce none), optional passive board powers
(Rome's leader discounts), and an ordered list of stages. Stages are built
left-to-right — or in any order on boards with ``any_order`` (the Great Wall); each
has a :class:`~wonders.cards.base.Cost` and grants effects (VP, coins, shields,
science, production, ...). A stage may instead *copy* a stage of a neighbour's board,
cost and effects alike (the Manneken Pis); the game resolves copies once it knows the
neighbours (``Game`` → ``Player.slots``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from wonders.cards.base import Cost, Resource
from wonders.effects.base import Effect

# A copied stage: which neighbour ('left' / 'right') and which of its stages (1-based, or 'last').
StageCopy = Tuple[str, Union[int, str]]


@dataclass
class WonderStage:
    cost: Cost
    effects: Tuple[Effect, ...] = ()
    copy: Optional[StageCopy] = None  # resolved per game: this stage is a neighbour's stage

    def __post_init__(self) -> None:
        self.effects = tuple(self.effects)


@dataclass
class Wonder:
    name: str
    starting_resource: Optional[Resource]
    stages: Tuple[WonderStage, ...]
    side: str = 'A'
    effects: Tuple[Effect, ...] = ()  # passive powers of the board itself, active from the start
    any_order: bool = False  # the stages may be built in any order (the Great Wall)
    extra_coins: int = 0  # coins on top of the starting treasury (the Manneken Pis)
    requires: Tuple[str, ...] = ()  # expansions a game needs to offer this board (Abu Simbel: leaders)

    def __post_init__(self) -> None:
        self.stages = tuple(self.stages)
        self.effects = tuple(self.effects)
        self.requires = tuple(self.requires)

    @property
    def starting_production(self) -> frozenset:
        return frozenset({self.starting_resource}) if self.starting_resource is not None else frozenset()

    def stage_variants(self, which: Union[int, str]) -> Tuple[WonderStage, ...]:
        """The stages a neighbour's copy of stage ``which`` (1-based or ``'last'``) may be.

        One stage normally; any of them on a board whose stages are built in any order (the
        owner of the Manneken Pis chooses which Great Wall stage to copy).
        """
        if self.any_order:
            return tuple(s for s in self.stages if s.copy is None)
        index = len(self.stages) if which == 'last' else int(which)
        if not 1 <= index <= len(self.stages) or self.stages[index - 1].copy is not None:
            return ()
        return (self.stages[index - 1],)
