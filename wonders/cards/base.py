"""Core static card types: colors, resources, costs and the Card itself.

This module deliberately imports nothing from :mod:`wonders.effects` at runtime so
that the effect system can depend on these enums without an import cycle. The
``effects`` field is only type-annotated (resolved lazily via
``from __future__ import annotations``).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.effects.base import Effect


class Color(enum.Enum):
    """Card categories. WHITE/BLACK are reserved for the Leaders/Cities expansions."""

    BROWN = 'brown'  # raw materials
    GREY = 'grey'  # manufactured goods
    BLUE = 'blue'  # civilian (victory points)
    YELLOW = 'yellow'  # commercial (coins / trade / resources)
    RED = 'red'  # military
    GREEN = 'green'  # science
    PURPLE = 'purple'  # guilds (Age III only)
    WHITE = 'white'  # Leaders expansion
    BLACK = 'black'  # Cities expansion


class Ressource(enum.Enum):
    """Tradeable resources.

    Spelling kept as ``Ressource`` to match the original scaffold. ``GOLD`` models
    coins and is intentionally *not* a producible/tradeable resource — coins are
    tracked as integers on the player and never satisfy a resource cost.
    """

    WOOD = 'wood'
    STONE = 'stone'
    ORE = 'ore'
    BRICKS = 'bricks'  # a.k.a. clay
    GLASS = 'glass'
    LINEN = 'linen'  # a.k.a. loom / textile
    PAPER = 'paper'  # a.k.a. papyrus
    GOLD = 'gold'  # coins; see note above

    @property
    def is_raw(self) -> bool:
        return self in RAW_MATERIALS

    @property
    def is_manufactured(self) -> bool:
        return self in MANUFACTURED_GOODS


RAW_MATERIALS = frozenset({Ressource.WOOD, Ressource.STONE, Ressource.ORE, Ressource.BRICKS})
MANUFACTURED_GOODS = frozenset({Ressource.GLASS, Ressource.LINEN, Ressource.PAPER})


class ScienceSymbol(enum.Enum):
    TABLET = 'tablet'
    GEAR = 'gear'
    COMPASS = 'compass'


@dataclass(frozen=True)
class Cost:
    """What a structure (or wonder stage) costs to build.

    ``coins`` are paid to the bank; ``resources`` must be produced by the player or
    bought from a neighbour. A free build is ``Cost()``.
    """

    coins: int = 0
    resources: Counter = field(default_factory=Counter)

    @classmethod
    def of(cls, *resources: Ressource, coins: int = 0) -> 'Cost':
        """Build a cost from positional resources plus optional coins.

        Example: ``Cost.of(WOOD, WOOD, coins=2)``.
        """
        return cls(coins=coins, resources=Counter(resources))

    @classmethod
    def free(cls) -> 'Cost':
        return cls()

    @property
    def is_free(self) -> bool:
        return self.coins == 0 and not self.resources

    def __post_init__(self) -> None:
        # Normalise a plain mapping/iterable into a Counter for ergonomics.
        if not isinstance(self.resources, Counter):
            object.__setattr__(self, 'resources', Counter(self.resources))


# A single unit of production offers a *choice* of one resource among a set.
# e.g. a brown "Wood" card -> frozenset({WOOD}); a "choice" card -> frozenset of options.
ResourceOption = frozenset


@dataclass
class Card:
    """A structure that can be built into a player's city.

    ``chain_requires`` is the name of a previously-built structure that lets this
    card be built for free; ``chain_enables`` lists structures this one unlocks.
    """

    name: str
    color: Color
    cost: Cost = field(default_factory=Cost)
    effects: Tuple['Effect', ...] = ()
    chain_requires: Optional[str] = None
    chain_enables: Tuple[str, ...] = ()
    age: Optional[int] = None
    # Deck-composition metadata (used by the deck builder, not the rules engine):
    expansion: str = 'base'
    # Player counts at which a copy of this card enters the deck. The number of
    # copies used in an N-player game is how many of these values are <= N.
    # Guild cards leave this empty — the builder draws (players + 2) of them at random.
    at_players: Tuple[int, ...] = ()
    guild: bool = False

    def __post_init__(self) -> None:
        self.effects = tuple(self.effects)
        self.chain_enables = tuple(self.chain_enables)
        self.at_players = tuple(self.at_players)

    def copies_for(self, n_players: int) -> int:
        """How many copies of this card a game with ``n_players`` uses."""
        return sum(1 for p in self.at_players if p <= n_players)
