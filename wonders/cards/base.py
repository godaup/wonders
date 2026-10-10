"""Core static card types: colors, resources, costs and the Card itself.

This module deliberately imports nothing from :mod:`wonders.effects` at runtime so
that the effect system can depend on these enums without an import cycle. The
``effects`` field is only type-annotated (resolved lazily via
``from __future__ import annotations``).
"""

from __future__ import annotations

import enum
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.effects.base import Effect


class IdentityEnum(enum.Enum):
    """An Enum hashed by identity.

    Enum members are singletons compared by identity, so ``object.__hash__`` is a
    valid hash — and it is C-level, unlike ``Enum.__hash__`` (a Python function
    hashing the member name). These enums are hashed millions of times per batch of
    games (resource Counters/frozensets in the payment solver), so this matters.
    Hash values never leak into results: nothing iterates sets of them in a way that
    affects play.
    """

    __hash__ = object.__hash__


SCORE_COLUMNS = ('civilian', 'commercial', 'guilds', 'leaders', 'cities')  # where a colour's card VP are scored
_COLOR_ID = re.compile(r'[a-z][a-z0-9_-]{0,31}')
_HEX = re.compile(r'#[0-9a-fA-F]{6}')


class _ColorMeta(type):
    def __iter__(cls) -> Iterator['Color']:
        return iter(list(cls._members.values()))  # type: ignore[attr-defined]

    def __len__(cls) -> int:
        return len(cls._members)  # type: ignore[attr-defined]


class Color(metaclass=_ColorMeta):
    """A card colour: the nine official ones, plus any colour an expansion defines.

    Behaves like the enum it replaced: ``Color('brown') is Color.BROWN``, members compare and hash by
    identity, ``.value`` is the JSON id and ``list(Color)`` iterates every defined colour. Expansion
    files declare new colours under ``"colors"`` (:meth:`define`); a colour is data, never code:

    * ``label`` — the category name shown on cards and in the GUI.
    * ``fill`` — the card colour (``#rrggbb``).
    * ``score`` — the :class:`~wonders.engine.game.ScoreBreakdown` column its cards' VP count towards.
    * ``tradeable`` — whether neighbours may buy its cards' production (true for brown / grey).
    """

    __slots__ = ('value', 'name', 'label', 'fill', 'score', 'tradeable', 'official')
    _members: ClassVar[Dict[str, 'Color']] = {}

    BROWN: ClassVar['Color']  # raw materials
    GREY: ClassVar['Color']  # manufactured goods
    BLUE: ClassVar['Color']  # civilian (victory points)
    YELLOW: ClassVar['Color']  # commercial (coins / trade / resources)
    RED: ClassVar['Color']  # military
    GREEN: ClassVar['Color']  # science
    PURPLE: ClassVar['Color']  # guilds (Age III only)
    WHITE: ClassVar['Color']  # Leaders expansion
    BLACK: ClassVar['Color']  # Cities expansion

    value: str
    name: str
    label: str
    fill: str
    score: str
    tradeable: bool
    official: bool

    def __new__(cls, value: str) -> 'Color':
        try:
            return cls._members[value]
        except KeyError:
            raise ValueError(f'{value!r} is not a known card colour') from None

    @classmethod
    def define(
        cls,
        value: str,
        label: str = '',
        fill: str = '#9aa3ad',
        score: str = 'commercial',
        tradeable: bool = False,
        official: bool = False,
    ) -> 'Color':
        """Define a colour (idempotent for identical attributes) and return it.

        Raises:
            ValueError: a malformed id, fill or score column, or a colour already defined differently
                (the official colours cannot be redefined).
        """
        label = label or value.replace('_', ' ').replace('-', ' ').title()
        existing = cls._members.get(value)
        if existing is not None:
            if (existing.label, existing.fill, existing.score, existing.tradeable) != (label, fill, score, tradeable):
                raise ValueError(f'card colour {value!r} is already defined differently')
            return existing
        if not _COLOR_ID.fullmatch(value):
            raise ValueError(f'bad colour id {value!r} (lowercase letters, digits, - and _)')
        if not _HEX.fullmatch(fill):
            raise ValueError(f'colour {value!r}: fill must look like #a1b2c3, not {fill!r}')
        if score not in SCORE_COLUMNS:
            raise ValueError(f'colour {value!r}: score must be one of {", ".join(SCORE_COLUMNS)}, not {score!r}')
        color = object.__new__(cls)
        for attr, val in zip(cls.__slots__, (value, value.upper(), label, fill, score, bool(tradeable), official)):
            object.__setattr__(color, attr, val)
        cls._members[value] = color
        return color

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> 'Color':
        """Define a colour from its expansion-file entry (``{"id": ..., "label": ..., ...}``)."""
        unknown = set(spec) - {'id', 'label', 'fill', 'score', 'tradeable'}
        if unknown:
            raise ValueError(f'colour {spec.get("id")!r}: unknown keys {", ".join(sorted(unknown))}')
        return cls.define(
            spec['id'],
            spec.get('label', ''),
            spec.get('fill', '#9aa3ad'),
            spec.get('score', 'commercial'),
            bool(spec.get('tradeable', False)),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('card colours are immutable')

    __hash__ = object.__hash__  # identity, like IdentityEnum

    def __repr__(self) -> str:
        return f'Color.{self.name}'

    def __reduce__(self) -> Tuple[Any, ...]:
        # Pickled by definition, so worker processes get extension colours they have not loaded yet.
        return (Color.define, (self.value, self.label, self.fill, self.score, self.tradeable, self.official))

    def __copy__(self) -> 'Color':
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'Color':
        return self


for _attr, _value, _label, _fill, _score in (
    ('BROWN', 'brown', 'Raw material', '#8a5a2b', 'commercial'),
    ('GREY', 'grey', 'Manufactured good', '#7c828a', 'commercial'),
    ('BLUE', 'blue', 'Civilian', '#2f6fb0', 'civilian'),
    ('YELLOW', 'yellow', 'Commercial', '#d99a1c', 'commercial'),
    ('RED', 'red', 'Military', '#b23b3b', 'commercial'),
    ('GREEN', 'green', 'Scientific', '#3f9e5a', 'commercial'),
    ('PURPLE', 'purple', 'Guild', '#7a4fb0', 'guilds'),
    ('WHITE', 'white', 'Leader', '#c8ccd2', 'leaders'),
    ('BLACK', 'black', 'City', '#2b2f36', 'cities'),
):
    setattr(
        Color, _attr, Color.define(_value, _label, _fill, _score, tradeable=_attr in ('BROWN', 'GREY'), official=True)
    )
OFFICIAL_COLORS: Tuple[Color, ...] = tuple(Color)  # fixed order: encoders keep one feature per official colour


class Resource(IdentityEnum):
    """Tradeable resources.

    Spelling kept as ``Resource`` to match the original scaffold. ``GOLD`` models
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


RAW_MATERIALS = frozenset({Resource.WOOD, Resource.STONE, Resource.ORE, Resource.BRICKS})
MANUFACTURED_GOODS = frozenset({Resource.GLASS, Resource.LINEN, Resource.PAPER})


class ScienceSymbol(IdentityEnum):
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
    def of(cls, *resources: Resource, coins: int = 0) -> 'Cost':
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
