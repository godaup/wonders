"""Optional, combinable expansions.

An expansion is a named module that can adjust setup and rules. A :class:`Ruleset`
is assembled from any combination of expansion ids (``base`` is always implied) and
composes their contributions — e.g. Leaders raises the starting treasury to 6. The
content side combines the same way: ``load_content(*ruleset.content_expansions)``
merges each expansion's cards and wonders.

Most rule tweaks are expressed as effects/player-state that are simply inert when the
expansion is absent (debt tokens are 0, diplomacy tokens are 0), so combining is
just a union — no special-casing in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

DEFAULT_STARTING_COINS = 3


@dataclass(frozen=True)
class Expansion:
    id: str
    name: str
    starting_coins: int = DEFAULT_STARTING_COINS
    # False for expansions whose subsystems are not yet mechanized (content/registry
    # only). They can still be listed in a ruleset; the engine flags the gap.
    implemented: bool = True
    note: str = ''


EXPANSIONS: Dict[str, Expansion] = {
    'base': Expansion('base', 'Base game'),
    'leaders': Expansion('leaders', 'Leaders', starting_coins=6),
    'cities': Expansion('cities', 'Cities'),
    'armada': Expansion(
        'armada',
        'Armada',
        implemented=False,
        note='naval boards, fleets and naval combat are not yet mechanized',
    ),
    'babel': Expansion(
        'babel',
        'Babel',
        implemented=False,
        note='the Tower of Babel and Great Projects modules are not yet mechanized',
    ),
}


@dataclass(frozen=True)
class Ruleset:
    """A combination of active expansions (``base`` always included)."""

    expansions: Tuple[str, ...]

    @classmethod
    def of(cls, *ids: str) -> 'Ruleset':
        combined = tuple(dict.fromkeys(('base', *ids)))  # dedupe, base first, keep order
        unknown = [i for i in combined if i not in EXPANSIONS]
        if unknown:
            raise ValueError(f'unknown expansion(s): {unknown}')
        return cls(combined)

    def uses(self, expansion_id: str) -> bool:
        return expansion_id in self.expansions

    @property
    def content_expansions(self) -> Tuple[str, ...]:
        return self.expansions

    @property
    def starting_coins(self) -> int:
        return max(EXPANSIONS[i].starting_coins for i in self.expansions)

    @property
    def unimplemented(self) -> Tuple[str, ...]:
        """Active expansions whose subsystems are not fully mechanized yet."""
        return tuple(i for i in self.expansions if not EXPANSIONS[i].implemented)
