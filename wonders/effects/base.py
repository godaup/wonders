"""The effect system.

Every card / wonder-stage benefit is an :class:`Effect`. Effects expose a *uniform*
set of hooks with neutral defaults; the engine queries all effects the same way and
never needs to know concrete subclasses. This keeps content (effects) and rules
(engine) decoupled and makes the whole game state cheap to introspect — useful for
the project's Deep-Learning goal.

Hooks:
    * ``production()``      -> resource options this effect makes available each turn
    * ``shields()``         -> military strength contributed
    * ``science()``         -> a science symbol, or ``None``
    * ``science_wild()``    -> number of "choose any symbol" wilds contributed
    * ``trade_rules()``     -> trade-price modifiers
    * ``coins_on_build()``  -> coins gained from the bank when the card is built
    * ``victory_points()``  -> end-game victory points

Turn-structure powers (the special wonder stages) are hooks too, so the engine asks
every effect the same questions instead of switching on concrete classes:

    * ``plays_last_card()``     -> may play the 7th card of each Age (Babylon B)
    * ``free_builds_per_age()`` -> free builds from hand allowed each Age (Olympia A)
    * ``discard_builds()``      -> free builds from the discard pile, granted once on build
    * ``guild_copies()``        -> neighbour guilds copied at scoring (Olympia B)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from wonders.cards.base import Ressource, ScienceSymbol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.engine.player import Player


@dataclass
class Context:
    """A player together with their two immediate neighbours.

    Passed to effects that depend on surrounding cities (guilds, some yellow cards)
    and to immediate build-time coin effects.
    """

    player: 'Player'
    left: 'Player'
    right: 'Player'

    @property
    def neighbours(self) -> Tuple['Player', 'Player']:
        return (self.left, self.right)


@dataclass(frozen=True)
class TradeRule:
    """Reduces the price paid to a neighbour for a category of resources.

    ``side`` is ``'left'``, ``'right'`` or ``'both'``; ``category`` is ``'raw'``,
    ``'manufactured'`` or ``'both'``. Default trade price is 2 coins/unit.
    """

    side: str = 'both'
    category: str = 'both'
    price: int = 1

    def applies(self, side: str, resource: Ressource) -> bool:
        if self.side != 'both' and self.side != side:
            return False
        if self.category == 'raw' and not resource.is_raw:
            return False
        if self.category == 'manufactured' and not resource.is_manufactured:
            return False
        return True


class Effect:
    """Base effect. Subclasses override the hooks they care about."""

    def production(self) -> Tuple[frozenset, ...]:
        return ()

    def shields(self) -> int:
        return 0

    def science(self) -> Optional[ScienceSymbol]:
        return None

    def science_wild(self) -> int:
        return 0

    def trade_rules(self) -> Tuple[TradeRule, ...]:
        return ()

    def coins_on_build(self, ctx: Context) -> int:
        return 0

    def on_build(self, ctx: Context) -> None:
        """Arbitrary state change at build time (e.g. impose debt, grant diplomacy).

        Runs in addition to ``coins_on_build``; may mutate ``ctx.player`` and its
        neighbours. Default no-op.
        """

    def victory_points(self, ctx: Context) -> int:
        return 0

    def plays_last_card(self) -> bool:
        return False

    def free_builds_per_age(self) -> int:
        return 0

    def discard_builds(self) -> int:
        return 0

    def guild_copies(self) -> int:
        return 0
