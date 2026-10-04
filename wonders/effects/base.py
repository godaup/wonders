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

Leaders / Cities powers:

    * ``science_set_bonus()``   -> extra VP per complete science set (Aristotle)
    * ``science_copies()``      -> neighbour science symbols copied at scoring
    * ``build_income(event)``   -> coins whenever the owner builds (Xenophon, Vitruvius, ...)
    * ``coins_per_victory()``   -> coins per military victory token won (Nero)
    * ``redirects_defeats()``   -> own defeat tokens go to the winning neighbour (Tomyris)
    * ``resource_discount(t)``  -> resource units waived when paying for target ``t``
    * ``builds_free(card)``     -> build ``card`` without paying anything (Ramses, Maecenas)
    * ``bank_purchases()``      -> resources buyable from the bank per build (Bilkis)
    * ``private_production(p)`` -> own-use production slots, given produced resources ``p``
    * ``trade_reduction(...)``  -> coins off a neighbour purchase (Smugglers Cache)
    * ``coin_discount(card)``   -> coins off the owner's own card cost (Rome B: leaders)
    * ``neighbour_coin_discount(card)`` -> coins off the *neighbours'* card cost (Rome B)
    * ``leader_draws()``        -> leaders drawn from the box when built (Rome B stage 1)
    * ``extra_recruits()``      -> extra leaders the owner may recruit when built (Rome B)
    * ``leader_copies()``       -> neighbour leaders copied when built (Courtesans' Guild)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from wonders.cards.base import Ressource, ScienceSymbol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.cards.base import Card
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
    table: Tuple['Player', ...] = ()  # every player, seat order (for "all other players" effects)

    @property
    def neighbours(self) -> Tuple['Player', 'Player']:
        return (self.left, self.right)

    @property
    def others(self) -> Tuple['Player', ...]:
        """Every other player at the table (falls back to the neighbours without a table)."""
        if not self.table:
            return tuple(dict.fromkeys(p for p in self.neighbours if p is not self.player))
        return tuple(p for p in self.table if p is not self.player)


@dataclass(frozen=True)
class BuildEvent:
    """What the owner just built, for :meth:`Effect.build_income`."""

    color: Optional[str]  # card colour value, or None for a wonder stage
    via_chain: bool = False  # built for free through a chain
    traded: int = 0  # coins paid to neighbours for resources
    neighbours_traded: int = 0  # how many neighbours (0-2) resources were bought from


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

    def science_set_bonus(self) -> int:
        return 0

    def science_copies(self) -> int:
        return 0

    def build_income(self, event: BuildEvent) -> int:
        return 0

    def coins_per_victory(self) -> int:
        return 0

    def redirects_defeats(self) -> bool:
        return False

    def resource_discount(self, target: str) -> int:
        """Resource units waived when paying for ``target`` (a card colour value or ``'wonder'``)."""
        return 0

    def builds_free(self, card: 'Card') -> bool:
        return False

    def bank_purchases(self) -> int:
        return 0

    def private_production(self, produced: frozenset) -> Tuple[frozenset, ...]:
        """Production slots only the owner may use (neighbours cannot buy them)."""
        return ()

    def trade_reduction(self, player: 'Player', side: str, resource: Ressource) -> int:
        return 0

    def coin_discount(self, card: 'Card') -> int:
        return 0

    def neighbour_coin_discount(self, card: 'Card') -> int:
        return 0

    def leader_draws(self) -> int:
        return 0

    def extra_recruits(self) -> int:
        return 0

    def leader_copies(self) -> int:
        return 0
