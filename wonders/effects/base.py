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

Armada powers (fleet spaces of the Shipyard count as effects too, those of the space a fleet is on):

    * ``naval_strength()``      -> naval strength in the naval conflicts
    * ``commercial_level()``    -> the commercial level (protects from Taxes)
    * ``tax()``                 -> raise a Tax when it enters play (end of turn, the highest applies)
    * ``explores()``            -> island levels explored at the end of the turn
    * ``island_draws()``        -> island levels whose top card comes into play at once
    * ``fleet_advances()``      -> fleets advanced for free (by track)
    * ``fleet_choices()``       -> free advances of fleets of the owner's choice (each entry: different fleets)
    * ``boards()``              -> Boarding: seat offsets of the players given a Boarding token
    * ``far_trade()``           -> purchases from the players two seats away (Emporiums)
    * ``coin_loss_immune()``, ``naval_opt_out()``, ``fleet_income()``, ``wonder_fleet_choice()``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from wonders.cards.base import Resource, ScienceSymbol

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
    debt_tokens: bool = True  # debt tokens are in play (Cities): unpaid plunder becomes debt
    final: bool = False  # the game is over (hidden things are revealed: Stonehenge B's buried card)
    age: int = 0  # the current Age (Arsinoe's plunder, Nitocris' token)
    victory_value: int = 0  # the military victory token of the current Age

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

    def applies(self, side: str, resource: Resource) -> bool:
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

    def free_build_colors(self) -> Tuple[str, ...]:
        """Colour values ``free_builds_per_age`` is limited to (empty: any card; Caligula: black)."""
        return ()

    def science_most(self) -> int:
        """Extra science symbols of the kind the owner has most of, at the end (Enheduanna)."""
        return 0

    def science_replaces(self) -> int:
        """Science symbols the owner may change into any other at the end (Aganice)."""
        return 0

    def reserve_bonus(self) -> int:
        """Extra coins, once per turn, when the owner gains coins from the reserve (Berenice)."""
        return 0

    def stage_plunder(self) -> int:
        """Coins every other player loses each time the owner builds a wonder stage (Octavia)."""
        return 0

    def free_first_of_color(self) -> bool:
        """The first card of each colour in the owner's city costs nothing (Olympia day, 2nd edition)."""
        return False

    def free_build_turns(self) -> Tuple[str, ...]:
        """Turns of each Age (``'first'`` / ``'last'``) whose card may be built for free (Olympia night)."""
        return ()

    def discard_builds(self) -> int:
        return 0

    def guild_copies(self) -> int:
        return 0

    def discard_build_priority(self) -> int:
        """Order among discard builds owed in one turn, lowest first (Wonder Pack FAQ)."""
        return 0

    def leader_entombs(self) -> int:
        """Abu Simbel: recruited leaders to entomb (decision at the end of the turn)."""
        return 0

    def buried(self, card: 'Card') -> 'Effect':
        """This effect of a wonder stage built with ``card`` (Stonehenge B remembers its colour)."""
        return self

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

    def trade_reduction(self, player: 'Player', side: str, resource: Resource) -> int:
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

    # -- Armada ------------------------------------------------------------
    def naval_strength(self) -> int:
        return 0

    def commercial_level(self) -> int:
        return 0

    def tax(self) -> int:
        """A Tax raised when this enters play: every other player loses this minus their commercial level."""
        return 0

    def explores(self) -> Tuple[int, ...]:
        """Island levels to explore at the end of the turn this enters play (green fleet spaces)."""
        return ()

    def island_draws(self) -> Tuple[int, ...]:
        """Island levels whose top card comes into play at once (no choice)."""
        return ()

    def fleet_advances(self) -> Tuple[str, ...]:
        """Fleets (tracks) advanced one space for free when this enters play."""
        return ()

    def fleet_choices(self) -> Tuple[int, ...]:
        """Free advances of fleets of the owner's choice: one entry per group of ``k`` different fleets."""
        return ()

    def boards(self) -> Tuple[int, ...]:
        """Boarding: seat offsets (+2: left of the left neighbour, -2: right of the right one) given a token."""
        return ()

    def far_trade(self) -> Tuple[Tuple[int, int, int], ...]:
        """``(seat offset, units per turn, price)``: resources the owner may buy from a non-neighbour."""
        return ()

    def coin_loss_immune(self) -> bool:
        """No coin losses from other players' Taxes and Pirates."""
        return False

    def naval_opt_out(self) -> bool:
        """May sit out a naval conflict."""
        return False

    def fleet_income(self) -> int:
        """Coins each time one of the owner's fleets advances."""
        return 0

    def wonder_fleet_choice(self) -> bool:
        """No fleet goes with the wonder: each wonder stage advances a fleet of the owner's choice for free."""
        return False

    def counts(self) -> Optional[Dict[str, object]]:
        """What this effect pays per unit, for agents valuing combos (``None`` if nothing).

        A JSON-able dict: ``what`` (e.g. ``'colors'``, ``'wonder_stages'``, ``'color_sets'``,
        ``'build_income'``), ``scope`` (``'self'`` / ``'neighbours'`` / ``'both'``), ``colors``
        (card colour values) and ``vp_each`` / ``coins_each``. Descriptive only: the engine never
        reads it.
        """
        return None
