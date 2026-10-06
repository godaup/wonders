"""Concrete effects covering the base-game mechanics.

These are enough to express brown/grey production, red military, green science,
blue civilian points, yellow commercial cards (trade discounts, coin income,
end-game coin/VP-per-card) and purple guild scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

from wonders.cards.base import MANUFACTURED_GOODS, RAW_MATERIALS, Card, Color, Ressource, ScienceSymbol
from wonders.effects.base import BuildEvent, Context, Effect, TradeRule

TRADEABLE = RAW_MATERIALS | MANUFACTURED_GOODS


class Production(Effect):
    """Produces one resource per turn, chosen from ``options`` each time it is used.

    A single-resource brown/grey card offers one resource; a "choice" card
    (Forum, Caravansery, many wonder starts) offers several — but still only one
    unit per turn.
    """

    def __init__(self, *resources: Ressource, choice: Iterable[Ressource] = ()):
        self.options = frozenset(resources) | frozenset(choice)

    def production(self) -> Tuple[frozenset, ...]:
        return (self.options,)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f'Production({sorted(r.value for r in self.options)})'


class MultiProduction(Effect):
    """Produces several resources simultaneously (e.g. a wonder giving Wood + Ore).

    Each argument is one independent production slot (itself possibly a choice).
    """

    def __init__(self, *slots: Iterable[Ressource]):
        self.slots = tuple(frozenset(s) for s in slots)

    def production(self) -> Tuple[frozenset, ...]:
        return self.slots


@dataclass
class Coins(Effect):
    """Flat coins from the bank when built (e.g. Tavern, East Trading Post start)."""

    amount: int

    def coins_on_build(self, ctx: Context) -> int:
        return self.amount


@dataclass
class Military(Effect):
    shields_count: int

    def shields(self) -> int:
        return self.shields_count


@dataclass
class Science(Effect):
    symbol: ScienceSymbol

    def science(self):
        return self.symbol


@dataclass
class ScienceWild(Effect):
    """Choose any one science symbol (Scientists Guild, Babylon B stage)."""

    count: int = 1

    def science_wild(self) -> int:
        return self.count


@dataclass
class Trade(Effect):
    """A trade discount (East/West Trading Post, Marketplace, some wonders)."""

    side: str = 'both'
    category: str = 'both'
    price: int = 1

    def trade_rules(self) -> Tuple[TradeRule, ...]:
        return (TradeRule(side=self.side, category=self.category, price=self.price),)


@dataclass
class VictoryPoints(Effect):
    """Fixed end-game points (blue civilian cards, most wonder stages)."""

    points: int

    def victory_points(self, ctx: Context) -> int:
        return self.points


@dataclass
class Special(Effect):
    """A power that is not yet mechanized in the engine (mostly wonder stages).

    Inert on every hook, so it neither scores nor produces — it only documents the
    intended power (e.g. play-from-discard, free-build, copy-a-guild, play the 7th
    card). Implementing these requires turn-loop support; until then this keeps the
    data complete and renderable and makes the gaps explicit.
    """

    power: str
    description: str = ''


@dataclass
class PlayLastCard(Effect):
    """Babylon B: play the 7th card of each Age instead of discarding it."""

    def plays_last_card(self) -> bool:
        return True


@dataclass
class FreeBuildPerAge(Effect):
    """Olympia A: once per Age, build a structure from hand without paying its cost."""

    count: int = 1

    def free_builds_per_age(self) -> int:
        return self.count


@dataclass
class BuildFromDiscard(Effect):
    """Halicarnassus: at the end of the turn, build one discarded card for free."""

    count: int = 1

    def discard_builds(self) -> int:
        return self.count


@dataclass
class CopyGuild(Effect):
    """Olympia B: at game end, copy one guild built by a neighbour."""

    count: int = 1

    def guild_copies(self) -> int:
        return self.count


def _targets(ctx: Context, scope: str):
    if scope == 'self':
        return (ctx.player,)
    if scope == 'neighbours':
        return (ctx.left, ctx.right)
    if scope == 'both':
        return (ctx.player, ctx.left, ctx.right)
    if scope == 'others':  # every other player at the table
        return ctx.others
    raise ValueError(f'unknown scope {scope!r}')


@dataclass
class Tally(Effect):
    """Count something across a scope and grant coins (on build) and/or end VP.

    Covers a wide swathe of yellow and purple cards:

    * ``what='colors'`` with ``colors=...`` counts matching cards.
      - Vineyard: ``Tally('colors', 'both', {BROWN}, coins_each=1)``
      - Haven:    ``Tally('colors', 'self', {BROWN}, coins_each=1, vp_each=1)``
      - Workers Guild: ``Tally('colors', 'neighbours', {BROWN}, vp_each=1)``
    * ``what='wonder_stages'`` counts built wonder stages.
      - Arena: ``Tally('wonder_stages', 'self', coins_each=3, vp_each=1)``
      - Builders Guild: ``Tally('wonder_stages', 'both', vp_each=1)``
    * ``what='defeat_tokens'`` counts military defeats (Strategists Guild, neighbours).
    """

    what: str
    scope: str
    colors: Iterable[Color] = field(default_factory=frozenset)
    coins_each: int = 0
    vp_each: int = 0
    token_value: Optional[int] = None  # for what='victory_tokens': only count tokens of this value

    def __post_init__(self) -> None:
        self.colors = frozenset(self.colors)

    def _metric(self, player) -> int:
        if self.what == 'colors':
            return sum(1 for c in player.cards if c.color in self.colors)
        if self.what == 'wonder_stages':
            return player.wonder_stages_built
        if self.what == 'defeat_tokens':
            return sum(1 for t in player.tokens if t < 0)
        if self.what == 'tokens':  # every military token, victory or defeat
            return len(player.tokens)
        if self.what == 'victory_tokens':  # positive military tokens (optionally of one Age's value)
            if self.token_value is not None:
                return sum(1 for t in player.tokens if t == self.token_value)
            return sum(1 for t in player.tokens if t > 0)
        if self.what == 'coins':  # 1 unit per 3 coins (Midas, Gamers' Guild)
            return player.coins // 3
        raise ValueError(f'unknown tally target {self.what!r}')

    def _count(self, ctx: Context) -> int:
        return sum(self._metric(t) for t in _targets(ctx, self.scope))

    def coins_on_build(self, ctx: Context) -> int:
        # Immediate coin income is based on the player's own city at build time.
        if not self.coins_each:
            return 0
        return self.coins_each * self._count(ctx)

    def victory_points(self, ctx: Context) -> int:
        if not self.vp_each:
            return 0
        return self.vp_each * self._count(ctx)

    def counts(self) -> Optional[Dict[str, object]]:
        return {
            'what': self.what,
            'scope': self.scope,
            'colors': sorted(c.value for c in self.colors),
            'vp_each': self.vp_each,
            'coins_each': self.coins_each,
        }


@dataclass
class Debt(Effect):
    """Cities: impose debt tokens (each worth -1 VP at game end) on a scope of cities."""

    amount: int
    scope: str = 'self'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            player.debt += self.amount


@dataclass
class Diplomacy(Effect):
    """Cities: gain diplomacy tokens; each lets you sit out one Age's military conflict."""

    count: int = 1

    def on_build(self, ctx: Context) -> None:
        ctx.player.diplomacy += self.count


@dataclass
class Plunder(Effect):
    """Cities: each target loses ``amount`` coins; any shortfall becomes debt.

    The default scope ``'others'`` is every other player at the table.
    """

    amount: int
    scope: str = 'others'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            lost = min(self.amount, player.coins)
            player.coins -= lost
            player.debt += self.amount - lost


@dataclass
class Grant(Effect):
    """Cities: each target *gains* coins (e.g. Customs, Mint force opponents to profit)."""

    amount: int
    scope: str = 'others'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            player.coins += self.amount


@dataclass
class GainToken(Effect):
    """Cities Raider cards: immediately gain a military victory token of ``value`` (1/3/5)."""

    value: int

    def on_build(self, ctx: Context) -> None:
        ctx.player.tokens.append(self.value)
        if self.value > 0:  # a victory token: Nero pays for every one gained
            ctx.player.coins += ctx.player.coins_per_victory()


# --- Leaders / Cities powers ------------------------------------------------


@dataclass
class ColorSets(Effect):
    """End VP per complete set of one card of each of ``colors`` (Justinian, Plato)."""

    colors: Iterable[Color]
    vp_each: int

    def __post_init__(self) -> None:
        self.colors = frozenset(self.colors)

    def victory_points(self, ctx: Context) -> int:
        counts = [sum(1 for c in ctx.player.cards if c.color is color) for color in self.colors]
        return self.vp_each * min(counts, default=0)

    def counts(self) -> Optional[Dict[str, object]]:
        return {
            'what': 'color_sets',
            'scope': 'self',
            'colors': sorted(c.value for c in self.colors),
            'vp_each': self.vp_each,
        }


@dataclass
class ScienceSetBonus(Effect):
    """Each complete science set (tablet + gear + compass) is worth ``points`` more (Aristotle)."""

    points: int = 3

    def science_set_bonus(self) -> int:
        return self.points


@dataclass
class ScienceCopy(Effect):
    """At game end, copy one science symbol a neighbour has (Pigeonhole, Band of Spies, ...)."""

    count: int = 1

    def science_copies(self) -> int:
        return self.count


@dataclass
class BuildIncome(Effect):
    """Coins whenever the owner builds something matching every given condition.

    * Xenophon: ``BuildIncome(2, color='yellow')``
    * Vitruvius: ``BuildIncome(2, via_chain=True)``
    * Hatshepsut: ``BuildIncome(1, after_trade=True)`` — paid per neighbour bought from
    """

    coins: int
    color: Optional[str] = None
    via_chain: bool = False
    after_trade: bool = False

    def build_income(self, event: BuildEvent) -> int:
        if self.color is not None and event.color != self.color:
            return 0
        if self.via_chain and not event.via_chain:
            return 0
        if self.after_trade:
            return self.coins * event.neighbours_traded
        return self.coins

    def counts(self) -> Optional[Dict[str, object]]:
        return {
            'what': 'build_income',
            'scope': 'self',
            'colors': [self.color] if self.color else [],
            'coins_each': self.coins,
            'via_chain': self.via_chain,
            'after_trade': self.after_trade,
        }


@dataclass
class VictoryIncome(Effect):
    """Coins each time the owner wins a military conflict token (Nero)."""

    coins: int = 2

    def coins_per_victory(self) -> int:
        return self.coins


@dataclass
class RedirectDefeats(Effect):
    """The owner's defeat tokens go to the neighbour who beat them (Tomyris)."""

    def redirects_defeats(self) -> bool:
        return True


@dataclass
class ResourceDiscount(Effect):
    """Pay ``amount`` fewer resource units for ``target`` (a card colour value or ``'wonder'``).

    Archimedes / Leonidas / Hammurabi: 1 off green / red / blue cards; Imhotep: 1 off
    wonder stages; Architect Firm: wonder stages need no resources (a large amount).
    The payment solver waives whichever units are dearest.
    """

    target: str
    amount: int = 1

    def resource_discount(self, target: str) -> int:
        return self.amount if target == self.target else 0

    def counts(self) -> Optional[Dict[str, object]]:
        return {'what': 'discount', 'scope': 'self', 'colors': [self.target], 'amount': self.amount}


@dataclass
class BuildFree(Effect):
    """Build cards of ``colors`` without paying anything (Ramses: guilds, Maecenas: leaders)."""

    colors: Iterable[Color]

    def __post_init__(self) -> None:
        self.colors = frozenset(self.colors)

    def builds_free(self, card: Card) -> bool:
        return card.color in self.colors

    def counts(self) -> Optional[Dict[str, object]]:
        return {'what': 'builds_free', 'scope': 'self', 'colors': sorted(c.value for c in self.colors)}


@dataclass
class BankPurchase(Effect):
    """Once per build, buy any one resource from the bank for 1 coin (Bilkis)."""

    count: int = 1

    def bank_purchases(self) -> int:
        return self.count


@dataclass
class ProduceUnowned(Effect):
    """One extra unit per build of a resource the owner does not produce (Black Market).

    "Produce" counts only brown/grey cards and the board's starting resource.
    """

    def private_production(self, produced: frozenset) -> Tuple[frozenset, ...]:
        options = TRADEABLE - produced
        return (frozenset(options),) if options else ()


@dataclass
class ProduceOwned(Effect):
    """One extra unit per build of a resource the owner already produces (Secret Warehouse).

    "Produces" counts only brown/grey cards and the board's starting resource.
    """

    def private_production(self, produced: frozenset) -> Tuple[frozenset, ...]:
        options = produced & TRADEABLE
        return (frozenset(options),) if options else ()


@dataclass
class StartingResourceDiscount(Effect):
    """Buy your wonder's starting resource ``amount`` coins cheaper (Smugglers Cache)."""

    amount: int = 1

    def trade_reduction(self, player, side: str, resource: Ressource) -> int:
        return self.amount if resource is player.wonder.starting_resource else 0


@dataclass
class CashDefeats(Effect):
    """Gain ``coins_each`` per defeat token, then discard them all (Memorial)."""

    coins_each: int = 2

    def on_build(self, ctx: Context) -> None:
        defeats = sum(1 for t in ctx.player.tokens if t < 0)
        ctx.player.coins += self.coins_each * defeats
        ctx.player.tokens = [t for t in ctx.player.tokens if t >= 0]


@dataclass
class PlunderPer(Effect):
    """Each target loses ``amount`` coins per unit of a :class:`Tally` metric; shortfall is debt.

    Sepulcher / Cenotaph: per military token; Chamber of Builders: per wonder stage.
    """

    what: str
    amount: int = 1
    scope: str = 'others'

    def on_build(self, ctx: Context) -> None:
        metric = Tally(self.what, 'self')
        for player in _targets(ctx, self.scope):
            owed = self.amount * metric._metric(player)
            paid = min(owed, player.coins)
            player.coins -= paid
            player.debt += owed - paid


@dataclass
class CoinDiscount(Effect):
    """Cards of ``colors`` cost the owner ``amount`` coins less, and neighbours ``neighbours_amount``.

    Rome A recruits leaders free (use :class:`BuildFree`); Rome B: ``CoinDiscount({WHITE}, 2, 1)``.
    """

    colors: Iterable[Color]
    amount: int = 0
    neighbours_amount: int = 0

    def __post_init__(self) -> None:
        self.colors = frozenset(self.colors)

    def coin_discount(self, card: Card) -> int:
        return self.amount if card.color in self.colors else 0

    def neighbour_coin_discount(self, card: Card) -> int:
        return self.neighbours_amount if card.color in self.colors else 0


@dataclass
class LeaderDraw(Effect):
    """Draw ``count`` leaders from the undealt leader deck into the owner's hand (Rome B)."""

    count: int = 4

    def leader_draws(self) -> int:
        return self.count


@dataclass
class ExtraRecruit(Effect):
    """Immediately recruit one more leader from hand, paying its (discounted) cost (Rome B)."""

    count: int = 1

    def extra_recruits(self) -> int:
        return self.count


@dataclass
class CopyLeader(Effect):
    """Copy one leader recruited by a neighbour: gain its powers, now and later (Courtesans' Guild).

    The copy is not a leader card of the owner (it doesn't count for the Diplomats' Guild).
    """

    count: int = 1

    def leader_copies(self) -> int:
        return self.count
