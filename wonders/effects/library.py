"""Concrete effects covering the base-game mechanics.

These are enough to express brown/grey production, red military, green science,
blue civilian points, yellow commercial cards (trade discounts, coin income,
end-game coin/VP-per-card) and purple guild scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

from wonders.cards.base import MANUFACTURED_GOODS, RAW_MATERIALS, Card, Color, Resource, ScienceSymbol
from wonders.effects.base import BuildEvent, Context, Effect, TradeRule

TRADEABLE = RAW_MATERIALS | MANUFACTURED_GOODS


class Production(Effect):
    """Produces one resource per turn, chosen from ``options`` each time it is used.

    A single-resource brown/grey card offers one resource; a "choice" card
    (Forum, Caravansery, many wonder starts) offers several — but still only one
    unit per turn.
    """

    def __init__(self, *resources: Resource, choice: Iterable[Resource] = ()):
        self.options = frozenset(resources) | frozenset(choice)

    def production(self) -> Tuple[frozenset, ...]:
        return (self.options,)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f'Production({sorted(r.value for r in self.options)})'


class MultiProduction(Effect):
    """Produces several resources simultaneously (e.g. a wonder giving Wood + Ore).

    Each argument is one independent production slot (itself possibly a choice).
    """

    def __init__(self, *slots: Iterable[Resource]):
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
    colors: Tuple[str, ...] = ()  # only cards of these colour values (Caligula: black); empty: any card

    def free_builds_per_age(self) -> int:
        return self.count

    def free_build_colors(self) -> Tuple[str, ...]:
        return tuple(self.colors)


@dataclass
class FreeFirstOfColor(Effect):
    """Olympia day side (2nd edition): the first card of each colour in your city costs nothing."""

    def free_first_of_color(self) -> bool:
        return True


@dataclass
class FreeTurnBuild(Effect):
    """Olympia night side (2nd edition): build the card of the Age's ``'first'`` / ``'last'`` turn for free."""

    turn: str = 'first'

    def __post_init__(self) -> None:
        if self.turn not in ('first', 'last'):
            raise ValueError(f"a free turn build is on the 'first' or 'last' turn, not {self.turn!r}")

    def free_build_turns(self) -> Tuple[str, ...]:
        return (self.turn,)


@dataclass
class WonderComplete(Effect):
    """Decorators Guild (2nd edition): victory points only if every stage of your wonder is built."""

    points: int = 7

    def victory_points(self, ctx: Context) -> int:
        player = ctx.player
        return self.points if player.wonder_stages_built >= len(player.slots) else 0


@dataclass
class BuildFromDiscard(Effect):
    """Halicarnassus: at the end of the turn, build one discarded card for free.

    ``priority`` orders several discard builds of one turn (Wonder Pack FAQ: Halicarnassus 0,
    the Great Wall 1, a stage the Manneken Pis copies 2, Solomon 3, the Courtesans' Guild 4).
    """

    count: int = 1
    priority: int = 0

    def discard_builds(self) -> int:
        return self.count

    def discard_build_priority(self) -> int:
        return self.priority


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
    * ``what='resource_symbols'`` counts the ``resource`` symbols on the production of cards of
      ``colors`` (Stonehenge: stone on brown cards; a choice card shows the symbol once).
    """

    what: str
    scope: str
    colors: Iterable[Color] = field(default_factory=frozenset)
    coins_each: int = 0
    vp_each: int = 0
    token_value: Optional[int] = None  # for what='victory_tokens': only count tokens of this value
    resource: Optional[Resource] = None  # for what='resource_symbols'

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
        if self.what == 'resource_symbols':
            return sum(
                1
                for c in player.cards
                if c.color in self.colors
                for e in c.effects
                for slot in e.production()
                if self.resource in slot
            )
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
class BuriedColorTally(Effect):
    """Stonehenge B: at the end, points per card of the buried card's colour in the neighbours' cities.

    The card used to build the stage stays hidden until the end of the game, so the stage scores
    nothing before (``Context.final``) and its summary does not name the colour.
    """

    vp_each: int = 1
    scope: str = 'neighbours'
    color: Optional[Color] = None  # set when the stage is built (see :meth:`buried`)

    def buried(self, card: 'Card') -> 'Effect':
        return BuriedColorTally(self.vp_each, self.scope, card.color)

    def victory_points(self, ctx: Context) -> int:
        if not ctx.final or self.color is None:
            return 0
        return self.vp_each * sum(1 for p in _targets(ctx, self.scope) for c in p.cards if c.color is self.color)

    def counts(self) -> Optional[Dict[str, object]]:
        return {'what': 'buried_color', 'scope': self.scope, 'colors': [], 'vp_each': self.vp_each}


@dataclass
class EntombLeader(Effect):
    """Abu Simbel: entomb one of your recruited leaders — its power is lost, it scores twice its cost."""

    count: int = 1

    def leader_entombs(self) -> int:
        return self.count


@dataclass
class Plunder(Effect):
    """Cities: each target loses ``amount`` coins; any shortfall becomes debt.

    The default scope ``'others'`` is every other player at the table.
    """

    amount: int
    scope: str = 'others'
    per_age: bool = False  # lose as many coins as the current Age instead (Arsinoe)

    def on_build(self, ctx: Context) -> None:
        amount = ctx.age if self.per_age else self.amount
        for player in _targets(ctx, self.scope):
            lost = min(amount, player.coins)
            player.coins -= lost
            if ctx.debt_tokens:  # without Cities a player short of coins just pays what they have
                player.debt += amount - lost


@dataclass
class Grant(Effect):
    """Cities: each target *gains* coins (e.g. Customs, Mint force opponents to profit)."""

    amount: int
    scope: str = 'others'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            player.gain(self.amount)


@dataclass
class GainToken(Effect):
    """Immediately gain a military victory token of ``value`` (Cities Raider cards: 1/3/5).

    ``current_age``: the token of the Age it is gained in (Nitocris).
    """

    value: int = 0
    current_age: bool = False

    def on_build(self, ctx: Context) -> None:
        value = ctx.victory_value if self.current_age else self.value
        ctx.player.tokens.append(value)
        if value > 0:  # a victory token: Nero pays for every one gained
            ctx.player.gain(ctx.player.coins_per_victory())


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
    wonder: bool = False  # only for wonder stages (Octavia)

    def build_income(self, event: BuildEvent) -> int:
        if self.color is not None and event.color != self.color:
            return 0
        if self.wonder and event.color is not None:
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

    def trade_reduction(self, player, side: str, resource: Resource) -> int:
        return self.amount if resource is player.wonder.starting_resource else 0


@dataclass
class CashDefeats(Effect):
    """Gain ``coins_each`` per defeat token, then discard them all (Memorial)."""

    coins_each: int = 2

    def on_build(self, ctx: Context) -> None:
        defeats = sum(1 for t in ctx.player.tokens if t < 0)
        ctx.player.gain(self.coins_each * defeats)
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
            if ctx.debt_tokens:
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


# --- second-edition Leaders -----------------------------------------------------


@dataclass
class ScienceMost(Effect):
    """At the end, ``count`` extra science symbols of the kind you have most of (Enheduanna)."""

    count: int = 1

    def science_most(self) -> int:
        return self.count


@dataclass
class ScienceReplace(Effect):
    """At the end, change ``count`` of your science symbols into any other (Aganice)."""

    count: int = 1

    def science_replaces(self) -> int:
        return self.count


@dataclass
class Majority(Effect):
    """End VP if you have more of something than each neighbour, counted separately.

    ``what='colors'``: cards of ``colors`` (Phryne: blue, Cornelia: yellow, ...); ``'coins'``: coins (Makeda).
    """

    what: str
    points: int = 5
    colors: Iterable[Color] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.colors = frozenset(self.colors)
        if self.what not in ('colors', 'coins'):
            raise ValueError(f"a majority counts 'colors' or 'coins', not {self.what!r}")

    def _metric(self, player) -> int:
        if self.what == 'coins':
            return player.coins
        return sum(1 for c in player.cards if c.color in self.colors)

    def victory_points(self, ctx: Context) -> int:
        mine = self._metric(ctx.player)
        return self.points if all(mine > self._metric(p) for p in ctx.neighbours) else 0


@dataclass
class NoDefeats(Effect):
    """End VP if you hold no military defeat token (Cynisca)."""

    points: int = 6

    def victory_points(self, ctx: Context) -> int:
        return 0 if any(t < 0 for t in ctx.player.tokens) else self.points


@dataclass
class TokenPairs(Effect):
    """End VP: each pair of identical military victory tokens scores their value once more (Gorgo)."""

    def victory_points(self, ctx: Context) -> int:
        counts: Dict[int, int] = {}
        for t in ctx.player.tokens:
            if t > 0:
                counts[t] = counts.get(t, 0) + 1
        return sum(value * (n // 2) for value, n in counts.items())


@dataclass
class OnlyLeader(Effect):
    """End VP if this is the only leader face up in your city (Agrippina)."""

    points: int = 7

    def victory_points(self, ctx: Context) -> int:
        return self.points if sum(1 for c in ctx.player.cards if c.color is Color.WHITE) == 1 else 0


@dataclass
class TokenPurge(Effect):
    """Discard all your military defeat tokens; every other player discards one victory token (Telesilla).

    The others choose which: each gives up its lowest-valued one.
    """

    def on_build(self, ctx: Context) -> None:
        ctx.player.tokens = [t for t in ctx.player.tokens if t >= 0]
        for player in ctx.others:
            victories = [t for t in player.tokens if t > 0]
            if victories:
                player.tokens.remove(min(victories))


@dataclass
class ReserveBonus(Effect):
    """Once per turn, gaining coins from the reserve earns ``coins`` more (Berenice)."""

    coins: int = 1

    def reserve_bonus(self) -> int:
        return self.coins


@dataclass
class StagePlunder(Effect):
    """Each time you build a wonder stage, every other player loses ``amount`` coins (Octavia)."""

    amount: int = 1

    def stage_plunder(self) -> int:
        return self.amount
