"""Concrete effects covering the base-game mechanics.

These are enough to express brown/grey production, red military, green science,
blue civilian points, yellow commercial cards (trade discounts, coin income,
end-game coin/VP-per-card) and purple guild scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from wonders.cards.base import Color, Ressource, ScienceSymbol
from wonders.effects.base import Context, Effect, TradeRule


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


def _targets(ctx: Context, scope: str):
    if scope == 'self':
        return (ctx.player,)
    if scope == 'neighbours':
        return (ctx.left, ctx.right)
    if scope == 'both':
        return (ctx.player, ctx.left, ctx.right)
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

    Scope ``'neighbours'`` approximates "all other players" (exact for 3 players).
    """

    amount: int
    scope: str = 'neighbours'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            lost = min(self.amount, player.coins)
            player.coins -= lost
            player.debt += self.amount - lost


@dataclass
class Grant(Effect):
    """Cities: each target *gains* coins (e.g. Customs, Mint force opponents to profit)."""

    amount: int
    scope: str = 'neighbours'

    def on_build(self, ctx: Context) -> None:
        for player in _targets(ctx, self.scope):
            player.coins += self.amount


@dataclass
class GainToken(Effect):
    """Cities Raider cards: immediately gain a military victory token of ``value`` (1/3/5)."""

    value: int

    def on_build(self, ctx: Context) -> None:
        ctx.player.tokens.append(self.value)
