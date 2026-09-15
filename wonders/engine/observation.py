"""The environment state handed to an agent.

An :class:`Observation` is everything a single seat may legally see at decision
time: the full public tableaus of every city, the agent's own (private) hand, the
set of legal actions, and a filtered public move history. 7 Wonders is a
partial-information game — opponents' hands are hidden, and cards played to a wonder
or discarded are public *events* but their identity is hidden — so the history is
filtered per viewer.

The Observation is representation-agnostic: it is a structured object with a
``to_dict()`` view. Turning it into a tensor/feature vector is the job of an
:mod:`wonders.agents.encoders` encoder, so different representations can be tried
without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from wonders.cards.base import Card
from wonders.engine.actions import Action, ActionKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.effects.base import Context
    from wonders.engine.game import Game
    from wonders.engine.player import Player


def effect_summary(effect, ctx: 'Context') -> Dict[str, object]:
    """Structured, JSON-able summary of one effect (from the uniform hooks)."""
    out: Dict[str, object] = {'type': type(effect).__name__}
    production = effect.production()
    if production:
        out['production'] = [sorted(r.value for r in slot) for slot in production]
    if effect.shields():
        out['shields'] = effect.shields()
    symbol = effect.science()
    if symbol is not None:
        out['science'] = symbol.value
    if effect.science_wild():
        out['science_wild'] = effect.science_wild()
    vp = effect.victory_points(ctx)
    if vp:
        out['victory_points'] = vp
    return out


def _cost_dict(card_or_stage) -> Dict[str, object]:
    cost = card_or_stage.cost
    return {'coins': cost.coins, 'resources': {r.value: n for r, n in cost.resources.items()}}


@dataclass
class CardView:
    """A card in the agent's hand, with what it would cost/do right now."""

    name: str
    color: str
    cost: Dict[str, object]
    chain_requires: Optional[str]
    chain_enables: Tuple[str, ...]
    effects: List[Dict[str, object]]
    can_build: bool
    build_coin_cost: Optional[int]  # coins (bank + trade) to build now, if affordable

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class CityView:
    """Public state of one player's city — visible to everyone."""

    seat: int
    name: str
    coins: int
    wonder: str
    wonder_side: str
    stages_built: int
    stages_total: int
    shields: int
    military_tokens: int
    cards: List[str]
    cards_by_color: Dict[str, int]
    science: Dict[str, int]
    score: int

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class MoveEvent:
    """One past decision, as this viewer is allowed to see it."""

    age: int
    turn: int
    seat: int
    kind: str  # 'build' | 'wonder' | 'discard'
    card: Optional[str]  # None when hidden (opponent wonder/discard)

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class Observation:
    """Full per-seat environment state at a decision point."""

    seat: int
    age: int
    turn: int
    phase: str
    n_players: int
    hand: List[CardView]
    legal_actions: List[Action]
    wonder_stage_affordable: bool
    me: CityView
    left: CityView
    right: CityView
    table: List[CityView]  # every city (public), seat order
    history: List[MoveEvent]
    discard_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            'seat': self.seat,
            'age': self.age,
            'turn': self.turn,
            'phase': self.phase,
            'n_players': self.n_players,
            'hand': [c.to_dict() for c in self.hand],
            'legal_actions': [{'kind': a.kind.value, 'card': a.card.name} for a in self.legal_actions],
            'wonder_stage_affordable': self.wonder_stage_affordable,
            'me': self.me.to_dict(),
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
            'table': [c.to_dict() for c in self.table],
            'history': [m.to_dict() for m in self.history],
            'discard_count': self.discard_count,
        }


def _city_view(game: 'Game', seat: int) -> CityView:
    from collections import Counter

    p: 'Player' = game.players[seat]
    return CityView(
        seat=seat,
        name=p.name,
        coins=p.coins,
        wonder=p.wonder.name,
        wonder_side=p.wonder.side,
        stages_built=p.wonder_stages_built,
        stages_total=len(p.wonder.stages),
        shields=p.shields(),
        military_tokens=sum(p.tokens),
        cards=[c.name for c in p.cards],
        cards_by_color=dict(Counter(c.color.value for c in p.cards)),
        science={s.value: n for s, n in p.science().items()},
        score=game.score(seat).total,
    )


def _card_view(game: 'Game', seat: int, card: Card) -> CardView:
    ctx = game.context(seat)
    payment = game._card_payment(seat, card)  # (coins_to_bank, Payment) | None
    build_coin_cost = None
    if payment is not None:
        build_coin_cost = payment[0] + payment[1].total
    return CardView(
        name=card.name,
        color=card.color.value,
        cost=_cost_dict(card),
        chain_requires=card.chain_requires,
        chain_enables=card.chain_enables,
        effects=[effect_summary(e, ctx) for e in card.effects],
        can_build=payment is not None,
        build_coin_cost=build_coin_cost,
    )


def _filter_history(raw: Sequence, viewer: int) -> List[MoveEvent]:
    events: List[MoveEvent] = []
    for age, turn, moves in raw:
        for seat, (kind, card_name) in enumerate(moves):
            visible = seat == viewer or kind is ActionKind.BUILD
            events.append(MoveEvent(age, turn, seat, kind.value, card_name if visible else None))
    return events


def build_observation(
    game: 'Game',
    seat: int,
    hand: List[Card],
    history: Sequence = (),
) -> Observation:
    """Assemble the Observation for ``seat`` given its current ``hand``.

    ``history`` is the raw public log: a sequence of ``(age, turn, moves)`` where
    ``moves`` is a per-seat list of ``(ActionKind, card_name)``; it gets filtered so
    the viewer only sees what it may legally know.
    """
    table = [_city_view(game, i) for i in range(game.n)]
    stage_affordable = game._stage_payment(seat) is not None
    return Observation(
        seat=seat,
        age=game.age,
        turn=len([1 for age, _t, _m in history if age == game.age]),
        phase=game.phase.value,
        n_players=game.n,
        hand=[_card_view(game, seat, c) for c in hand],
        legal_actions=game.legal_actions(seat, hand),
        wonder_stage_affordable=stage_affordable,
        me=table[seat],
        left=table[(seat + 1) % game.n],
        right=table[(seat - 1) % game.n],
        table=table,
        history=_filter_history(history, seat),
        discard_count=len(game.discard_pile),
    )
