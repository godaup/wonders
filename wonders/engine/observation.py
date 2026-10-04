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

from dataclasses import dataclass, field
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
    coins = effect.coins_on_build(ctx)
    if coins:
        out['coins'] = coins
    if effect.trade_rules():
        out['trade'] = True
    # Turn-structure / expansion powers (uniform hooks), so agents can value them too.
    for key, value in (
        ('plays_last_card', effect.plays_last_card()),
        ('free_builds_per_age', effect.free_builds_per_age()),
        ('discard_builds', effect.discard_builds()),
        ('guild_copies', effect.guild_copies()),
        ('science_copies', effect.science_copies()),
        ('science_set_bonus', effect.science_set_bonus()),
        ('coins_per_victory', effect.coins_per_victory()),
        ('redirects_defeats', effect.redirects_defeats()),
        ('bank_purchases', effect.bank_purchases()),
        ('leader_draws', effect.leader_draws()),
        ('extra_recruits', effect.extra_recruits()),
        ('leader_copies', effect.leader_copies()),
    ):
        if value:
            out[key] = value
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
    kind: str  # 'build' | 'wonder' | 'discard' | 'free'
    card: Optional[str]  # None when hidden (opponent wonder/discard)

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class WonderStageView:
    """The player's next unbuilt wonder stage (what a WONDER action would gain)."""

    index: int  # 1-based stage number
    cost: Dict[str, object]
    effects: List[Dict[str, object]]
    affordable: bool

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class WonderBoardView:
    """The seat's whole wonder board (public): starting resource and every stage."""

    name: str
    side: str
    starting_resource: Optional[str]  # None for boards producing nothing (Rome)
    stages: List[Dict[str, object]]  # per stage: index (1-based), cost, effects, built
    effects: List[Dict[str, object]]  # passive board powers (e.g. Rome's leader discounts)

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
    next_wonder_stage: Optional[WonderStageView] = None
    decision: str = 'play'  # 'play' | 'last_card' | 'discard_build' (see DecisionKind)
    free_builds_left: int = 0  # Olympia: free builds from hand still available this Age
    wonder_board: Optional[WonderBoardView] = None  # the seat's full board, for board-aware agents
    production: List[List[str]] = field(default_factory=list)  # own production slots (resource options)

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
            'next_wonder_stage': self.next_wonder_stage.to_dict() if self.next_wonder_stage else None,
            'decision': self.decision,
            'free_builds_left': self.free_builds_left,
            'wonder_board': self.wonder_board.to_dict() if self.wonder_board else None,
            'production': self.production,
        }


def _city_view(game: 'Game', seat: int, total: Optional[int] = None) -> CityView:
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
        score=game.score(seat).total if total is None else total,
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


def _filter_entry(entry, viewer: int) -> List[MoveEvent]:
    """One raw history entry ``(age, turn, moves)`` as ``viewer`` may see it."""
    age, turn, moves = entry
    events: List[MoveEvent] = []
    for seat, move in enumerate(moves):
        if move is None:  # seat did not act (extra decisions are single-seat)
            continue
        kind, card_name = move
        visible = seat == viewer or kind in (ActionKind.BUILD, ActionKind.FREE)
        events.append(MoveEvent(age, turn, seat, kind.value, card_name if visible else None))
    return events


def _filter_history(raw: Sequence, viewer: int) -> List[MoveEvent]:
    return [event for entry in raw for event in _filter_entry(entry, viewer)]


def public_table(game: 'Game', totals: Optional[Sequence[int]] = None) -> List[CityView]:
    """Every city's public view, in seat order (identical for every viewer)."""
    return [_city_view(game, i, None if totals is None else totals[i]) for i in range(game.n)]


def build_observation(
    game: 'Game',
    seat: int,
    hand: List[Card],
    history: Sequence = (),
    *,
    table: Optional[List[CityView]] = None,
    events: Optional[List[MoveEvent]] = None,
) -> Observation:
    """Assemble the Observation for ``seat`` given its current ``hand``.

    ``history`` is the raw public log: a sequence of ``(age, turn, moves)`` where
    ``moves`` is a per-seat list of ``(ActionKind, card_name)`` (``None`` for a seat
    that did not act); it gets filtered so the viewer only sees what it may know.

    ``table`` (from :func:`public_table`) and ``events`` (the already-filtered history
    for ``seat``) let a caller reuse work across seats and turns — see
    :class:`ObservationBuilder`; when given, ``history`` is ignored.
    """
    if table is None:
        table = public_table(game)
    if events is None:
        events = _filter_history(history, seat)
    stage_affordable = game._stage_payment(seat) is not None
    stage = game.players[seat].next_stage
    next_stage_view = None
    if stage is not None:
        ctx = game.context(seat)
        next_stage_view = WonderStageView(
            index=game.players[seat].wonder_stages_built + 1,
            cost=_cost_dict(stage),
            effects=[effect_summary(e, ctx) for e in stage.effects],
            affordable=stage_affordable,
        )
    return Observation(
        seat=seat,
        age=game.age,
        turn=game.turn,
        phase=game.phase.value,
        n_players=game.n,
        hand=[_card_view(game, seat, c) for c in hand],
        legal_actions=game.legal_actions(seat, hand),
        wonder_stage_affordable=stage_affordable,
        me=table[seat],
        left=table[(seat + 1) % game.n],
        right=table[(seat - 1) % game.n],
        table=table,
        history=events,
        discard_count=len(game.discard_pile),
        next_wonder_stage=next_stage_view,
        decision=game.decision.value,
        free_builds_left=max(0, game.players[seat].free_builds_left),
        wonder_board=_board_view(game, seat),
        production=[
            sorted(r.value for r in slot)
            for slot in game.players[seat].production_slots() + game.players[seat].private_production_slots()
        ],
    )


def _board_view(game: 'Game', seat: int) -> WonderBoardView:
    player = game.players[seat]
    wonder = player.wonder
    ctx = game.context(seat)
    return WonderBoardView(
        name=wonder.name,
        side=wonder.side,
        starting_resource=wonder.starting_resource.value if wonder.starting_resource is not None else None,
        stages=[
            {
                'index': k + 1,
                'cost': _cost_dict(stage),
                'effects': [effect_summary(e, ctx) for e in stage.effects],
                'built': k < player.wonder_stages_built,
            }
            for k, stage in enumerate(wonder.stages)
        ],
        effects=[effect_summary(e, ctx) for e in wonder.effects],
    )


class ObservationBuilder:
    """Builds Observations for a running game, reusing work between calls.

    Equivalent to calling :func:`build_observation` with the full raw history, but:

    * the public table (every city's view and score) is computed once per engine
      state (``Game.tick``) and shared by all seats deciding at that point, instead of
      once per seat (n² city views per turn);
    * each viewer's filtered history is extended incrementally instead of re-filtered
      from the start of the game on every decision.

    Feed it every decision with :meth:`record` (the raw ``(age, turn, moves)`` entry).
    Only use it while the game is advanced by the engine — mutating players directly
    does not bump ``Game.tick`` and would leave the cached table stale. Observations
    built at the same decision share their :class:`CityView` objects: treat them as
    read-only.
    """

    def __init__(self) -> None:
        self.history: List[Tuple[int, int, List]] = []
        self._events: Dict[int, List[MoveEvent]] = {}  # per viewer, filtered so far
        self._consumed: Dict[int, int] = {}  # per viewer, raw history entries filtered
        self._table: Optional[List[CityView]] = None
        self._table_key: Optional[Tuple[int, int]] = None

    def record(self, age: int, turn: int, moves: List) -> None:
        self.history.append((age, turn, moves))

    def table(self, game: 'Game') -> List[CityView]:
        key = (id(game), game.tick)
        if self._table is None or self._table_key != key:
            self._table = public_table(game, game.cached_totals())
            self._table_key = key
        return self._table

    def events(self, seat: int) -> List[MoveEvent]:
        events = self._events.setdefault(seat, [])
        for entry in self.history[self._consumed.get(seat, 0) :]:
            events.extend(_filter_entry(entry, seat))
        self._consumed[seat] = len(self.history)
        return events

    def build(self, game: 'Game', seat: int, hand: List[Card]) -> Observation:
        return build_observation(game, seat, hand, table=self.table(game), events=list(self.events(seat)))
