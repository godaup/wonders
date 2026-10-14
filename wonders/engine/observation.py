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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from wonders.cards.base import MANUFACTURED_GOODS, RAW_MATERIALS, Card
from wonders.engine.actions import Action, ActionKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.effects.base import Context
    from wonders.engine.game import Game
    from wonders.engine.player import Player


# Context-independent parts of effect summaries, per effect object (effects are immutable
# content shared by every game, so this is safe to keep for the process lifetime).
_STATIC_SUMMARY: Dict[int, Tuple[object, Dict[str, object]]] = {}
_TRADEABLE = sorted(RAW_MATERIALS | MANUFACTURED_GOODS, key=lambda r: r.value)


def effect_summary(effect, ctx: 'Context') -> Dict[str, object]:
    """Structured, JSON-able summary of one effect (from the uniform hooks)."""
    hit = _STATIC_SUMMARY.get(id(effect))
    if hit is None or hit[0] is not effect:  # holding the effect keeps its id from being reused
        hit = (effect, _static_summary(effect))
        _STATIC_SUMMARY[id(effect)] = hit
    out = dict(hit[1])
    vp = effect.victory_points(ctx)
    if vp:
        out['victory_points'] = vp
    coins = effect.coins_on_build(ctx)
    if coins:
        out['coins'] = coins
    return out


def _static_summary(effect) -> Dict[str, object]:
    """The summary keys that don't depend on the game state."""
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
    counts = effect.counts()
    if counts:
        out['counts'] = counts  # what the effect pays per unit (public card text), for combo-aware agents
    return out


def _cost_dict(card_or_stage) -> Dict[str, object]:
    return _cost_value(card_or_stage.cost)


def _cost_value(cost) -> Dict[str, object]:
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
    chain_free: bool = False  # buildable for free through a chain
    bank_coins: int = 0  # of build_coin_cost: coins paid to the bank (card cost, Bilkis)
    trade_coins: int = 0  # of build_coin_cost: coins paid to neighbours

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
    tokens: List[int] = field(default_factory=list)  # military tokens (1/3/5 victories, -1 defeats)
    debt: int = 0  # Cities debt tokens
    diplomacy: int = 0  # Cities diplomacy tokens held
    science_wilds: int = 0  # choose-any science symbols
    production: List[List[str]] = field(default_factory=list)  # public production slots (options each)
    trade_production: List[List[str]] = field(default_factory=list)  # the slots neighbours may buy from
    private_slots: int = 0  # own-use production slots (Black Market / Secret Warehouse)
    score_breakdown: Dict[str, int] = field(default_factory=dict)  # ScoreBreakdown.as_dict()
    copied_leaders: List[str] = field(default_factory=list)  # Courtesans' Guild copies
    next_stage_cost: Optional[Dict[str, object]] = None  # cost of the next unbuilt wonder stage
    # Built stages in build order as [board stage index, variant] (the Great Wall builds in any order; a
    # Manneken Pis stage copying the Great Wall says which of its stages).
    stage_slots: List[List[int]] = field(default_factory=list)
    entombed: List[str] = field(default_factory=list)  # Abu Simbel: leaders placed face down (public)
    # Armada (public): the Shipyard, each fleet's space (0-6), islands in play, naval strength, commercial
    # level, naval tokens and the Boarding tokens received this Age.
    shipyard: Optional[str] = None
    fleets: Dict[str, int] = field(default_factory=dict)
    islands: List[str] = field(default_factory=list)
    naval_strength: int = 0
    commercial_level: int = 0
    naval_tokens: List[int] = field(default_factory=list)
    boardings: int = 0

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
    leaders_hand: List[str] = field(default_factory=list)  # own unplayed leaders (private)
    seen_cards: List[str] = field(default_factory=list)  # cards seen in earlier hands this Age, now elsewhere
    buy_prices: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {'left'/'right': {resource: coins}}
    expansions: List[str] = field(default_factory=list)  # expansions in play
    pass_left: bool = True  # this Age's hands pass to the left (clockwise)
    # What a WONDER action may build, ``Action.staged(index)`` picks one: per option index, the board
    # stage (1-based), cost, effects and affordable. One entry unless the board offers a choice.
    stage_options: List[Dict[str, object]] = field(default_factory=list)
    # Armada: the cost of each own fleet's next Naval Construction (None at the end), the fleet that goes with
    # the wonder (None: none) and how many islands are left per level.
    naval_costs: Dict[str, Optional[Dict[str, object]]] = field(default_factory=dict)
    wonder_track: Optional[str] = None
    island_decks: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            'seat': self.seat,
            'age': self.age,
            'turn': self.turn,
            'phase': self.phase,
            'n_players': self.n_players,
            'hand': [c.to_dict() for c in self.hand],
            'legal_actions': [
                {'kind': a.kind.value, 'card': a.card.name, **({'naval': True} if a.naval else {})}
                for a in self.legal_actions
            ],
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
            'leaders_hand': self.leaders_hand,
            'seen_cards': self.seen_cards,
            'buy_prices': self.buy_prices,
            'expansions': self.expansions,
            'pass_left': self.pass_left,
            'naval_costs': self.naval_costs,
            'wonder_track': self.wonder_track,
            'island_decks': self.island_decks,
        }


def _city_view(game: 'Game', seat: int, breakdown: Optional[Dict[str, int]] = None) -> CityView:
    from collections import Counter

    p: 'Player' = game.players[seat]
    if breakdown is None:
        breakdown = game.score(seat).as_dict()
    stage = p.next_stage
    view = CityView(
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
        score=breakdown['total'],
        tokens=list(p.tokens),
        debt=p.debt,
        diplomacy=p.diplomacy,
        science_wilds=p.science_wilds(),
        production=[sorted(r.value for r in slot) for slot in p.production_slots()],
        trade_production=[sorted(r.value for r in slot) for slot in p.tradeable_slots()],
        private_slots=len(p.private_production_slots()),
        score_breakdown=dict(breakdown),
        copied_leaders=[c.name for c in p.copied_leaders],
        next_stage_cost=_cost_dict(stage) if stage is not None else None,
        stage_slots=[[slot, variant] for slot, variant in zip(p.built_slots, p.built_variants)],
        entombed=[c.name for c in p.entombed],
    )
    if p.shipyard is not None:  # Armada
        view.shipyard = p.shipyard.name
        view.fleets = dict(p.fleets)
        view.islands = [c.name for c in p.islands]
        view.naval_strength = p.naval_strength()
        view.commercial_level = p.commercial_level()
        view.naval_tokens = list(p.naval_tokens)
        view.boardings = len(p.boarded_by)
    return view


def _card_view(game: 'Game', seat: int, card: Card) -> CardView:
    ctx = game.context(seat)
    payment = game._card_payment(seat, card)  # (coins_to_bank, Payment) | None
    build_coin_cost = None
    bank = trade = 0
    if payment is not None:
        build_coin_cost = payment[0] + payment[1].total
        trade = payment[1].traded
        bank = build_coin_cost - trade
    return CardView(
        name=card.name,
        color=card.color.value,
        cost=_cost_dict(card),
        chain_requires=card.chain_requires,
        chain_enables=card.chain_enables,
        effects=[effect_summary(e, ctx) for e in card.effects],
        can_build=payment is not None,
        build_coin_cost=build_coin_cost,
        chain_free=payment is not None and not game.players[seat].has(card.name) and game._chained(seat, card),
        bank_coins=bank,
        trade_coins=trade,
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


def public_table(game: 'Game', breakdowns: Optional[Sequence[Dict[str, int]]] = None) -> List[CityView]:
    """Every city's public view, in seat order (identical for every viewer)."""
    return [_city_view(game, i, None if breakdowns is None else breakdowns[i]) for i in range(game.n)]


def build_observation(
    game: 'Game',
    seat: int,
    hand: List[Card],
    history: Sequence = (),
    *,
    table: Optional[List[CityView]] = None,
    events: Optional[List[MoveEvent]] = None,
    seen_cards: Sequence[str] = (),
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
    default = game._stage_payment(seat)
    stage_affordable = default is not None
    options = game.stage_options(seat)
    ctx = game.context(seat)
    stage_options = [
        {
            'index': k,
            'stage': slot + 1,
            'cost': _cost_dict(stage),
            'effects': [effect_summary(e, ctx) for e in stage.effects],
            'affordable': game._stage_payment(seat, k) is not None,
        }
        for k, (slot, stage) in enumerate(options)
    ]
    next_stage_view = None
    if options:
        slot, stage = (default[0], default[1]) if default is not None else options[0]
        next_stage_view = WonderStageView(
            index=slot + 1,
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
        leaders_hand=[c.name for c in game.players[seat].leaders_hand],
        seen_cards=list(seen_cards),
        buy_prices={
            side: {r.value: game.players[seat].buy_price(side, r) for r in _TRADEABLE} for side in ('left', 'right')
        },
        expansions=list(game.expansions),
        pass_left=game.phase.pass_left if game.age else True,
        stage_options=stage_options if len(stage_options) > 1 else [],
        **_naval_view(game, seat),
    )


def _naval_view(game: 'Game', seat: int) -> Dict[str, Any]:
    """Armada's private observation fields (nothing without a Shipyard)."""
    player = game.players[seat]
    if player.shipyard is None:
        return {}
    costs = {}
    for track in player.fleets:
        cost = player.naval_cost(track)
        costs[track] = _cost_value(cost) if cost is not None else None
    return {
        'naval_costs': costs,
        'wonder_track': player.wonder_track(),
        'island_decks': {level: len(deck) for level, deck in sorted(game.island_decks.items())},
    }


def _board_view(game: 'Game', seat: int) -> WonderBoardView:
    player = game.players[seat]
    wonder = player.wonder
    ctx = game.context(seat)
    return WonderBoardView(
        name=wonder.name,
        side=wonder.side,
        starting_resource=wonder.starting_resource.value if wonder.starting_resource is not None else None,
        stages=[_stage_view(player, k, ctx) for k in range(len(wonder.stages))],
        effects=[effect_summary(e, ctx) for e in wonder.effects],
    )


def _stage_view(player: 'Player', k: int, ctx: Any) -> Dict[str, object]:
    """Board stage ``k``: the built stage, else its first possible form (``variants`` > 1: a choice)."""
    built = k in player.built_slots
    variants = player.slots[k]
    stage = player.stages_built[player.built_slots.index(k)] if built else (variants[0] if variants else None)
    copy = player.wonder.stages[k].copy
    return {
        'index': k + 1,
        'cost': _cost_dict(stage) if stage is not None else {},
        'effects': [effect_summary(e, ctx) for e in stage.effects] if stage is not None else [],
        'built': built,
        'variants': len(variants),
        'copies': f'{copy[0]} {copy[1]}' if copy is not None else None,  # e.g. 'left 1', 'right last'
    }


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
        self._hands: Dict[Tuple[int, int], Dict[int, List[str]]] = {}  # (seat, age) -> turn -> hand

    def record(self, age: int, turn: int, moves: List) -> None:
        self.history.append((age, turn, moves))

    def table(self, game: 'Game') -> List[CityView]:
        key = (id(game), game.tick)
        if self._table is None or self._table_key != key:
            self._table = public_table(game, game.cached_breakdowns())
            self._table_key = key
        return self._table

    def events(self, seat: int) -> List[MoveEvent]:
        events = self._events.setdefault(seat, [])
        for entry in self.history[self._consumed.get(seat, 0) :]:
            events.extend(_filter_entry(entry, seat))
        self._consumed[seat] = len(self.history)
        return events

    def seen_cards(self, game: 'Game', seat: int, hand: List[Card]) -> List[str]:
        """Cards ``seat`` held earlier this Age that are now in other hands (card counting).

        Each regular-turn hand is remembered; the seen cards are those earlier hands minus
        the seat's current hand and the cards it played itself. Only what this seat saw.
        """
        from collections import Counter

        if game.decision.value != 'play' or not game.age:
            return []
        hands = self._hands.setdefault((seat, game.age), {})
        hands.setdefault(game.turn, [c.name for c in hand])
        seen: Counter = Counter()
        for turn, names in hands.items():
            if turn < game.turn:
                seen.update(names)
        seen.subtract(c.name for c in hand)
        seen.subtract(e.card for e in self.events(seat) if e.seat == seat and e.age == game.age and e.card)
        return sorted(name for name, n in seen.items() for _ in range(max(0, n)))

    def build(self, game: 'Game', seat: int, hand: List[Card]) -> Observation:
        return build_observation(
            game,
            seat,
            hand,
            table=self.table(game),
            events=list(self.events(seat)),
            seen_cards=self.seen_cards(game, seat, hand),
        )
