"""Load card & wonder content from JSON data files and build decks.

Content lives in ``wonders/cards/data/<expansion>.json`` as declarative data, not
Python. This module parses it into :class:`~wonders.cards.base.Card` /
:class:`~wonders.cards.wonder.Wonder` objects (mapping a small typed *effect spec*
to concrete :class:`~wonders.effects.base.Effect` instances) and assembles the
per-Age decks for a given player count and expansion set.

Deck composition follows the game: a non-guild card contributes
``card.copies_for(n_players)`` copies (its ``at_players`` thresholds ≤ N), and the
Age III deck additionally gets ``n_players + 2`` guild cards drawn at random.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from importlib import resources
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from wonders.cards.base import Card, Color, Cost, Resource, ScienceSymbol
from wonders.cards.wonder import TRACK_SPACES, TRACKS, Shipyard, Wonder, WonderStage
from wonders.effects.base import Effect
from wonders.effects.library import (
    BankPurchase,
    BestColor,
    Boarding,
    BuildFree,
    BuildFromDiscard,
    BuildIncome,
    BuriedColorTally,
    CashDefeats,
    CoinDiscount,
    CoinLossImmunity,
    Coins,
    ColorSets,
    CommercialLevel,
    CopyGuild,
    CopyLeader,
    Debt,
    Diplomacy,
    EntombLeader,
    Explore,
    ExtraRecruit,
    FarTrade,
    FleetIncome,
    FleetMove,
    FreeBuildPerAge,
    FreeFirstOfColor,
    FreeTurnBuild,
    GainToken,
    Grant,
    IslandDraw,
    LeaderDraw,
    Majority,
    Military,
    MultiProduction,
    NavalOptOut,
    NavalStrength,
    NoDefeats,
    OnlyLeader,
    Pirates,
    PlayLastCard,
    Plunder,
    PlunderPer,
    ProduceOwned,
    ProduceUnowned,
    Production,
    RedirectDefeats,
    ReserveBonus,
    ResourceDiscount,
    Science,
    ScienceCopy,
    ScienceMost,
    ScienceReplace,
    ScienceSetBonus,
    ScienceWild,
    Special,
    StagePlunder,
    StartingResourceDiscount,
    Tally,
    Tax,
    TokenPairs,
    TokenPurge,
    Trade,
    VictoryIncome,
    VictoryPoints,
    WonderComplete,
    WonderFleetChoice,
)

DATA_PACKAGE = 'wonders.cards'
DATA_DIR = 'data'
GUILDS_OVER_PLAYERS = 2  # Age III uses (players + 2) guild cards


# --- effect spec registry ------------------------------------------------


def _production(spec: dict) -> Effect:
    resources = [Resource(r) for r in spec.get('resources', [])]
    choice = [Resource(r) for r in spec.get('choice', [])]
    return Production(*resources, choice=choice)


def _multi_production(spec: dict) -> Effect:
    slots = [[Resource(r) for r in slot] for slot in spec['slots']]
    return MultiProduction(*slots)


def _tally(spec: dict) -> Effect:
    return Tally(
        what=spec['what'],
        scope=spec['scope'],
        colors=[Color(c) for c in spec.get('colors', [])],
        coins_each=spec.get('coins_each', 0),
        vp_each=spec.get('vp_each', 0),
        token_value=spec.get('token_value'),
        resource=Resource(spec['resource']) if spec.get('resource') else None,
    )


EFFECT_BUILDERS: Dict[str, Callable[[dict], Effect]] = {
    'production': _production,
    'multi_production': _multi_production,
    'coins': lambda s: Coins(s['amount']),
    'military': lambda s: Military(s['shields']),
    'science': lambda s: Science(ScienceSymbol(s['symbol'])),
    'science_wild': lambda s: ScienceWild(s.get('count', 1)),
    'trade': lambda s: Trade(
        side=s.get('side', 'both'),
        category=s.get('category', 'both'),
        price=s.get('price', 1),
    ),
    'victory_points': lambda s: VictoryPoints(s['points']),
    'tally': _tally,
    'special': lambda s: Special(s['power'], s.get('description', '')),
    'debt': lambda s: Debt(s['amount'], s.get('scope', 'self')),
    'diplomacy': lambda s: Diplomacy(s.get('count', 1)),
    'plunder': lambda s: Plunder(s.get('amount', 0), s.get('scope', 'others'), bool(s.get('per_age', False))),
    'grant': lambda s: Grant(s['amount'], s.get('scope', 'others')),
    'gain_token': lambda s: GainToken(s.get('value', 0), bool(s.get('current_age', False))),
    'play_last_card': lambda s: PlayLastCard(),
    'free_build': lambda s: FreeBuildPerAge(s.get('count', 1), tuple(s.get('colors', ()))),
    'build_from_discard': lambda s: BuildFromDiscard(s.get('count', 1), s.get('priority', 0)),
    'copy_guild': lambda s: CopyGuild(s.get('count', 1)),
    # Leaders / Cities powers
    'color_sets': lambda s: ColorSets([Color(c) for c in s['colors']], s['vp_each']),
    'science_set_bonus': lambda s: ScienceSetBonus(s.get('points', 3)),
    'science_copy': lambda s: ScienceCopy(s.get('count', 1)),
    'build_income': lambda s: BuildIncome(
        s['coins'],
        color=s.get('color'),
        via_chain=s.get('via_chain', False),
        after_trade=s.get('after_trade', False),
        wonder=s.get('wonder', False),
    ),
    'victory_income': lambda s: VictoryIncome(s.get('coins', 2)),
    'redirect_defeats': lambda s: RedirectDefeats(),
    'resource_discount': lambda s: ResourceDiscount(s['target'], s.get('amount', 1)),
    'build_free': lambda s: BuildFree([Color(c) for c in s['colors']]),
    'bank_purchase': lambda s: BankPurchase(s.get('count', 1)),
    'produce_unowned': lambda s: ProduceUnowned(),
    'produce_owned': lambda s: ProduceOwned(),
    'starting_resource_discount': lambda s: StartingResourceDiscount(s.get('amount', 1)),
    'cash_defeats': lambda s: CashDefeats(s.get('coins_each', 2)),
    'plunder_per': lambda s: PlunderPer(s['what'], s.get('amount', 1), s.get('scope', 'others')),
    'coin_discount': lambda s: CoinDiscount(
        [Color(c) for c in s['colors']], s.get('amount', 0), s.get('neighbours_amount', 0)
    ),
    'leader_draw': lambda s: LeaderDraw(s.get('count', 4)),
    'extra_recruit': lambda s: ExtraRecruit(s.get('count', 1)),
    'copy_leader': lambda s: CopyLeader(s.get('count', 1)),
    # Wonder Pack powers
    'buried_color': lambda s: BuriedColorTally(s.get('vp_each', 1), s.get('scope', 'neighbours')),
    'entomb_leader': lambda s: EntombLeader(s.get('count', 1)),
    'free_first_of_color': lambda s: FreeFirstOfColor(),
    'free_turn_build': lambda s: FreeTurnBuild(s.get('turn', 'first')),
    'wonder_complete': lambda s: WonderComplete(s.get('points', 7)),
    'science_most': lambda s: ScienceMost(s.get('count', 1)),
    'science_replace': lambda s: ScienceReplace(s.get('count', 1)),
    'majority': lambda s: Majority(s['what'], s.get('points', 5), [Color(c) for c in s.get('colors', [])]),
    'no_defeats': lambda s: NoDefeats(s.get('points', 6)),
    'token_pairs': lambda s: TokenPairs(),
    'only_leader': lambda s: OnlyLeader(s.get('points', 7)),
    'token_purge': lambda s: TokenPurge(),
    'reserve_bonus': lambda s: ReserveBonus(s.get('coins', 1)),
    'stage_plunder': lambda s: StagePlunder(s.get('amount', 1)),
    # Armada powers
    'naval_strength': lambda s: NavalStrength(s['strength']),
    'commercial_level': lambda s: CommercialLevel(s['level']),
    'tax': lambda s: Tax(s['amount']),
    'explore': lambda s: Explore(s['level']),
    'island_draw': lambda s: IslandDraw(s['level']),
    'advance_fleet': lambda s: FleetMove(s.get('track', 'any'), s.get('count', 1)),
    'boarding': lambda s: Boarding(s['target']),
    'far_trade': lambda s: FarTrade(s['target'], s.get('count', 1), s.get('price', 1)),
    'pirates': lambda s: Pirates(),
    'coin_loss_immunity': lambda s: CoinLossImmunity(),
    'naval_opt_out': lambda s: NavalOptOut(),
    'fleet_income': lambda s: FleetIncome(s.get('coins', 1)),
    'wonder_fleet_choice': lambda s: WonderFleetChoice(),
    'best_color': lambda s: BestColor(s.get('vp_each', 1)),
}


def build_effect(spec: dict) -> Effect:
    """Turn one effect spec dict into an :class:`Effect`."""
    try:
        builder = EFFECT_BUILDERS[spec['type']]
    except KeyError:
        raise ValueError(f"unknown effect type {spec.get('type')!r}") from None
    effect = builder(spec)
    if 'description' in spec and not isinstance(effect, Special):
        effect.label = spec['description']  # type: ignore[attr-defined]  # shown by the card renderer
    return effect


# --- parsing -------------------------------------------------------------


def build_cost(spec: Optional[dict]) -> Cost:
    if not spec:
        return Cost()
    resources = Counter({Resource(r): n for r, n in spec.get('resources', {}).items()})
    return Cost(coins=spec.get('coins', 0), resources=resources)


def build_card(spec: dict, expansion: str) -> Card:
    return Card(
        name=spec['name'],
        color=Color(spec['color']),
        cost=build_cost(spec.get('cost')),
        effects=tuple(build_effect(e) for e in spec.get('effects', [])),
        chain_requires=spec.get('chain_requires'),
        chain_enables=tuple(spec.get('chain_enables', [])),
        age=spec.get('age'),
        expansion=spec.get('expansion', expansion),
        at_players=tuple(spec.get('at_players', [])),
        guild=spec.get('guild', False),
        age_cost=bool(spec.get('age_cost', False)),
        deck=spec.get('deck', 'age'),
        level=spec.get('level'),
    )


def build_island(spec: dict, expansion: str) -> Card:
    """An Armada island: a card of the ``island`` colour with a ``level`` (1-3), kept apart from the Age decks."""
    return build_card({**spec, 'color': spec.get('color', 'island'), 'deck': 'island'}, expansion)


def fleet_cards(expansion: str) -> List[Card]:
    """The four fleets as cards (``Red Fleet`` ...): a choice of fleet to advance is ``Action.build`` of one."""
    return [Card(f'{t.title()} Fleet', Color(t), expansion=expansion, deck='fleet') for t in TRACKS]


def build_shipyard(spec: dict, tracks: Dict[str, List[Tuple[Effect, ...]]]) -> Shipyard:
    costs = {t: tuple(build_cost(c) for c in spec['costs'][t]) for t in TRACKS}
    for track, track_costs in costs.items():
        if len(track_costs) != TRACK_SPACES:
            raise ValueError(f'shipyard {spec["name"]!r}: {track} needs {TRACK_SPACES} costs, not {len(track_costs)}')
    if spec['wonder_track'] not in TRACKS:
        raise ValueError(f'shipyard {spec["name"]!r}: unknown wonder track {spec["wonder_track"]!r}')
    return Shipyard(
        spec['name'], spec['wonder_track'], costs, {t: tuple(tracks[t]) for t in TRACKS}, spec.get('official', True)
    )


def build_tracks(raw: Dict[str, list]) -> Dict[str, List[Tuple[Effect, ...]]]:
    """The shared fleet spaces of every shipyard: ``{track: [effects of space 1, ..., space 6]}``."""
    tracks = {t: [tuple(build_effect(e) for e in space) for space in raw[t]] for t in TRACKS}
    for track, spaces in tracks.items():
        if len(spaces) != TRACK_SPACES:
            raise ValueError(f'track {track} needs {TRACK_SPACES} spaces, not {len(spaces)}')
    return tracks


def _stage_copy(spec: Optional[dict]) -> Optional[Tuple[str, Union[int, str]]]:
    """``{"side": "left", "stage": 1}`` (or ``"stage": "last"``): the stage copies a neighbour's."""
    if spec is None:
        return None
    side, which = spec['side'], spec['stage']
    if side not in ('left', 'right'):
        raise ValueError(f'a copied stage comes from the left or right neighbour, not {side!r}')
    if which != 'last' and not (isinstance(which, int) and which >= 1):
        raise ValueError(f'a copied stage is a stage number (1, 2, ...) or "last", not {which!r}')
    return (side, which)


def build_wonder(spec: dict) -> Wonder:
    stages = tuple(
        WonderStage(
            cost=build_cost(stage.get('cost')),
            effects=tuple(build_effect(e) for e in stage.get('effects', [])),
            copy=_stage_copy(stage.get('copy')),
        )
        for stage in spec['stages']
    )
    start = spec.get('starting_resource')
    return Wonder(
        name=spec['name'],
        starting_resource=Resource(start) if start else None,
        stages=stages,
        side=spec.get('side', 'A'),
        effects=tuple(build_effect(e) for e in spec.get('effects', [])),
        any_order=bool(spec.get('any_order', False)),
        extra_coins=int(spec.get('extra_coins', 0)),
        requires=tuple(spec.get('requires', ())),
    )


# --- content -------------------------------------------------------------


@dataclass
class Content:
    """Parsed cards and wonders from one or more expansions."""

    cards: List[Card] = field(default_factory=list)
    wonders: List[Wonder] = field(default_factory=list)
    # Armada: islands (dealt by exploration), the four fleets (choices of fleet) and the Shipyard boards.
    islands: List[Card] = field(default_factory=list)
    fleets: List[Card] = field(default_factory=list)
    shipyards: List[Shipyard] = field(default_factory=list)

    def wonder(self, name: str, side: Optional[str] = None) -> Wonder:
        for w in self.wonders:
            if w.name == name and (side is None or w.side == side):
                return w
        raise KeyError((name, side))

    def card(self, name: str) -> Card:
        for c in self.cards:
            if c.name == name:
                return c
        for c in self.islands + self.fleets:
            if c.name == name:
                return c
        raise KeyError(name)

    def shipyard(self, name: str) -> Shipyard:
        for s in self.shipyards:
            if s.name == name:
                return s
        raise KeyError(name)

    def choices(self) -> List[Card]:
        """Every card an action may name: the Age cards and leaders, islands and fleets (the action vocabulary)."""
        return self.cards + self.islands + self.fleets

    def guilds(self) -> List[Card]:
        return [c for c in self.cards if c.guild]

    def leaders(self) -> List[Card]:
        """Leader cards (white) — drafted/recruited separately, not dealt into Ages."""
        return [c for c in self.cards if c.color is Color.WHITE]


def read_expansion(expansion: str) -> dict:
    """The raw JSON of a bundled expansion."""
    path = resources.files(DATA_PACKAGE).joinpath(DATA_DIR, f'{expansion}.json')
    return json.loads(path.read_text(encoding='utf-8'))


def define_colors(raw: dict) -> List[Color]:
    """Define the card colours an expansion file declares (``"colors"``); call before building its cards."""
    return [Color.from_spec(spec) for spec in raw.get('colors', [])]


def load_content(*expansions: str) -> Content:
    """Load and merge one or more expansions (default: ``'base'``), bundled or from extensions."""
    from wonders.extensions import REGISTRY, ensure_loaded, provide

    ensure_loaded()
    provide(expansions)  # a catalog expansion that is switched off (e.g. a saved game's)
    from wonders.ruleset import family_of

    content = Content()
    # What a card or board may require: an expansion in any edition (Abu Simbel needs Leaders, the
    # second edition's bonus leaders need Cities).
    loaded = {family_of(e) for e in expansions or ('base',)}
    for expansion in expansions or ('base',):
        raw = REGISTRY.expansion_raw(expansion) or read_expansion(expansion)
        define_colors(raw)
        content.wonders.extend(build_wonder(w) for w in raw.get('wonders', []))
        cards = [c for c in raw.get('cards', []) if {family_of(r) for r in c.get('requires', ())} <= loaded]
        content.cards.extend(build_card(c, expansion) for c in cards)
        if raw.get('tracks'):  # Armada
            tracks = build_tracks(raw['tracks'])
            content.shipyards.extend(build_shipyard(y, tracks) for y in raw.get('shipyards', []))
            content.islands.extend(build_island(c, expansion) for c in raw.get('islands', []))
            content.fleets.extend(fleet_cards(expansion))
    content.wonders = [w for w in content.wonders if {family_of(r) for r in w.requires} <= loaded]
    return content


# --- deck building -------------------------------------------------------


def build_age_deck(
    content: Content,
    age: int,
    n_players: int,
    rng: Optional[random.Random] = None,
    extra_guilds: int = GUILDS_OVER_PLAYERS,
) -> List[Card]:
    """Assemble the shuffled deck for ``age`` at ``n_players``.

    Non-guild cards contribute ``copies_for(n_players)`` copies; Age III also gets
    ``n_players + 2`` randomly chosen guild cards. ``rng`` makes the guild draw and
    shuffle reproducible (important for training/self-play).
    """
    rng = rng or random.Random()
    deck: List[Card] = []
    for card in content.cards:
        if card.guild or card.color in (Color.WHITE, Color.BLACK) or card.age != age or card.deck != 'age':
            continue  # guilds/black/drawn handled below; leaders (white) are drafted separately
        deck.extend([card] * card.copies_for(n_players))

    if age == 3:
        guilds = content.guilds()
        want = n_players + extra_guilds
        deck.extend(guilds if want >= len(guilds) else rng.sample(guilds, want))

    # Cities: draw exactly n_players random black cards from this Age's pool (no
    # per-card numerals). This adds one card per player per Age (deal 8 not 7).
    black = [c for c in content.cards if c.color is Color.BLACK and c.age == age]
    if black:
        deck.extend(black if n_players >= len(black) else rng.sample(black, n_players))

    # Armada: as many random cards of this Age as players, from those used at this player count (4+ cards
    # leave a 3-player game), one more per player again.
    drawn = [c for c in content.cards if c.deck == 'draw' and c.age == age and c.copies_for(n_players)]
    if drawn:
        deck.extend(drawn if n_players >= len(drawn) else rng.sample(drawn, n_players))

    rng.shuffle(deck)
    return deck


LEADERS_PER_PLAYER = 4


def deal_leaders(content: 'Content', n_players: int, rng: random.Random) -> Tuple[List[List[Card]], List[Card]]:
    """Shuffle the leaders; deal ``LEADERS_PER_PLAYER`` to each seat for the draft.

    Returns ``(hands, box)``: ``box`` is the undealt rest, drawn from its end (Rome B).
    """
    deck = list(content.leaders())
    rng.shuffle(deck)
    if len(deck) < LEADERS_PER_PLAYER * n_players:
        raise ValueError(f'not enough leaders for {n_players} players')
    hands = [[deck.pop() for _ in range(LEADERS_PER_PLAYER)] for _ in range(n_players)]
    return hands, deck


def deal_hands(deck: Sequence[Card], n_players: int) -> List[List[Card]]:
    """Split a deck into ``n_players`` equal hands (drops any remainder)."""
    size = len(deck) // n_players
    return [list(deck[k * size : (k + 1) * size]) for k in range(n_players)]
