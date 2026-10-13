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
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.base import Effect
from wonders.effects.library import (
    BankPurchase,
    BuildFree,
    BuildFromDiscard,
    BuildIncome,
    BuriedColorTally,
    CashDefeats,
    CoinDiscount,
    Coins,
    ColorSets,
    CopyGuild,
    CopyLeader,
    Debt,
    Diplomacy,
    EntombLeader,
    ExtraRecruit,
    FreeBuildPerAge,
    FreeFirstOfColor,
    FreeTurnBuild,
    GainToken,
    Grant,
    LeaderDraw,
    Military,
    MultiProduction,
    PlayLastCard,
    Plunder,
    PlunderPer,
    ProduceOwned,
    ProduceUnowned,
    Production,
    RedirectDefeats,
    ResourceDiscount,
    Science,
    ScienceCopy,
    ScienceSetBonus,
    ScienceWild,
    Special,
    StartingResourceDiscount,
    Tally,
    Trade,
    VictoryIncome,
    VictoryPoints,
    WonderComplete,
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
    'plunder': lambda s: Plunder(s['amount'], s.get('scope', 'others')),
    'grant': lambda s: Grant(s['amount'], s.get('scope', 'others')),
    'gain_token': lambda s: GainToken(s['value']),
    'play_last_card': lambda s: PlayLastCard(),
    'free_build': lambda s: FreeBuildPerAge(s.get('count', 1)),
    'build_from_discard': lambda s: BuildFromDiscard(s.get('count', 1), s.get('priority', 0)),
    'copy_guild': lambda s: CopyGuild(s.get('count', 1)),
    # Leaders / Cities powers
    'color_sets': lambda s: ColorSets([Color(c) for c in s['colors']], s['vp_each']),
    'science_set_bonus': lambda s: ScienceSetBonus(s.get('points', 3)),
    'science_copy': lambda s: ScienceCopy(s.get('count', 1)),
    'build_income': lambda s: BuildIncome(
        s['coins'], color=s.get('color'), via_chain=s.get('via_chain', False), after_trade=s.get('after_trade', False)
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
    )


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

    def wonder(self, name: str, side: Optional[str] = None) -> Wonder:
        for w in self.wonders:
            if w.name == name and (side is None or w.side == side):
                return w
        raise KeyError((name, side))

    def card(self, name: str) -> Card:
        for c in self.cards:
            if c.name == name:
                return c
        raise KeyError(name)

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
    content = Content()
    for expansion in expansions or ('base',):
        raw = REGISTRY.expansion_raw(expansion) or read_expansion(expansion)
        define_colors(raw)
        content.wonders.extend(build_wonder(w) for w in raw.get('wonders', []))
        content.cards.extend(build_card(c, expansion) for c in raw.get('cards', []))
    loaded = set(expansions or ('base',))
    content.wonders = [w for w in content.wonders if set(w.requires) <= loaded]  # Abu Simbel needs Leaders
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
        if card.guild or card.color in (Color.WHITE, Color.BLACK) or card.age != age:
            continue  # guilds/black handled below; leaders (white) are drafted separately
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
