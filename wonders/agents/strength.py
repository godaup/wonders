"""Strength of cards and wonder boards in victory points: a rate card, paired simulations, calibration.

Three estimates of one quantity, from instant to slow:

1. **Rate card** (:func:`rate_card`, :func:`rate_wonder`) — milliseconds, for live editing. An item's
   effects are valued through the uniform :class:`~wonders.effects.base.Effect` hooks in *reference
   positions*: real tables at the end of each Age of a few simulated games (:class:`Reference`).
   Points, coins, science and on-build effects are measured as the change of the score margin when
   the item is added there; shields as the change of the remaining conflicts; production, trade
   discounts, chains and board powers by their remaining Ages. The features are weighted (priors from
   the rules' exchange rates, calibrated on simulated values of the official cards, see
   :func:`calibrate`) and summed.
2. **Paired simulation of one item** (:func:`simulate_card`, :func:`simulate_wonder`) — seconds. A card
   is put into a hand in a real game, which is played on twice from the same position: once building
   it, once discarding it (the deal and every seat's agent are identical up to that point, so their
   luck cancels). A board replaces an official one in otherwise identical games.
3. **Expansion simulation** (:func:`simulate_expansion`) — minutes. The same paired measurements
   with the whole expansion in the decks, every card where it naturally appears, then
   :func:`recalibrate` fits a correction of the rate card to the expansion's own simulated values.

All values are in **VP of final score margin** (own score minus the opponents' mean). A card's value is
relative to discarding it for coins (so a card worth building has a positive value); a board's value is
relative to an average official side (the scale of ``docs/wonder-strength.md``).

The scripted ``board`` agent plays every seat, so values describe the item as that agent uses it.
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from wonders.cards.base import MANUFACTURED_GOODS, OFFICIAL_COLORS, RAW_MATERIALS, Card, Color
from wonders.cards.content import Content, deal_leaders, load_content
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.base import Effect
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.game import Game
from wonders.engine.observation import ObservationBuilder
from wonders.engine.record import PlayerSetup, play_and_record
from wonders.ruleset import Rules, Ruleset

Item = Union[Card, WonderStage]

# -- the rate card ---------------------------------------------------------------------------------

FEATURES: Tuple[str, ...] = (
    'points',  # VP at the end of the game (incl. counting effects and guilds)
    'coins',  # coins when built, in VP
    'on_build',  # what on-build effects (plunder, debt, tokens, grants) change in the margin
    'military',  # change of the remaining conflicts' margin
    'science',  # change of the final science score
    'production',  # resource slots x Ages they are used in
    'trade',  # cheaper neighbour trade: resources covered x Ages
    'chain_out',  # later cards it makes free
    'chain_in',  # it can be built free through a chain
    'resource_cost',  # resource units it costs
    'coin_cost',  # coins it costs, in VP
    'powers',  # turn-structure and expansion powers (prior VP each)
    'turn',  # the play itself: the coins a discard would have given, in VP
)

# Prior weights: VP per feature unit from the rules' exchange rates (3 coins = 1 VP, a discard = 3 coins,
# buying a resource = 2 coins). The calibrated weights in ``data/strength.json`` replace them.
PRIOR_WEIGHTS: Dict[str, float] = {
    'points': 1.0,
    'coins': 1.0,
    'on_build': 1.0,
    'military': 1.0,
    'science': 1.0,
    'production': 0.6,
    'trade': 0.25,
    'chain_out': 1.0,
    'chain_in': 0.5,
    'resource_cost': -0.4,
    'coin_cost': -1.0,
    'powers': 1.0,
    'turn': -1.0,
}

# Prior VP of a power hook's unit (the uniform hooks of Effect), per Age it is used in when ``per_age``.
POWER_VALUES: Dict[str, Tuple[float, bool]] = {
    'plays_last_card': (1.5, True),
    'free_builds_per_age': (2.5, True),
    'discard_builds': (3.0, False),
    'guild_copies': (5.0, False),
    'science_copies': (3.0, False),
    'science_set_bonus': (1.5, False),
    'coins_per_victory': (0.5, False),
    'bank_purchases': (0.5, True),
    'leader_draws': (2.0, False),
    'extra_recruits': (2.0, False),
    'leader_copies': (2.0, False),
    'leader_entombs': (3.0, False),
}


# Resources neighbours trade, targets of resource discounts (card colours and wonder stages) and their prior
# VP per waived unit and Age (a target is built about once per Age), and hooks valued by a flat prior.
TRADEABLE = sorted(RAW_MATERIALS | MANUFACTURED_GOODS, key=lambda r: r.value)
DISCOUNT_TARGETS: Tuple[str, ...] = tuple(c.value for c in OFFICIAL_COLORS) + ('wonder',)
DISCOUNT_VALUE = 0.5
MAX_WAIVED = 3  # resource units a build typically costs: "no resources" (a large amount) waives this many
OTHER_POWERS: Tuple[str, ...] = (
    'build_income',
    'builds_free',
    'coin_discount',
    'neighbour_coin_discount',
    'redirects_defeats',
)
OTHER_POWER_VALUE = 1.0


@dataclass
class Reference:
    """Real positions to value items in: per seat, the table after each Age (``after[age]``) and at the end."""

    after: Dict[int, List[Game]]  # games as they stood after Age 1 and 2 (index: Age), and 3 (= final)
    rules: Rules
    expansions: Tuple[str, ...]

    @property
    def final(self) -> List[Game]:
        return self.after[3]


def _content(expansions: Tuple[str, ...]) -> Content:
    """Bundled content from the per-process cache; an extension's afresh (the builder edits it)."""
    from wonders.agents.bench import _content as cached
    from wonders.ruleset import BUNDLED

    if all(e in BUNDLED for e in expansions):
        return cached(expansions)
    return load_content(*expansions)


def _snapshot(game: Game, content: Optional[Content] = None) -> Game:
    """A deep copy of ``game`` that shares the (immutable) cards and boards."""
    cards = content.cards if content is not None else [c for p in game.players for c in p.cards] + game.discard_pile
    memo: Dict[int, Any] = {id(c): c for c in cards}
    memo.update({id(p.wonder): p.wonder for p in game.players})
    return copy.deepcopy(game, memo)


@lru_cache(maxsize=8)
def reference(
    expansions: Tuple[str, ...] = ('base',), players: Tuple[int, ...] = (3, 4, 5, 6, 7), games: int = 4, seed: int = 0
) -> Reference:
    """Reference positions: ``games`` games per player count, played by ``board`` agents (cached)."""
    content = _content(expansions)
    after: Dict[int, List[Game]] = {1: [], 2: [], 3: []}
    names = list(dict.fromkeys(w.name for w in content.wonders))
    rules: Optional[Rules] = None
    for n in players:
        for g in range(games):
            rng = random.Random(seed * 7_919 + n * 1_009 + g)
            setups = [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
            taken: Dict[int, Game] = {}

            def watch(game: Game, seat: int, k: int, hand: List[Card], _taken=taken) -> Optional[Action]:
                if game.age >= 2 and game.age - 1 not in _taken:  # first decision of an Age: the previous one is over
                    _taken[game.age - 1] = _snapshot(game, content)
                return None

            game = _run(content, setups, rng.random(), expansions, watch)
            rules = game.rules
            for age in (1, 2):
                after[age].append(taken[age])
            after[3].append(game)
    assert rules is not None
    return Reference(after, rules, tuple(expansions))


def _margin(values: Sequence[float], i: int) -> float:
    n = len(values)
    return values[i] - (sum(values) - values[i]) / (n - 1)


def _continuous_scores(game: Game) -> List[float]:
    """Score totals with the treasury counted exactly (coins / coins per point) instead of rounded down."""
    cpp = game.rules.coins_per_point
    out = []
    for i, p in enumerate(game.players):
        total = game.score(i).total
        out.append(total - p.coins // cpp + p.coins / cpp)
    return out


def _conflict_points(shields: Sequence[int], rules: Rules, age: int) -> List[int]:
    """Military points every seat gains in ``age``'s conflict (base rules: both neighbours, no diplomacy)."""
    n = len(shields)
    points = [0] * n
    for i in range(n):
        for j in {(i + 1) % n, (i - 1) % n} - {i}:
            if shields[i] > shields[j]:
                points[i] += rules.victory(age)
            elif shields[i] < shields[j]:
                points[i] += rules.defeat_points
    return points


def _stage_age(index: int, stages: int) -> int:
    """The Age a board's stage ``index`` (0-based) is typically built in."""
    return min(3, 1 + index * 3 // max(1, stages))


def _effect_overrides(effect: Effect, hook: str) -> bool:
    return getattr(type(effect), hook) is not getattr(Effect, hook)


def features(
    effects: Sequence[Effect],
    age: int,
    ref: Reference,
    cost: Any = None,
    chain_out: int = 0,
    chain_in: bool = False,
    turn: bool = True,
) -> Dict[str, float]:
    """The rate card's features of an item with ``effects`` built in ``age`` (see :data:`FEATURES`)."""
    cpp = ref.rules.coins_per_point
    out = dict.fromkeys(FEATURES, 0.0)
    out.update(_measured(effects, age, ref))
    for key, value in _by_ages(effects, age, ref).items():
        out[key] += value
    if cost is not None:
        out['resource_cost'] = float(sum(cost.resources.values()))
        out['coin_cost'] = cost.coins / cpp
    out['chain_out'] = float(chain_out) if age < 3 else 0.0
    out['chain_in'] = 1.0 if chain_in else 0.0
    out['turn'] = ref.rules.discard_coins / cpp if turn else 0.0
    return out


def _measured(effects: Sequence[Effect], age: int, ref: Reference) -> Dict[str, float]:
    """Points, coins, science, military and on-build effects, measured in the reference positions (seat mean)."""
    rules = ref.rules
    cpp = rules.coins_per_point
    points = coins = science = military = on_build = 0.0
    seats = 0
    symbols = [e.science() for e in effects if e.science() is not None]
    wilds = sum(e.science_wild() for e in effects)
    shields = sum(e.shields() for e in effects)
    mutating = [e for e in effects if _effect_overrides(e, 'on_build')]
    for g, final in enumerate(ref.final):
        built = ref.after[age][g]
        for i in range(final.n):
            seats += 1
            ctx = final.context(i)
            points += sum(e.victory_points(ctx) for e in effects)
            coins += sum(e.coins_on_build(built.context(i)) for e in effects) / cpp
            if symbols or wilds:
                player = final.players[i]
                counts = player.science()
                before = final._science(i, counts, player.science_wilds())
                for s in symbols:
                    counts[s] += 1
                science += final._science(i, counts, player.science_wilds() + wilds) - before
            if shields:
                for a in range(age, 4):
                    table = [p.shields() for p in ref.after[a][g].players]
                    base = _conflict_points(table, rules, a)
                    table[i] += shields
                    military += _margin(_conflict_points(table, rules, a), i) - _margin(base, i)
            if mutating and g < 6:  # a few copies suffice; deep copies are the slow part
                game = _snapshot(built)
                scores = _continuous_scores(game)
                for e in mutating:
                    e.on_build(game.context(i))
                game._flush_pending()
                on_build += (_margin(_continuous_scores(game), i) - _margin(scores, i)) * len(ref.final) / 6
    return {
        'points': points / seats,
        'coins': coins / seats,
        'science': science / seats,
        'military': military / seats,
        'on_build': on_build / seats if mutating else 0.0,
    }


def _by_ages(effects: Sequence[Effect], age: int, ref: Reference) -> Dict[str, float]:
    """Production, trade discounts and powers, valued by the Ages they are used in (the current one included)."""
    remaining = 4 - age
    out = {'production': 0.0, 'trade': 0.0, 'powers': 0.0}
    for e in effects:
        for slot in e.production():
            out['production'] += remaining * (1.0 + 0.25 * (len(slot) - 1))
        for rule in e.trade_rules():
            covered = (2 if rule.side == 'both' else 1) * (2 if rule.category == 'both' else 1)
            out['trade'] += remaining * covered
        for hook, (value, per_age) in POWER_VALUES.items():
            units = getattr(e, hook)()
            if units:
                out['powers'] += float(units) * value * (remaining if per_age else 1)
        # Own-use production (Leaders / Cities): what it adds without, and on top of, the usual goods.
        if _effect_overrides(e, 'private_production'):
            slots = len(e.private_production(frozenset())) + len(e.private_production(frozenset(TRADEABLE)))
            out['production'] += remaining * slots / 2
        if _effect_overrides(e, 'trade_reduction'):  # coins saved per unit bought, per side and resource
            saved = [
                e.trade_reduction(p, side, r)
                for game in ref.after[age]
                for p in game.players
                for side in ('left', 'right')
                for r in TRADEABLE
            ]
            out['trade'] += remaining * sum(saved) / max(1, len(saved)) * 2 * len(TRADEABLE) / 2
        if _effect_overrides(e, 'resource_discount'):  # resource units waived per build of a target
            waived = sum(min(e.resource_discount(t), MAX_WAIVED) for t in DISCOUNT_TARGETS)
            out['powers'] += remaining * DISCOUNT_VALUE * waived
        for hook in OTHER_POWERS:  # powers without a measure here: a flat prior per Age
            if _effect_overrides(e, hook):
                out['powers'] += remaining * OTHER_POWER_VALUE
    return out


@dataclass
class Rating:
    """A rate-card estimate: total VP and each feature's contribution."""

    value: float
    parts: Dict[str, float]
    corrected: Optional[float] = None  # after an expansion's recalibration (:func:`recalibrate`)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'value': round(self.value, 2),
            'parts': {k: round(v, 2) for k, v in self.parts.items() if abs(v) >= 0.005},
        }
        if self.corrected is not None:
            out['corrected'] = round(self.corrected, 2)
        return out


def _weigh(feats: Dict[str, float], weights: Dict[str, float], intercept: float = 0.0) -> Rating:
    parts = {k: weights.get(k, 0.0) * v for k, v in feats.items()}
    return Rating(sum(parts.values()) + intercept, parts)


def rate_card(card: Card, ref: Optional[Reference] = None, calibration: Optional['Calibration'] = None) -> Rating:
    """The rate card's value of ``card`` (VP of margin, relative to discarding it)."""
    ref = ref or reference()
    cal = calibration or default_calibration()
    age = card.age or 1
    feats = features(
        card.effects,
        age,
        ref,
        card.cost,
        chain_out=len(card.chain_enables),
        chain_in=bool(card.chain_requires),
    )
    return _weigh(feats, cal.weights, cal.intercept)


def rate_stages(
    wonder: Wonder, ref: Optional[Reference] = None, calibration: Optional['Calibration'] = None
) -> List[Rating]:
    """The rate card's value of each stage of ``wonder`` (a copied stage: the mean official stage there)."""
    ref = ref or reference()
    cal = calibration or default_calibration()
    out = []
    for k, stage in enumerate(wonder.stages):
        if stage.copy is not None:
            out.append(Rating(cal.mean_stage(k), {'copy': cal.mean_stage(k)}))
            continue
        feats = features(stage.effects, _stage_age(k, len(wonder.stages)), ref, stage.cost)
        out.append(_weigh(feats, cal.weights, cal.intercept))
    return out


def board_raw(
    wonder: Wonder, ref: Optional[Reference] = None, calibration: Optional['Calibration'] = None
) -> Dict[str, float]:
    """The parts of a board's raw rate: its stages, starting resource, extra coins and board powers."""
    ref = ref or reference()
    cal = calibration or default_calibration()
    stages = rate_stages(wonder, ref, cal)
    # Later stages are built less often: weight each by the share of official boards' stages built there.
    built = [cal.stage_built(k, len(wonder.stages)) for k in range(len(wonder.stages))]
    passive = _weigh(features(wonder.effects, 1, ref, None, turn=False), cal.weights).value if wonder.effects else 0.0
    return {
        'stages': sum(r.value * b for r, b in zip(stages, built)),
        'resource': (3.0 * cal.weights['production']) if wonder.starting_resource is not None else 0.0,
        'coins': wonder.extra_coins / ref.rules.coins_per_point,
        'powers': passive,
    }


def rate_wonder(wonder: Wonder, ref: Optional[Reference] = None, calibration: Optional['Calibration'] = None) -> Rating:
    """The rate card's value of a board side (VP of margin vs an average official side)."""
    cal = calibration or default_calibration()
    parts = board_raw(wonder, ref, cal)
    raw = sum(parts.values())
    value = cal.board_slope * (raw - cal.board_mean)
    return Rating(
        value, {k: cal.board_slope * v for k, v in parts.items()} | {'average': -cal.board_slope * cal.board_mean}
    )


# -- calibration ------------------------------------------------------------------------------------


@dataclass
class Calibration:
    """Weights of the rate card and the official items' simulated values (``data/strength.json``)."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(PRIOR_WEIGHTS))
    intercept: float = 0.0
    cards: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # official cards: value, se, age, color, rate
    boards: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # official sides: value, se, raw
    board_slope: float = 1.0
    board_mean: float = 0.0
    stages: List[float] = field(default_factory=lambda: [2.0, 3.0, 4.0, 4.0])  # mean official stage value
    built: List[List[float]] = field(default_factory=list)  # share of stages built, per board length
    error: Dict[str, float] = field(default_factory=dict)  # leave-one-out RMSE of the rate card

    def mean_stage(self, k: int) -> float:
        return self.stages[min(k, len(self.stages) - 1)] if self.stages else 0.0

    def stage_built(self, k: int, stages: int) -> float:
        for row in self.built:
            if len(row) == stages:
                return row[k]
        return max(0.3, 1.0 - 0.2 * k)

    def official(self, age: Optional[int], color: Optional[str]) -> List[float]:
        """Simulated values of the official cards of an Age and colour (for comparison)."""
        return sorted(
            v['value']
            for v in self.cards.values()
            if v.get('age') == age and (color is None or v.get('color') == color)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'intercept': self.intercept,
            'cards': self.cards,
            'boards': self.boards,
            'board_slope': self.board_slope,
            'board_mean': self.board_mean,
            'stages': self.stages,
            'built': self.built,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Calibration':
        cal = cls()
        for key, value in data.items():
            if hasattr(cal, key):
                setattr(cal, key, value)
        cal.weights = {**PRIOR_WEIGHTS, **cal.weights}
        return cal


@lru_cache(maxsize=1)
def default_calibration() -> Calibration:
    """The bundled calibration (``wonders/agents/data/strength.json``), or the priors without it."""
    try:
        text = resources.files('wonders.agents').joinpath('data/strength.json').read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError):
        return Calibration()
    return Calibration.from_dict(json.loads(text))


def _solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve ``A x = b`` (small, symmetric positive definite) by Gaussian elimination with pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def fit_weights(
    rows: Sequence[Dict[str, float]],
    targets: Sequence[float],
    errors: Sequence[float],
    prior: Dict[str, float] = PRIOR_WEIGHTS,
    strength: float = 4.0,
    tau: float = 1.0,
) -> Tuple[Dict[str, float], float]:
    """Weighted ridge regression of simulated values on the features, shrunk towards ``prior``.

    Each item weighs ``1 / (se^2 + tau^2)`` (``tau``: what the features cannot explain); ``strength`` is
    the prior's weight in item equivalents per feature. Returns the weights and an intercept (not shrunk).
    """
    names = list(FEATURES)
    p = len(names) + 1
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for feats, y, se in zip(rows, targets, errors):
        x = [feats[k] for k in names] + [1.0]
        w = 1.0 / (se * se + tau * tau)
        for r in range(p):
            b[r] += w * x[r] * y
            for c in range(p):
                A[r][c] += w * x[r] * x[c]
    for r, k in enumerate(names):  # pull towards the prior, scaled by the feature's spread
        # Scaled by the feature's spread, but never below one unit: a feature that (almost) no item has
        # must stay near its prior instead of fitting the one item that has it.
        scale = statistics.pvariance([f[k] for f in rows]) if len(rows) > 1 else 0.0
        lam = strength / (tau * tau) * max(scale, 1.0)
        A[r][r] += lam
        b[r] += lam * prior[k]
    beta = _solve(A, b)
    return dict(zip(names, beta[:-1])), beta[-1]


def recalibrate(rates: Sequence[float], values: Sequence[float], errors: Sequence[float]) -> Tuple[float, float]:
    """An expansion's correction ``value ≈ offset + slope × rate``, shrunk towards ``(0, 1)``.

    Few items move it a little, many items with tight estimates a lot. Returns ``(offset, slope)``.
    """
    lam_offset, lam_slope = 4.0, 20.0  # prior precision in item equivalents
    tau2 = 1.0
    s = [[lam_offset, 0.0], [0.0, lam_slope]]
    t = [0.0, lam_slope * 1.0]
    for r, y, se in zip(rates, values, errors):
        w = 1.0 / (se * se + tau2)
        s[0][0] += w
        s[0][1] += w * r
        s[1][0] += w * r
        s[1][1] += w * r * r
        t[0] += w * y
        t[1] += w * r * y
    offset, slope = _solve(s, t)
    return offset, slope


# -- paired simulations -----------------------------------------------------------------------------

Hook = Callable[[Game, int, int, List[Card]], Optional[Action]]


def _run(
    content: Content,
    setups: Sequence[PlayerSetup],
    seed: float,
    expansions: Tuple[str, ...],
    hook: Hook,
    agent: str = 'board',
) -> Game:
    """Play one game with ``agent`` in every seat; ``hook(game, seat, k, hand)`` may force seat's ``k``-th play.

    Everything random derives from ``seed``, so two runs that force different moves at the same point
    are identical up to it.
    """
    from wonders.agents.bench import AGENTS

    rng = random.Random(seed)
    agents = [AGENTS[agent].build(random.Random(rng.random())) for _ in setups]
    observations = ObservationBuilder()
    count = [0] * len(setups)

    def policy(game: Game, seat: int, hand: List[Card]) -> Action:
        if game.decision is DecisionKind.PLAY:
            k = count[seat]
            count[seat] += 1
            forced = hook(game, seat, k, hand)
            if forced is not None:
                return forced
        return agents[seat].act(observations.build(game, seat, hand), hand)

    def on_turn_extra(age: int, turn: int, actions: Sequence[Optional[Action]]) -> None:
        observations.record(age, turn, [None if a is None else (a.kind, a.card.name) for a in actions])

    leaders = Ruleset.of(*expansions).uses('leaders')
    draft, box = deal_leaders(content, len(setups), rng) if leaders else (None, [])
    game, _ = play_and_record(
        content, setups, policy, rng, expansions, on_turn_extra, leader_draft=draft, leader_box=box
    )
    return game


def _margins(game: Game) -> List[float]:
    totals = [game.score(i).total for i in range(game.n)]
    return [_margin(totals, i) for i in range(game.n)]


def _legal(game: Game, seat: int, hand: List[Card], action: Action) -> bool:
    return action in game.legal_actions(seat, hand)


def _play_for(card: Card) -> Callable[[Game, int, List[Card]], Optional[Action]]:
    """The build of ``card`` if legal (a free build through Olympia counts as a build)."""

    def choose(game: Game, seat: int, hand: List[Card]) -> Optional[Action]:
        for action in (Action.build(card), Action.free(card)):
            if _legal(game, seat, hand, action):
                return action
        return None

    return choose


@dataclass
class Estimate:
    """A simulated value: mean VP of margin, its standard error and the number of paired samples."""

    value: float
    se: float
    samples: int
    unaffordable: int = 0  # (cards) positions where it could not be built, skipped

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready (an unknown standard error, from fewer than two samples, is ``None``)."""
        se = round(self.se, 2) if math.isfinite(self.se) else None
        return {'value': round(self.value, 2), 'se': se, 'samples': self.samples}


def _estimate(diffs: Sequence[float], skipped: int = 0) -> Estimate:
    n = len(diffs)
    if n == 0:
        return Estimate(0.0, float('inf'), 0, skipped)
    mean = sum(diffs) / n
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else float('inf')
    return Estimate(mean, se, n, skipped)


def _deal(content: Content, n: int, rng: random.Random) -> List[PlayerSetup]:
    names = list(dict.fromkeys(w.name for w in content.wonders))
    return [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]


def card_pair(
    content: Content, card: Card, n: int, seed: int, expansions: Tuple[str, ...] = ('base',)
) -> Optional[float]:
    """One paired sample: ``card`` put into a random seat's hand at a random turn of its Age, built vs discarded.

    The card replaces a random card of that hand (in both runs). ``None`` when it cannot be built there.
    """
    rng = random.Random(seed)
    setups = _deal(content, n, rng)
    seat = rng.randrange(n)
    age = card.age or 1
    turn = rng.randrange(5)  # early to late in the Age (not the last play, a choice between two cards)
    game_seed = rng.random()
    swap = rng.random()
    can_build = _play_for(card)
    margins: Dict[str, float] = {}
    for mode in ('build', 'discard'):

        def hook(game: Game, s: int, k: int, hand: List[Card], _mode=mode) -> Optional[Action]:
            if s != seat or game.age != age or game.turn != turn:
                return None
            hand[int(swap * len(hand))] = card  # the hand is the engine's own list
            action = can_build(game, s, hand) if _mode == 'build' else Action.discard(card)
            if action is None:
                raise _Unaffordable
            return action

        try:
            game = _run(content, setups, game_seed, expansions, hook)
        except _Unaffordable:
            return None
        margins[mode] = _margins(game)[seat]
    return margins['build'] - margins['discard']


class _Unaffordable(Exception):
    pass


def simulate_card(
    card: Card,
    samples: int = 40,
    seed: int = 0,
    expansions: Tuple[str, ...] = ('base',),
    players: Sequence[int] = (3, 4, 5, 6, 7),
    content: Optional[Content] = None,
) -> Estimate:
    """Paired simulation of one card in base-game decks (see :func:`card_pair`)."""
    content = content or _content(expansions)
    diffs: List[float] = []
    skipped = 0
    attempt = 0
    while len(diffs) < samples and attempt < samples * 4:
        rng = random.Random(seed * 1_000_003 + attempt)
        d = card_pair(content, card, rng.choice(list(players)), rng.randrange(2**31), expansions)
        attempt += 1
        if d is None:
            skipped += 1
        else:
            diffs.append(d)
    return _estimate(diffs, skipped)


def wonder_pair(
    content: Content,
    wonder: Wonder,
    n: int,
    seed: int,
    expansions: Tuple[str, ...] = ('base',),
    calibration: Optional[Calibration] = None,
) -> Optional[float]:
    """One paired sample of a board: seat 0 plays ``wonder``, then an official side, in the same deal.

    Returns the margin difference plus that official side's measured strength, so the result is on the
    scale of an average official side.
    """
    cal = calibration or default_calibration()
    rng = random.Random(seed)
    # Official sides with a measured strength; without a calibration every other side, counted as average.
    in_play = [f'{w.name} {w.side}' for w in content.wonders if w.name != wonder.name]
    official = [k for k in in_play if k in cal.boards] or in_play
    other = rng.choice(official)
    other_name, other_side = other.rsplit(' ', 1)
    others = [w for w in dict.fromkeys(w.name for w in content.wonders) if w not in (wonder.name, other_name)]
    if len(others) < n - 1:
        return None
    setups = [PlayerSetup('p0', wonder.name, wonder.side)]  # seat 0 is set below
    setups += [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(others, n - 1), 1)]
    game_seed = rng.random()
    margins = []
    for name, side in ((wonder.name, wonder.side), (other_name, other_side)):
        setups[0] = PlayerSetup('p0', name, side)
        game = _run(content, setups, game_seed, expansions, lambda *a: None)
        margins.append(_margins(game)[0])
    return margins[0] - margins[1] + float(cal.boards.get(other, {}).get('value', 0.0))


def simulate_wonder(
    wonder: Wonder,
    samples: int = 30,
    seed: int = 0,
    expansions: Tuple[str, ...] = ('base',),
    players: Sequence[int] = (3, 4, 5, 6, 7),
    content: Optional[Content] = None,
    calibration: Optional[Calibration] = None,
) -> Estimate:
    """Paired simulation of one board side (see :func:`wonder_pair`)."""
    base = content or _content(expansions)
    content = Content(list(base.cards), [w for w in base.wonders if w.name != wonder.name] + [wonder])
    diffs = []
    for s in range(samples):
        rng = random.Random(seed * 1_000_003 + s)
        d = wonder_pair(content, wonder, rng.choice(list(players)), rng.randrange(2**31), expansions, calibration)
        if d is not None:
            diffs.append(d)
    return _estimate(diffs)


def natural_pairs(
    content: Content,
    names: Sequence[str],
    n: int,
    seed: int,
    expansions: Tuple[str, ...],
) -> Dict[str, float]:
    """Paired samples of cards where they naturally come up: one game is watched, then branched.

    For every card in ``names`` that some seat could build in the watched game, one such moment is
    picked at random and played on twice (build vs discard). Returns ``{card: margin difference}``.
    """
    rng = random.Random(seed)
    setups = _deal(content, n, rng)
    game_seed = rng.random()
    wanted = set(names)
    seen: Dict[str, List[Tuple[int, int]]] = {}

    def watch(game: Game, seat: int, k: int, hand: List[Card]) -> Optional[Action]:
        for card in hand:
            if card.name in wanted and _play_for(card)(game, seat, hand) is not None:
                seen.setdefault(card.name, []).append((seat, k))
        return None

    _run(content, setups, game_seed, expansions, watch)
    out: Dict[str, float] = {}
    for name in sorted(seen):
        seat, target = rng.choice(seen[name])
        margins = {}
        for mode in ('build', 'discard'):

            def hook(game: Game, s: int, k: int, hand: List[Card], _mode=mode, _name=name) -> Optional[Action]:
                if s != seat or k != target:
                    return None
                card = next(c for c in hand if c.name == _name)
                return _play_for(card)(game, s, hand) if _mode == 'build' else Action.discard(card)

            margins[mode] = _margins(_run(content, setups, game_seed, expansions, hook))[seat]
        out[name] = margins['build'] - margins['discard']
    return out


def _natural_job(job: Tuple[Tuple[str, ...], Tuple[str, ...], int, int]) -> Dict[str, float]:
    expansions, names, n, seed = job
    return natural_pairs(_content(expansions), names, n, seed, expansions)


def _wonder_job(job: Tuple[Tuple[str, ...], str, str, int, int]) -> Optional[float]:
    expansions, name, side, n, seed = job
    content = _content(expansions)
    return wonder_pair(content, content.wonder(name, side), n, seed, expansions)


def simulate_expansion(
    expansion: str,
    games: int = 200,
    seed: int = 0,
    players: Sequence[int] = (3, 4, 5, 6, 7),
    processes: int = 1,
    progress: Optional[Callable[[float], None]] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """Every new card and board of ``expansion`` simulated with the whole expansion in play.

    Cards: :func:`natural_pairs` over ``games`` watched games; boards: :func:`wonder_pair` with the
    expansion's decks, ``games // 2`` per side. Returns ``{'cards': {name: Estimate}, 'wonders':
    {'name side': Estimate}}``. ``context`` picks the worker start method (default:
    :func:`wonders._mp.get_context`); the GUI's server passes ``'spawn'``, as forking a process that
    runs threads is unsafe.
    """
    from wonders._mp import get_context, init_worker

    expansions = ('base', expansion)
    own = load_content(expansion)
    names = tuple(sorted({c.name for c in own.cards if c.color is not Color.WHITE}))
    sides = [(w.name, w.side) for w in own.wonders]
    rng = random.Random(seed)
    card_jobs = (
        [(expansions, names, rng.choice(list(players)), rng.randrange(2**31)) for _ in range(games)] if names else []
    )
    wonder_jobs = [
        (expansions, name, side, rng.choice(list(players)), rng.randrange(2**31))
        for name, side in sides
        for _ in range(max(1, games // 2))
    ]
    total = len(card_jobs) + len(wonder_jobs)
    done = 0
    card_results: List[Dict[str, float]] = []
    wonder_results: List[Optional[float]] = []

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done / max(1, total))

    if processes > 1 and total > 1:
        import multiprocessing

        ctx = multiprocessing.get_context(context) if context else get_context()
        with ctx.Pool(processes, initializer=init_worker) as pool:
            for result in pool.imap(_natural_job, card_jobs):
                card_results.append(result)
                tick()
            for w in pool.imap(_wonder_job, wonder_jobs):
                wonder_results.append(w)
                tick()
    else:
        for job in card_jobs:
            card_results.append(_natural_job(job))
            tick()
        for wjob in wonder_jobs:
            wonder_results.append(_wonder_job(wjob))
            tick()
    diffs: Dict[str, List[float]] = {name: [] for name in names}
    for result in card_results:
        for name, d in result.items():
            diffs[name].append(d)
    per_side: Dict[str, List[float]] = {f'{n} {s}': [] for n, s in sides}
    for wjob, w in zip(wonder_jobs, wonder_results):
        if w is not None:
            per_side[f'{wjob[1]} {wjob[2]}'].append(w)
    return {
        'cards': {name: _estimate(d) for name, d in diffs.items()},
        'wonders': {key: _estimate(d) for key, d in per_side.items()},
    }


# -- calibration on the official content (regeneration script) ---------------------------------------

CALIBRATION_EXPANSIONS: Tuple[Tuple[str, ...], ...] = (('base',), ('base', 'cities'))


def refit(cal: Calibration) -> Calibration:  # pragma: no cover - script
    """Fit the weights again to the official cards' stored simulated values (no new games)."""
    ref = reference()
    content = _content(('base', 'cities'))
    items = [content.card(name) for name in sorted(cal.cards)]
    _fit(cal, ref, items, [cal.cards[c.name]['value'] for c in items], [cal.cards[c.name]['se'] for c in items])
    _calibrate_boards(cal, ref)
    return cal


def _fit(cal: Calibration, ref: Reference, items: List[Card], targets: List[float], errors: List[float]) -> None:
    """Fit the rate card's weights and record each card's rate and the leave-one-out error."""
    rows = [features(c.effects, c.age or 1, ref, c.cost, len(c.chain_enables), bool(c.chain_requires)) for c in items]
    cal.weights, cal.intercept = fit_weights(rows, targets, errors)
    sq = 0.0
    for k in range(len(rows)):
        w, b = fit_weights(rows[:k] + rows[k + 1 :], targets[:k] + targets[k + 1 :], errors[:k] + errors[k + 1 :])
        pred = sum(w[f] * rows[k][f] for f in FEATURES) + b
        sq += (pred - targets[k]) ** 2
    cal.error['cards_loo_rmse'] = round(math.sqrt(sq / max(1, len(rows))), 3)
    cal.error['cards_sd'] = round(statistics.pstdev(targets), 3)
    for card, feats in zip(items, rows):
        cal.cards[card.name]['rate'] = round(_weigh(feats, cal.weights, cal.intercept).value, 3)


def calibrate(games: int = 3000, seed: int = 0, processes: int = 1) -> Calibration:  # pragma: no cover - script
    """Simulate the official cards (natural pairs) and fit the rate card's weights to them.

    Boards use the measured strengths of ``docs/wonder-strength.json`` (``main``).
    """
    from wonders._mp import get_context, init_worker

    ref = reference()
    jobs = []
    rng = random.Random(seed)
    for expansions in CALIBRATION_EXPANSIONS:
        content = _content(expansions)
        names = tuple(sorted({c.name for c in content.cards if c.color is not Color.WHITE}))
        jobs += [(expansions, names, rng.choice([3, 4, 5, 6, 7]), rng.randrange(2**31)) for _ in range(games)]
    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            results = pool.map(_natural_job, jobs, chunksize=4)
    else:
        results = [_natural_job(job) for job in jobs]
    diffs: Dict[str, List[float]] = {}
    for result in results:
        for name, d in result.items():
            diffs.setdefault(name, []).append(d)
    content = _content(('base', 'cities'))
    cal = Calibration()
    targets, errors, items = [], [], []
    for name, samples in sorted(diffs.items()):
        est = _estimate(samples)
        if est.samples < 10:
            continue
        card = content.card(name)
        targets.append(est.value)
        errors.append(est.se)
        items.append(card)
        cal.cards[name] = {
            'value': round(est.value, 3),
            'se': round(est.se, 3),
            'samples': est.samples,
            'age': card.age,
            'color': card.color.value,
        }
    _fit(cal, ref, items, targets, errors)
    _calibrate_boards(cal, ref)
    return cal


def _calibrate_boards(cal: Calibration, ref: Reference) -> None:  # pragma: no cover - script
    """Fit the board scale to the measured side strengths (``docs/wonder-strength.json``)."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / 'docs' / 'wonder-strength.json'
    measured = json.loads(path.read_text(encoding='utf-8'))['main']
    content = load_content('base', 'leaders', 'cities', 'wonderpack')
    # Mean official stage value by position (for copied stages).
    by_pos: Dict[int, List[float]] = {}
    for w in content.wonders:
        for k, r in enumerate(rate_stages(w, ref, cal)):
            if w.stages[k].copy is None:
                by_pos.setdefault(k, []).append(r.value)
    cal.stages = [round(statistics.mean(by_pos[k]), 3) for k in sorted(by_pos)]
    cal.built = [[1.0, 0.9], [1.0, 0.85, 0.7], [1.0, 0.85, 0.7, 0.55]]
    raws, values = [], []
    for w in content.wonders:
        key = f'{w.name} {w.side}'
        if key not in measured:
            continue
        raw = sum(board_raw(w, ref, cal).values())
        raws.append(raw)
        values.append(measured[key]['value'])
        cal.boards[key] = {'value': measured[key]['value'], 'se': measured[key]['se'], 'raw': round(raw, 3)}
    mean_raw = statistics.mean(raws)
    mean_val = statistics.mean(values)
    sxx = sum((r - mean_raw) ** 2 for r in raws)
    sxy = sum((r - mean_raw) * (v - mean_val) for r, v in zip(raws, values))
    cal.board_slope = round(sxy / sxx, 4) if sxx else 1.0
    cal.board_mean = round(mean_raw - mean_val / cal.board_slope if cal.board_slope else mean_raw, 4)
    pred = [cal.board_slope * (r - cal.board_mean) for r in raws]
    cal.error['boards_rmse'] = round(math.sqrt(statistics.mean((p - v) ** 2 for p, v in zip(pred, values))), 3)
    cal.error['boards_r'] = round(statistics.correlation(raws, values), 3)


__all__ = [
    'Calibration',
    'Estimate',
    'FEATURES',
    'PRIOR_WEIGHTS',
    'Rating',
    'Reference',
    'calibrate',
    'default_calibration',
    'features',
    'fit_weights',
    'rate_card',
    'rate_stages',
    'rate_wonder',
    'recalibrate',
    'reference',
    'simulate_card',
    'simulate_expansion',
    'simulate_wonder',
]


if __name__ == '__main__':  # pragma: no cover - regeneration script
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Calibrate the rate card on simulated values of the official cards.')
    parser.add_argument('-g', '--games', type=int, default=3000, help='watched games per expansion set')
    parser.add_argument('-p', '--processes', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--refit', action='store_true', help='refit the weights to the stored simulations only')
    args = parser.parse_args()
    out = Path(__file__).with_name('data') / 'strength.json'
    if args.refit:
        result = refit(Calibration.from_dict(json.loads(out.read_text(encoding='utf-8'))))
    else:
        result = calibrate(args.games, args.seed, args.processes)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result.error), f'-> {out}')
