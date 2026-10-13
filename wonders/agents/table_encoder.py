"""Encoder v2: everything a seat may legally observe, as one fixed-length vector.

:class:`~wonders.agents.encoders.CardEncoder` (v1) encodes the viewer and its two
neighbours with coarse features. Benchmarks showed network capacity was not the limit
(a 3x larger MLP did no better, and imitation plateaued at ~74% agreement with
deterministic teachers), pointing at missing information. :class:`TableEncoder` covers
the whole table and every observable detail; see ``docs/encoder-v2.md`` for the design.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from wonders.agents.encoders import Encoder
from wonders.engine.observation import CityView, Observation
from wonders.ruleset import has_family

MAX_SEATS = 7
RESOURCES = ('bricks', 'glass', 'linen', 'ore', 'paper', 'stone', 'wood')
SYMBOLS = ('tablet', 'gear', 'compass')
ALL_COLORS = ('brown', 'grey', 'blue', 'yellow', 'red', 'green', 'purple', 'white', 'black')
SCORE_KEYS = ('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science', 'leaders', 'cities')
MOVE_KINDS = ('build', 'wonder', 'discard', 'free')
MAX_STAGES = 4
PLAYER_COUNTS = (3, 4, 5, 6, 7)
POWERS = (
    'plays_last_card',
    'free_builds_per_age',
    'discard_builds',
    'guild_copies',
    'leader_draws',
    'extra_recruits',
    'leader_copies',
    'science_copies',
)
# vp, shields, coins, science symbol, science wild, production slots, powers..., diplomacy
EFFECT_FEATURES = 6 + len(POWERS) + 1
STAGE_FEATURES = len(RESOURCES) + 1 + EFFECT_FEATURES + 1  # cost resources, cost coins, effects, built


def _num(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else 0.0


def effect_features(effects: Sequence[Dict[str, object]]) -> List[float]:
    """A fixed vector summarizing a list of effect summaries (e.g. one wonder stage)."""
    vec = [0.0] * EFFECT_FEATURES
    for e in effects:
        vec[0] += _num(e.get('victory_points')) / 5
        vec[1] += _num(e.get('shields')) / 3
        vec[2] += _num(e.get('coins')) / 5
        vec[3] += 1.0 if e.get('science') else 0.0
        vec[4] += _num(e.get('science_wild'))
        production = e.get('production')
        vec[5] += float(len(production)) if isinstance(production, list) else 0.0
        for k, power in enumerate(POWERS):
            vec[6 + k] += _num(e.get(power))
        vec[6 + len(POWERS)] += 1.0 if e.get('type') == 'Diplomacy' else 0.0
    return vec


def _cost_into(vec: List[float], offset: int, cost: object) -> None:
    """Write a cost dict (``{'coins', 'resources'}``) as resource units / 4 and coins / 10."""
    if not isinstance(cost, dict):
        return
    resources = cost.get('resources', {})
    if isinstance(resources, dict):
        for res, amount in resources.items():
            if res in RESOURCES:
                vec[offset + RESOURCES.index(res)] = _num(amount) / 4
    vec[offset + len(RESOURCES)] = _num(cost.get('coins')) / 10


class TableEncoder(Encoder):
    """Everything a seat may legally observe, seat-relative and padded to 7 seats.

    Slot 0 is the viewer and slot ``k`` the city ``k`` seats clockwise (slot 1 = left
    neighbour, slot ``n-1`` = right neighbour); unused slots stay zero, so the size does
    not depend on the player count.

    Sections (:meth:`sections`; a structured model can embed the seven ``city`` blocks
    with shared weights):

    * ``city0`` … ``city6`` — present / left / right flags; coins, shields, stages,
      score and its nine-category breakdown; military tokens by value and defeats;
      debt, diplomacy; science symbols and wilds; card counts per colour; production
      (fixed units and choice slots per resource, own-use slots); wonder identity and
      side; next stage's cost; Courtesans copies; built cards (multi-hot).
    * ``hand`` — per card name: copies held, buildable, coin cost, trade part, chain-free.
    * ``private`` — own leader hand, cards seen earlier this Age that moved on (card
      counting), cards the viewer discarded / buried (it knows them), trade prices per
      neighbour and resource, free builds left, stage affordable, and the viewer's
      whole wonder board (per stage: cost, effects, built).
    * ``context`` — Age, turn, hand size, discard pile size, player count, expansions,
      pass direction, decision kind.
    * ``history`` — visible moves per seat slot and kind.

    Hidden information never enters: opponents' hands and leader hands, and the
    identity of cards others buried or discarded.
    """

    version = 'table-v2'

    def __init__(self, vocab: Sequence[str], decisions: Sequence[str], wonders: Sequence[str]):
        self.vocab = list(vocab)
        self.decisions = tuple(decisions)
        self.wonders = list(wonders)
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.vocab)}
        self._wonder_index: Dict[str, int] = {name: i for i, name in enumerate(self.wonders)}
        v, r = len(self.vocab), len(RESOURCES)
        self.city_size = (
            3  # present, left neighbour, right neighbour
            + 5  # coins, shields, stages built, stages total, score
            + len(SCORE_KEYS)
            + 4  # victory tokens 1 / 3 / 5, defeats
            + 2  # debt, diplomacy
            + len(SYMBOLS)
            + 1  # science wilds
            + len(ALL_COLORS)
            + 2 * r
            + 1  # production: fixed units, choice slots, own-use slots
            + len(self.wonders)
            + 1  # wonder identity, side B
            + r
            + 2  # next stage cost (resources, coins), has a next stage
            + 1  # Courtesans copies
            + v  # built cards
        )
        self._sections: List[Tuple[str, int]] = [(f'city{k}', self.city_size) for k in range(MAX_SEATS)]
        self._sections += [
            ('hand', 5 * v),
            ('private', 3 * v + 2 * r + 2 + MAX_STAGES * STAGE_FEATURES),
            ('context', 3 + 3 + len(PLAYER_COUNTS) + 3 + len(self.decisions)),
            ('history', MAX_SEATS * len(MOVE_KINDS)),
        ]
        self._offsets = {name: offset for name, offset, _n in self.sections()}

    @property
    def size(self) -> int:
        return sum(n for _name, n in self._sections)

    def sections(self) -> List[Tuple[str, int, int]]:
        """``(name, offset, length)`` of every section, in order."""
        out, offset = [], 0
        for name, n in self._sections:
            out.append((name, offset, n))
            offset += n
        return out

    # -- encode ------------------------------------------------------------------------
    def encode(self, obs: Observation) -> List[float]:
        vec = [0.0] * self.size
        n = obs.n_players
        for slot in range(min(n, MAX_SEATS)):
            self._city(vec, self._offsets[f'city{slot}'], obs.table[(obs.seat + slot) % n], slot, n)
        self._hand(vec, self._offsets['hand'], obs)
        self._private(vec, self._offsets['private'], obs)
        self._context(vec, self._offsets['context'], obs)
        o = self._offsets['history']
        for e in obs.history:
            slot = (e.seat - obs.seat) % n
            if e.kind in MOVE_KINDS:
                vec[o + slot * len(MOVE_KINDS) + MOVE_KINDS.index(e.kind)] += 1.0 / 18
        return vec

    def _city(self, vec: List[float], o: int, city: CityView, slot: int, n: int) -> None:
        r = len(RESOURCES)
        vec[o : o + 3] = [1.0, float(slot == 1), float(slot == n - 1)]
        o += 3
        vec[o : o + 5] = [
            city.coins / 10,
            city.shields / 5,
            city.stages_built / 4,
            city.stages_total / 4,
            city.score / 50,
        ]
        o += 5
        for k, key in enumerate(SCORE_KEYS):
            vec[o + k] = city.score_breakdown.get(key, 0) / 20
        o += len(SCORE_KEYS)
        t = city.tokens
        vec[o : o + 4] = [t.count(1) / 3, t.count(3) / 3, t.count(5) / 3, sum(1 for x in t if x < 0) / 6]
        o += 4
        vec[o : o + 2] = [city.debt / 5, float(city.diplomacy)]
        o += 2
        for k, symbol in enumerate(SYMBOLS):
            vec[o + k] = city.science.get(symbol, 0) / 4
        vec[o + len(SYMBOLS)] = float(city.science_wilds)
        o += len(SYMBOLS) + 1
        for k, color in enumerate(ALL_COLORS):
            vec[o + k] = city.cards_by_color.get(color, 0) / 5
        o += len(ALL_COLORS)
        for options in city.production:
            if len(options) == 1 and options[0] in RESOURCES:
                vec[o + RESOURCES.index(options[0])] += 1.0
            else:
                for res in options:
                    if res in RESOURCES:
                        vec[o + r + RESOURCES.index(res)] += 1.0
        vec[o + 2 * r] = float(city.private_slots)
        o += 2 * r + 1
        if city.wonder in self._wonder_index:
            vec[o + self._wonder_index[city.wonder]] = 1.0
        vec[o + len(self.wonders)] = float(city.wonder_side == 'B')
        o += len(self.wonders) + 1
        if city.next_stage_cost is not None:
            _cost_into(vec, o, city.next_stage_cost)
            vec[o + r + 1] = 1.0
        o += r + 2
        vec[o] = float(len(city.copied_leaders))
        o += 1
        for name in city.cards:
            vec[o + self._index[name]] = 1.0

    def _hand(self, vec: List[float], o: int, obs: Observation) -> None:
        v = len(self.vocab)
        for card in obs.hand:
            i = self._index[card.name]
            vec[o + i] += 1.0
            if card.can_build:
                vec[o + v + i] = 1.0
                vec[o + 2 * v + i] = (card.build_coin_cost or 0) / 10
                vec[o + 3 * v + i] = card.trade_coins / 10
                vec[o + 4 * v + i] = float(card.chain_free)

    def _private(self, vec: List[float], o: int, obs: Observation) -> None:
        v, r = len(self.vocab), len(RESOURCES)
        for name in obs.leaders_hand:
            vec[o + self._index[name]] = 1.0
        for name in obs.seen_cards:
            vec[o + v + self._index[name]] += 1.0
        for e in obs.history:  # the viewer knows the cards it discarded / buried itself
            if e.seat == obs.seat and e.kind in ('discard', 'wonder') and e.card:
                vec[o + 2 * v + self._index[e.card]] += 1.0
        o += 3 * v
        for k, side in enumerate(('left', 'right')):
            prices = obs.buy_prices.get(side, {})
            for j, res in enumerate(RESOURCES):
                vec[o + k * r + j] = prices.get(res, 2) / 2
        o += 2 * r
        vec[o : o + 2] = [float(obs.free_builds_left), float(obs.wonder_stage_affordable)]
        o += 2
        if obs.wonder_board is not None:
            for k, stage in enumerate(obs.wonder_board.stages[:MAX_STAGES]):
                s = o + k * STAGE_FEATURES
                _cost_into(vec, s, stage['cost'])
                effects = stage['effects']
                vec[s + r + 1 : s + r + 1 + EFFECT_FEATURES] = effect_features(effects)  # type: ignore[arg-type]
                vec[s + r + 1 + EFFECT_FEATURES] = float(bool(stage['built']))

    def _context(self, vec: List[float], o: int, obs: Observation) -> None:
        for k, age in enumerate((1, 2, 3)):
            vec[o + k] = float(obs.age == age)
        vec[o + 3 : o + 6] = [obs.turn / 7, len(obs.hand) / 8, obs.discard_count / 30]
        o += 6
        for k, count in enumerate(PLAYER_COUNTS):
            vec[o + k] = float(obs.n_players == count)
        o += len(PLAYER_COUNTS)
        leaders, cities = has_family(obs.expansions, 'leaders'), has_family(obs.expansions, 'cities')
        vec[o : o + 3] = [float(leaders), float(cities), float(obs.pass_left)]
        o += 3
        for k, decision in enumerate(self.decisions):
            vec[o + k] = float(obs.decision == decision)
