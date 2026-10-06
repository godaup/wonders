"""Adaptive scripted agents: board-aware play whose priorities follow the state of the table.

:class:`AdaptiveAgent` keeps :class:`~wonders.agents.board.BoardAwareAgent`'s value function
(and the data-driven leader play) but replaces its fixed weights with situational rules:

* **Contestable military fronts.** Each neighbour's shields are projected to every remaining
  conflict from their pace so far; extra shields are worth the change in *expected* conflict
  tokens (a smooth win probability per front, later conflicts discounted because they can
  still be contested later). Hopeless and already-safe fronts get no investment.
* **Phase awareness.** Production and trade compound only while there are turns left to use
  them, and chains only pay off in a later Age: their value decays with the game.
* **Science commitment.** Beyond the immediate (non-linear) science gain, a symbol is worth
  more when the city already collects science, turns remain and few rivals compete for green.
* **Wheel awareness.** While a hand will come back around (more cards than players), a card
  nobody else wants can wait: contested cards are taken first.

:class:`RivalAgent` adds standings awareness: it tracks the score leader among the
opponents, denies cards to that player when they receive the hand next, and weighs the
military front against them more.

Both read only the Observation (public tableaus, own hand), like every agent.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from wonders.agents.board import BOARD_WEIGHTS, BoardAwareAgent
from wonders.agents.heuristic import _num
from wonders.agents.strategies import _appeal_to
from wonders.cards.content import load_content
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import CONFLICT_VALUE, DEFEAT_VALUE
from wonders.engine.observation import CardView, CityView, Observation, _static_summary
from wonders.ruleset import Ruleset

Spec = Dict[str, object]

ADAPTIVE_WEIGHTS: Dict[str, float] = {
    **BOARD_WEIGHTS,
    'front': 0.9,  # per expected conflict-token point gained across the remaining conflicts
    'front_noise': 1.0,  # shield-gap scale of the win probability (projection uncertainty)
    'front_discount': 0.6,  # later conflicts can still be contested later
    'front_prior': 0.25,  # prior shields per turn for projecting a city's military pace
    'science_potential': 1.2,  # future value of a symbol for a committed, uncontested collector
    'production_floor': 0.15,  # share of production / trade value left on the last turns
    'wheel': 0.35,  # share of an uncontested card's value given up by waiting for it to wheel
    'rival_deny': 0.4,  # RivalAgent: denial weight when the score leader receives the hand next
    'rival_front': 1.5,  # RivalAgent: multiplier for the military front against the score leader
    'combo': 0.5,  # value of what a card adds to payoff effects the city already owns
    'combo_future': 0.35,  # share of a counting card's projected future count that is credited now
    'combo_hand': 0.5,  # weight of payoffs on leaders still in hand (not recruited yet)
    'feed': 0.15,  # cost of a card a neighbour's guild counts (their VP)
    'stage_odds': 0.3,  # weight of unbuilt wonder stages (their shields / science / tallies) in projections
    'pace_prior': 0.15,  # prior cards of a colour per turn when projecting a city's collection
    # Research-backed rules (strategy guides + top-player game data, see the module docstring):
    'np_military': 1.0,  # 0..1: blend in the player-count military profile (MILITARY_BY_PLAYERS)
    'upstream_front': 0.6,  # Age III: the upstream neighbour sees the cards first; discount that front
    'science_rules': 1.0,  # 0..1: stop committing after two sets, no fresh commitment in Age III
    'raw_soft_cap': 4.0,  # own production slots beyond which more production is worth less
    'raw_penalty': 0.3,  # value lost per slot over the cap (as a divisor term)
    'unique_resource': 0.3,  # +/- share for a resource neither / both neighbours produce
    'coin_hoard': 10.0,  # coins beyond which extra coins are worth less (spend, don't bank)
    'coin_hoard_factor': 0.6,  # value of coins beyond the hoard threshold
}

# Military matters most at small tables (3-4 players: near-mandatory) and least at 6-7.
MILITARY_BY_PLAYERS: Dict[int, float] = {3: 1.25, 4: 1.15, 5: 1.0, 6: 0.75, 7: 0.6}


@lru_cache(maxsize=16)
def _catalog(expansions: Tuple[str, ...]) -> Dict[str, Tuple[str, Tuple[Dict[str, object], ...]]]:
    """Public card text: name -> (colour, static effect summaries) for every card in play."""
    content = load_content(*Ruleset.of(*expansions).content_expansions)
    cards = {c.name: c for c in content.cards}
    cards.update((c.name, c) for c in content.leaders())
    return {name: (c.color.value, tuple(_static_summary(e) for e in c.effects)) for name, c in cards.items()}


def _turns_per_age(obs: Observation) -> int:
    return 7 if 'cities' in obs.expansions else 6


def _turns_left(obs: Observation) -> int:
    """Card plays left in the game, this one included (approximate during special decisions)."""
    per_age = _turns_per_age(obs)
    this_age = max(1, len(obs.hand) - 1) if obs.decision == 'play' else max(1, per_age - obs.turn)
    return this_age + per_age * max(0, 3 - max(1, obs.age))


def _spec_colors(spec: Spec) -> List[str]:
    raw = spec.get('colors')
    return [str(c) for c in raw] if isinstance(raw, list) else []


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AdaptiveAgent(BoardAwareAgent):
    """BoardAwareAgent whose military, economy, science and draft priorities adapt to the table."""

    name = 'adaptive'

    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = 'adaptive'):
        super().__init__(dict(ADAPTIVE_WEIGHTS, **(weights or {})), name=name)

    # -- situational pieces -----------------------------------------------------
    def _phase(self, obs: Observation) -> float:
        """1.0 at the start of the game, falling to ``production_floor`` on the last turn."""
        total = 3 * _turns_per_age(obs)
        left = min(total, _turns_left(obs))
        floor = self.w['production_floor']
        return floor + (1.0 - floor) * (left - 1) / max(1, total - 1)

    def _front_weight(self, obs: Observation, neighbour: CityView) -> float:
        w = self.w
        weight = 1.0 + w['np_military'] * (MILITARY_BY_PLAYERS.get(obs.n_players, 1.0) - 1.0)
        upstream = obs.right if obs.pass_left else obs.left  # the neighbour who passes to us
        if obs.age == 3 and neighbour.seat == upstream.seat:
            weight *= w['upstream_front']
        return weight

    def _front_gain(self, obs: Observation, extra: float) -> float:
        """Expected conflict tokens gained over the remaining conflicts from ``extra`` shields."""
        w = self.w
        per_age = _turns_per_age(obs)
        age = max(1, min(3, obs.age))
        played = (age - 1) * per_age + max(0, per_age - max(1, len(obs.hand) - 1))

        def pace(city: CityView) -> float:  # shields per turn, shrunk towards a prior early on
            return (city.shields + w['front_prior'] * per_age) / (played + per_age)

        # Wonder combo: shields on unbuilt stages will likely join the city's own military.
        stage_shields = w['stage_odds'] * sum(_num(e.get('shields')) for e in self._stage_effects(obs))
        gain = 0.0
        for nb in (obs.left, obs.right):
            weight = self._front_weight(obs, nb)
            for a in range(age, 4):
                turns_until = (a - age) * per_age + max(0, len(obs.hand) - 1)
                theirs = nb.shields + pace(nb) * turns_until
                share = min(1.0, turns_until / max(1, _turns_left(obs) - 1))  # stages come over the game
                mine = obs.me.shields + pace(obs.me) * turns_until + stage_shields * share
                gain += (
                    weight
                    * w['front_discount'] ** (a - age)
                    * (self._expected_tokens(mine + extra, theirs, a) - self._expected_tokens(mine, theirs, a))
                )
        return gain

    def _expected_tokens(self, mine: float, theirs: float, age: int) -> float:
        scale = self.w['front_noise']
        win = _sigmoid((mine - theirs - 0.5) / scale)
        lose = _sigmoid((theirs - mine - 0.5) / scale)
        return win * CONFLICT_VALUE[age] + lose * DEFEAT_VALUE

    def _science_potential(self, obs: Observation) -> float:
        mine: float = sum(obs.me.science.values()) + obs.me.science_wilds
        # Wonder combo: science on unbuilt stages (e.g. Babylon) counts towards the commitment.
        mine += self.w['stage_odds'] * sum(
            1 for e in self._stage_effects(obs) if e.get('science') or e.get('science_wild')
        )
        if self.w['science_rules']:
            symbols = [obs.me.science.get(k, 0) for k in ('tablet', 'gear', 'compass')]
            if min(symbols) + obs.me.science_wilds >= 2:  # two sets: stop investing beyond the exact gain
                return 0.0
            if obs.age >= 3 and mine < 2:  # too late to start a science plan
                return 0.0
        others = [c for c in obs.table if c.seat != obs.seat]
        rivals = sum(c.cards_by_color.get('green', 0) for c in others) / max(1, len(others))
        future = (_turns_left(obs) - 1) / (3 * _turns_per_age(obs))
        return future * min(mine, 4) / (1.0 + rivals)

    def _coin_value(self, obs: Observation, coins: float) -> float:
        """Coins are worth less once the treasury is large (spend them, don't bank them)."""
        w = self.w
        room = max(0.0, w['coin_hoard'] - obs.me.coins)
        plain = min(coins, room)
        return (plain + (coins - plain) * w['coin_hoard_factor']) * w['coin']

    def _production_factor(self, obs: Observation, slots: List[object]) -> float:
        """Production slots, scaled by scarcity among the neighbours and the city's surplus."""
        w = self.w
        neighbours = [set(r for slot in nb.production for r in slot) for nb in (obs.left, obs.right)]
        total = 0.0
        for slot in slots:
            resources = set(slot) if isinstance(slot, list) else set()
            covered = sum(1 for have in neighbours if resources & have)
            total += 1.0 + w['unique_resource'] * (1 - covered)  # +share if neither, -share if both
        surplus = max(0.0, len(obs.production) - w['raw_soft_cap'])
        return total / (1.0 + w['raw_penalty'] * surplus)

    # -- combos ---------------------------------------------------------------------
    @staticmethod
    def _specs(effects: Iterable[Dict[str, object]]) -> List[Spec]:
        specs: List[Spec] = []
        for e in effects:
            counts = e.get('counts')
            if isinstance(counts, dict):
                specs.append(counts)
        return specs

    def _city_specs(self, obs: Observation, names: Iterable[str]) -> List[Spec]:
        catalog = _catalog(tuple(obs.expansions) or ('base',))
        return [spec for name in names if name in catalog for spec in self._specs(catalog[name][1])]

    def _own_payoffs(self, obs: Observation) -> List[Tuple[Spec, float]]:
        """Payoff effects the city owns (weight 1) or holds as unrecruited leaders (``combo_hand``)."""
        owned = self._city_specs(obs, [*obs.me.cards, *obs.me.copied_leaders])
        board = obs.wonder_board
        if board is not None:
            owned += self._specs(board.effects)
            for stage in board.stages:
                effects = stage['effects']
                if stage['built'] and isinstance(effects, list):
                    owned += self._specs(e for e in effects if isinstance(e, dict))
        held = self._city_specs(obs, obs.leaders_hand)
        return [(s, 1.0) for s in owned] + [(s, self.w['combo_hand']) for s in held]

    def _played(self, obs: Observation) -> int:
        per_age = _turns_per_age(obs)
        return (max(1, obs.age) - 1) * per_age + max(0, per_age - max(1, len(obs.hand) - 1))

    def _pace(self, obs: Observation, city: CityView, colors: Iterable[object]) -> float:
        """Cards of ``colors`` the city is projected to add over the rest of the game."""
        per_age = _turns_per_age(obs)
        have = sum(city.cards_by_color.get(str(c), 0) for c in colors)
        rate = (have + self.w['pace_prior'] * per_age) / (self._played(obs) + per_age)
        return rate * max(0, _turns_left(obs) - 1)

    def _unbuilt_stages(self, obs: Observation) -> List[Dict[str, object]]:
        board = obs.wonder_board
        return [s for s in board.stages if not s['built']] if board is not None else []

    def _stage_effects(self, obs: Observation) -> List[Dict[str, object]]:
        """Effect summaries of the unbuilt wonder stages."""
        out: List[Dict[str, object]] = []
        for stage in self._unbuilt_stages(obs):
            effects = stage['effects']
            if isinstance(effects, list):
                out += [e for e in effects if isinstance(e, dict)]
        return out

    def _enabler_value(self, obs: Observation, card: CardView) -> float:
        """What ``card`` adds to the city's payoffs, minus what it feeds neighbours' guilds."""
        w = self.w
        value = 0.0
        for spec, weight in self._own_payoffs(obs):
            colors = _spec_colors(spec)
            if card.color not in colors:
                continue
            what = spec.get('what')
            if what == 'colors' and spec.get('scope') in ('self', 'both'):
                value += weight * _num(spec.get('vp_each')) * w['vp']
            elif what == 'build_income':
                if (spec.get('via_chain') and not card.chain_free) or (
                    spec.get('after_trade') and not card.trade_coins
                ):
                    continue
                value += weight * _num(spec.get('coins_each')) * w['coin']
            elif what == 'color_sets':
                have = {c: obs.me.cards_by_color.get(c, 0) for c in colors}
                completes = have[card.color] == min(have.values())
                value += weight * _num(spec.get('vp_each')) * w['vp'] * (1.0 if completes else 0.2)
        for nb in (obs.left, obs.right):
            for spec in self._city_specs(obs, nb.cards):
                if spec.get('what') == 'colors' and spec.get('scope') in ('neighbours', 'both'):
                    if card.color in _spec_colors(spec):
                        value -= w['feed'] * _num(spec.get('vp_each')) * w['vp']
        return w['combo'] * value

    def _payoff_future(self, obs: Observation, card: CardView) -> float:
        """Projected future units of a counting card (its current count is already in its VP)."""
        w = self.w
        value = 0.0
        for spec in self._specs(card.effects):
            what, scope = spec.get('what'), spec.get('scope')
            cities: List[CityView] = []
            if scope in ('self', 'both'):
                cities.append(obs.me)
            if scope in ('neighbours', 'both'):
                cities += [obs.left, obs.right]
            colors = _spec_colors(spec)
            units: float = 0.0
            if what == 'colors':
                units = sum(self._pace(obs, c, colors) for c in cities)
                value += _num(spec.get('vp_each')) * units * w['vp']
            elif what == 'wonder_stages':
                if obs.me in cities:
                    units += w['stage_odds'] * len(self._unbuilt_stages(obs))
                units += sum(
                    w['stage_odds'] * max(0, c.stages_total - c.stages_built) for c in cities if c is not obs.me
                )
                value += _num(spec.get('vp_each')) * units * w['vp']
            elif what == 'build_income':
                units = self._pace(obs, obs.me, colors) if colors else 0.8 * max(0, _turns_left(obs) - 1)
                value += _num(spec.get('coins_each')) * units * w['coin']
        return w['combo_future'] * value

    def _stage_payoffs(self, obs: Observation) -> float:
        """What one more built wonder stage adds to the city's stage-counting payoffs."""
        value = 0.0
        for spec, weight in self._own_payoffs(obs):
            if spec.get('what') == 'wonder_stages' and spec.get('scope') in ('self', 'both'):
                value += weight * _num(spec.get('vp_each')) * self.w['vp']
        return self.w['combo'] * value

    def _wonder_value(self, obs: Observation) -> float:
        value = super()._wonder_value(obs)
        return value if value == float('-inf') else value + self._stage_payoffs(obs)

    # -- value function -----------------------------------------------------------
    def _effects_value(self, obs: Observation, effects: List[Dict[str, object]]) -> float:
        w = self.w
        phase = self._phase(obs)
        value = 0.0
        shields = 0.0
        adds_symbol = False
        for e in effects:
            value += _num(e.get('victory_points')) * w['vp']
            shields += _num(e.get('shields'))
            value += self._coin_value(obs, _num(e.get('coins')))
            production = e.get('production')
            if isinstance(production, list) and production:
                value += w['production'] * phase * self._production_factor(obs, production)
            if e.get('trade'):
                value += w['trade'] * phase
            adds_symbol = adds_symbol or bool(e.get('science')) or bool(e.get('science_wild'))
        if shields:
            value += w['front'] * self._front_gain(obs, shields)
        value += self._science_gain(obs, effects) * w['science']
        if adds_symbol:
            value += w['science_potential'] * self._science_potential(obs)
        return value + self._power_value(obs, effects)

    def _card_value(self, obs: Observation, card: CardView) -> float:
        value = super()._card_value(obs, card)
        later_ages = max(0, 3 - max(1, obs.age)) / 2  # chains pay off only in a later Age
        value -= self.w['chain'] * len(card.chain_enables) * (1.0 - later_ages)
        return value + self._enabler_value(obs, card) + self._payoff_future(obs, card)

    def _wheel_discount(self, obs: Observation, card: Optional[CardView]) -> float:
        """Share of value to give up for a card likely to come back around uncontested."""
        if card is None or obs.decision != 'play' or len(obs.hand) <= obs.n_players:
            return 0.0
        others = [c for c in obs.table if c.seat != obs.seat]
        contest = max((_appeal_to(card, c) for c in others), default=0.0)
        return self.w['wheel'] * max(0.0, 1.0 - contest / 3.0)

    # -- decision -------------------------------------------------------------------
    def decide(self, obs: Observation) -> Action:
        if obs.decision in ('leader_build', 'copy_leader'):
            pick = self._leader_pick(obs)
            if pick is not None:
                return pick
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        candidates = [a for a in obs.legal_actions if a.kind is not ActionKind.DISCARD]
        if not candidates:
            discards = [a for a in obs.legal_actions if a.kind is ActionKind.DISCARD]
            return max(discards, key=lambda a: self._denial(obs, views.get(a.card.name)))

        def value(action: Action) -> float:
            card = views.get(action.card.name)
            if action.kind is ActionKind.WONDER:
                base = self._wonder_value(obs)
            elif card is None:  # pragma: no cover - every legal action names a card in hand
                base = 0.0
            else:
                base = self._card_value(obs, card)
                if action.kind is ActionKind.FREE:
                    spent = 0.0 if obs.decision == 'discard_build' else self.w['free_build']
                    base += (card.build_coin_cost or 0) * self.w['coin'] - spent
                base -= max(0.0, base) * self._wheel_discount(obs, card)
            return base + self._denial(obs, card)

        return max(candidates, key=value)


class RivalAgent(AdaptiveAgent):
    """AdaptiveAgent that also plays against the current score leader among its opponents."""

    name = 'rival'

    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = 'rival'):
        super().__init__(weights, name=name)

    @staticmethod
    def _rival(obs: Observation) -> CityView:
        return max((c for c in obs.table if c.seat != obs.seat), key=lambda c: c.score)

    def _front_weight(self, obs: Observation, neighbour: CityView) -> float:
        base = super()._front_weight(obs, neighbour)
        return base * (self.w['rival_front'] if neighbour.seat == self._rival(obs).seat else 1.0)

    def _denial(self, obs: Observation, card: Optional[CardView]) -> float:
        value = super()._denial(obs, card)
        if card is None or obs.decision != 'play':
            return value
        downstream = obs.left if obs.pass_left else obs.right
        if downstream.seat == self._rival(obs).seat and downstream.score >= obs.me.score:
            value += self.w['rival_deny'] * _appeal_to(card, downstream)
        return value
