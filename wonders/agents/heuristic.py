"""A stronger, still-transparent agent: evaluate every legal action by a heuristic.

Unlike the myopic :class:`~wonders.agents.baselines.GreedyAgent` (which only reads a
card's immediate points), this scores each choice with a tunable value function that
accounts for:

* victory points and per-card/guild scoring (already in the effect summaries),
* the *marginal* science-set gain (Σn² + 7·sets is non-linear, so the value of a
  symbol depends on what you already have),
* military: the swing in this Age's conflict tokens against each neighbour that the
  shields would cause right now, plus a smaller Age-scaled value for future conflicts,
* economy — coins, resource production, and trade discounts, and
* wonder-stage progression (valued from the observation's next-stage view).

The weights are exposed so this can double as a baseline policy to tune or to seed a
learned value function later. It reads only the Observation, so it respects the same
partial-information contract as every other agent.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from wonders.agents.base import Agent
from wonders.agents.leader_values import LEADER_VALUES
from wonders.cards.base import Card, ScienceSymbol
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import CONFLICT_VALUE, DEFEAT_VALUE, science_score
from wonders.engine.observation import CardView, Observation

# Tuned by random search against a fixed gauntlet of baseline lineups (3-5 players)
# with ``wonders.agents.bench`` and validated on held-out seeds.
DEFAULT_WEIGHTS: Dict[str, float] = {
    'vp': 0.89,
    'science': 1.0,
    'shield': 0.17,  # per shield, multiplied by Age (1..3): future conflicts
    'military_gain': 0.71,  # per conflict-token point the shields would swing this Age
    'coin': 0.35,  # 3 coins ~= 1 VP
    'production': 0.43,  # resources compound (fewer future trades / more builds)
    'trade': 0.62,
    'chain': 0.8,  # each structure this unlocks is a potential future free build
    'wonder_bonus': 0.53,  # tempo value of advancing the wonder at all
    'discard': 1.0,  # = the 3 coins a discard yields (fallback only)
    'free_build': 2.11,  # opportunity cost of spending Olympia's once-per-Age free build
    'leader': 1.0,  # per VP of a leader's data-driven value (wonders.agents.leader_values)
    'leader_base': 1.05,  # what the alternative to recruiting (3 coins / a stage) is worth on that scale
}

_SYMBOLS = {'tablet': ScienceSymbol.TABLET, 'gear': ScienceSymbol.GEAR, 'compass': ScienceSymbol.COMPASS}


def _num(value: object) -> float:
    """Effect-summary values are loosely typed; read numbers safely."""
    return float(value) if isinstance(value, (int, float)) else 0.0


class HeuristicAgent(Agent):
    name = 'heuristic'

    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = 'heuristic'):
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.name = name

    # -- Leaders (values estimated from random-leader games, see leader_values) ----
    @staticmethod
    def leader_value(name: str) -> float:
        return LEADER_VALUES.get(name, 0.0)

    def draft_leader(self, options: List[Card]) -> Card:
        return max(options, key=lambda c: self.leader_value(c.name))

    def recruit(self, leaders: List[Card], obs: Observation) -> Optional[Action]:
        """Recruit the best affordable leader unless a wonder stage or 3 coins is worth more."""
        if not leaders:
            return None
        w = self.w

        def value(action: Action) -> float:
            if action.kind is ActionKind.BUILD:
                return w['leader_base'] + w['leader'] * self.leader_value(action.card.name)
            if action.kind is ActionKind.WONDER:
                return self._wonder_value(obs)
            return 3 * w['coin']  # discard

        best = max(value(a) for a in obs.legal_actions)
        # Among equally good plays, give up (bury / discard) the least valuable leader.
        return max(
            (a for a in obs.legal_actions if value(a) == best),
            key=lambda a: self.leader_value(a.card.name) * (1 if a.kind is ActionKind.BUILD else -1),
        )

    def _leader_pick(self, obs: Observation) -> Optional[Action]:
        """Rome's extra recruit / the Courtesans' copy: the most valuable leader on offer."""
        builds = [a for a in obs.legal_actions if a.kind is ActionKind.BUILD]
        return max(builds, key=lambda a: self.leader_value(a.card.name)) if builds else None

    def decide(self, obs: Observation) -> Action:
        if obs.decision in ('leader_build', 'copy_leader'):
            pick = self._leader_pick(obs)
            if pick is not None:
                return pick
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        # Build (or advance the wonder) whenever possible — in this engine taking the
        # 3 discard coins over a compounding resource/chain card loses ground. Discard
        # is only a fallback when nothing can be built.
        candidates = [a for a in obs.legal_actions if a.kind is not ActionKind.DISCARD]
        if not candidates:
            return next(a for a in obs.legal_actions if a.kind is ActionKind.DISCARD)

        def value(action: Action) -> float:
            if action.kind is ActionKind.WONDER:
                return self._wonder_value(obs)
            card = views[action.card.name]
            if action.kind is ActionKind.FREE:
                spent = 0.0 if obs.decision == 'discard_build' else self.w['free_build']
                return self._card_value(obs, card) + (card.build_coin_cost or 0) * self.w['coin'] - spent
            return self._card_value(obs, card)

        return max(candidates, key=value)

    # -- value function ---------------------------------------------------
    def _effects_value(self, obs: Observation, effects: List[Dict[str, object]]) -> float:
        w = self.w
        value = 0.0
        shields = 0.0
        for e in effects:
            value += _num(e.get('victory_points')) * w['vp']
            shields += _num(e.get('shields'))
            value += _num(e.get('coins')) * w['coin']
            production = e.get('production')
            if isinstance(production, list) and production:
                value += w['production'] * len(production)  # each producible slot
            if e.get('trade'):
                value += w['trade']
        if shields:
            value += shields * w['shield'] * obs.age
            value += self._military_gain(obs, shields) * w['military_gain']
        value += self._science_gain(obs, effects) * w['science']
        return value

    def _card_value(self, obs: Observation, card: CardView) -> float:
        value = self._effects_value(obs, card.effects)
        value += self.w['chain'] * len(card.chain_enables)  # tempo: future free builds
        value -= (card.build_coin_cost or 0) * self.w['coin']
        return value

    def _wonder_value(self, obs: Observation) -> float:
        stage = obs.next_wonder_stage
        if stage is None:
            return float('-inf')  # nothing to build
        return self.w['wonder_bonus'] + self._effects_value(obs, stage.effects)

    @staticmethod
    def _military_gain(obs: Observation, extra: float) -> float:
        """Conflict-token swing (this Age's values) from ``extra`` shields, vs current neighbours."""
        age = max(1, min(3, obs.age))

        def outcome(mine: float, theirs: float) -> int:
            if mine > theirs:
                return CONFLICT_VALUE[age]
            return DEFEAT_VALUE if mine < theirs else 0

        me = obs.me.shields
        return float(sum(outcome(me + extra, nb.shields) - outcome(me, nb.shields) for nb in (obs.left, obs.right)))

    def _science_gain(self, obs: Observation, effects: List[Dict[str, object]]) -> float:
        base: Counter = Counter()
        for symbol, n in obs.me.science.items():
            if symbol in _SYMBOLS:
                base[_SYMBOLS[symbol]] += n
        after = Counter(base)
        wilds = 0
        changed = False
        for e in effects:
            sym = e.get('science')
            if isinstance(sym, str) and sym in _SYMBOLS:
                after[_SYMBOLS[sym]] += 1
                changed = True
            wilds += int(_num(e.get('science_wild')))
        if not changed and not wilds:
            return 0.0
        return science_score(after, wilds) - science_score(base)
