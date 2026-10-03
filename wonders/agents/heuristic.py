"""A stronger, still-transparent agent: evaluate every legal action by a heuristic.

Unlike the myopic :class:`~wonders.agents.baselines.GreedyAgent` (which only reads a
card's immediate points), this scores each choice with a tunable value function that
accounts for:

* victory points and per-card/guild scoring (already in the effect summaries),
* the *marginal* science-set gain (Σn² + 7·sets is non-linear, so the value of a
  symbol depends on what you already have),
* military scaled by Age (later shields decide bigger conflict tokens),
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
from wonders.cards.base import ScienceSymbol
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import science_score
from wonders.engine.observation import CardView, Observation

DEFAULT_WEIGHTS: Dict[str, float] = {
    'vp': 1.0,
    'science': 1.0,
    'shield': 0.6,  # multiplied by Age (1..3)
    'coin': 0.34,  # 3 coins ~= 1 VP
    'production': 0.55,  # resources compound (fewer future trades / more builds)
    'trade': 0.5,
    'chain': 0.7,  # each structure this unlocks is a potential future free build
    'wonder_bonus': 0.75,  # tempo value of advancing the wonder at all
    'discard': 1.0,  # = the 3 coins a discard yields (fallback only)
    'free_build': 1.5,  # opportunity cost of spending Olympia's once-per-Age free build
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

    def decide(self, obs: Observation) -> Action:
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
        for e in effects:
            value += _num(e.get('victory_points')) * w['vp']
            value += _num(e.get('shields')) * w['shield'] * obs.age
            value += _num(e.get('coins')) * w['coin']
            production = e.get('production')
            if isinstance(production, list) and production:
                value += w['production'] * len(production)  # each producible slot
            if e.get('trade'):
                value += w['trade']
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
