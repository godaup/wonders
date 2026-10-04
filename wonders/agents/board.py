"""Board-aware agents: play around your own wonder board.

The other agents treat the wonder as "the next stage, if affordable". These read the
whole :class:`~wonders.engine.observation.WonderBoardView` and:

* value **board powers** that the plain heuristic scores as zero — Babylon B's 7th
  card and Olympia A's free build (worth more the more Ages remain), Halicarnassus'
  discard build (worth more once the discard pile has filled), Olympia B's guild copy,
  diplomacy, Cities masks and the Leaders powers (Rome's draws / extra recruits,
  Courtesans' copy);
* **plan resources** for the stages still to build: production cards that cover a
  resource the remaining stages need, but the city cannot make yet, get a bonus;
* **choose which card to spend**: every action removes the chosen card from the hand
  passed downstream, so the card the downstream neighbour wants most is preferred
  (most visibly when burying a card under a wonder stage).

:class:`BoardFocusAgent` additionally infers a colour strategy from its board (e.g.
shields on the stages → military, science → green, mostly VP → civic) and favours
those cards. Benchmarks show the colour lean does not add strength (larger ``focus``
weights lose ground); at the default weight it plays as well as BoardAwareAgent while
choosing differently, which makes it a useful, diverse self-play opponent. Like every
agent, both read only the Observation.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

from wonders.agents.heuristic import DEFAULT_WEIGHTS, HeuristicAgent, _num
from wonders.agents.strategies import _appeal_to
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import CardView, CityView, Observation

# Tuned by random search with ``wonders.agents.bench`` against a gauntlet of HeuristicAgent
# lineups (3-4 players, random boards) and checked on held-out seeds.
BOARD_WEIGHTS: Dict[str, float] = {
    **DEFAULT_WEIGHTS,
    'wonder_bonus': 0.46,
    'production': 0.72,
    'chain': 0.85,
    'last_card': 3.4,  # Babylon B: one extra card play per remaining Age
    'free_build_power': 2.0,  # Olympia A: one free build per remaining Age
    'discard_build': 3.2,  # Halicarnassus: best discarded card, scaled by how full the pile is
    'guild_copy': 6.9,  # Olympia B: a neighbour guild at the end
    'science_copy': 1.5,  # Cities masks
    'diplomacy': 1.5,  # per stronger neighbour whose defeat token is avoided
    'leader_power': 0.5,  # Rome B draws / extra recruits, Courtesans' copy (per unit)
    'plan': 5.0,  # production covering a resource the remaining stages still need
    'deny': 0.1,  # downstream neighbour's appetite for the card this action spends
    'focus': 0.6,  # BoardFocusAgent: bonus for cards of the board's colours
}

_RAW_OR_MANUFACTURED = {'wood', 'stone', 'ore', 'bricks', 'glass', 'linen', 'paper'}


def _remaining_ages(obs: Observation) -> int:
    """Ages in which a power gained now still applies (the current one included)."""
    return max(1, 4 - max(1, obs.age))


class BoardAwareAgent(HeuristicAgent):
    """HeuristicAgent that values board powers, plans stage resources and picks what to spend."""

    name = 'board'

    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = 'board'):
        super().__init__(dict(BOARD_WEIGHTS, **(weights or {})), name=name)

    # -- decision ------------------------------------------------------------
    def decide(self, obs: Observation) -> Action:
        views: Dict[str, CardView] = {c.name: c for c in obs.hand}
        candidates = [a for a in obs.legal_actions if a.kind is not ActionKind.DISCARD]
        if not candidates:
            discards = [a for a in obs.legal_actions if a.kind is ActionKind.DISCARD]
            return max(discards, key=lambda a: self._denial(obs, views.get(a.card.name)))

        def value(action: Action) -> float:
            card = views.get(action.card.name)
            if action.kind is ActionKind.WONDER:
                base = self._wonder_value(obs)
            elif action.kind is ActionKind.FREE and card is not None:
                spent = 0.0 if obs.decision == 'discard_build' else self.w['free_build']
                base = self._card_value(obs, card) + (card.build_coin_cost or 0) * self.w['coin'] - spent
            elif card is not None:
                base = self._card_value(obs, card)
            else:  # pragma: no cover - every legal action names a card in hand
                base = 0.0
            return base + self._denial(obs, card)

        return max(candidates, key=value)

    def _denial(self, obs: Observation, card: Optional[CardView]) -> float:
        """Value of keeping ``card`` away from the neighbour who receives this hand next."""
        if card is None or obs.decision != 'play':
            return 0.0
        downstream: CityView = obs.left if obs.age in (1, 3) else obs.right
        return self.w['deny'] * _appeal_to(card, downstream)

    # -- value function ------------------------------------------------------
    def _effects_value(self, obs: Observation, effects: List[Dict[str, object]]) -> float:
        return super()._effects_value(obs, effects) + self._power_value(obs, effects)

    def _power_value(self, obs: Observation, effects: Iterable[Dict[str, object]]) -> float:
        w = self.w
        ages = _remaining_ages(obs)
        value = 0.0
        for e in effects:
            value += w['last_card'] * ages * float(bool(e.get('plays_last_card')))
            value += w['free_build_power'] * ages * _num(e.get('free_builds_per_age'))
            pile = min(1.0, obs.discard_count / 8.0)
            value += w['discard_build'] * (0.4 + 0.6 * pile) * _num(e.get('discard_builds'))
            neighbour_guilds = obs.left.cards_by_color.get('purple', 0) + obs.right.cards_by_color.get('purple', 0)
            guild_odds = 1.0 if neighbour_guilds else (0.6 if obs.age < 3 else 0.3)
            value += w['guild_copy'] * guild_odds * _num(e.get('guild_copies'))
            value += w['science_copy'] * _num(e.get('science_copies'))
            leader_units = (
                _num(e.get('leader_draws')) / 4 + _num(e.get('extra_recruits')) + _num(e.get('leader_copies'))
            )
            value += w['leader_power'] * leader_units
            if e.get('type') == 'Diplomacy':
                stronger = sum(1 for nb in (obs.left, obs.right) if nb.shields > obs.me.shields)
                value += w['diplomacy'] * (0.5 + stronger)
        return value

    def _stage_needs(self, obs: Observation) -> Counter:
        """Resource units the unbuilt stages need beyond what the city already produces."""
        board = obs.wonder_board
        if board is None:
            return Counter()
        need: Counter = Counter()
        for stage in board.stages:
            if stage['built']:
                continue
            cost = stage['cost']
            resources = cost.get('resources', {}) if isinstance(cost, dict) else {}
            for res, n in resources.items():
                need[res] = max(need[res], int(n))
        supply: Counter = Counter()
        for slot in obs.production:
            for res in slot:
                supply[res] += 1
        return Counter({r: n - supply[r] for r, n in need.items() if n > supply[r]})

    def _card_value(self, obs: Observation, card: CardView) -> float:
        value = super()._card_value(obs, card)
        needs = self._stage_needs(obs)
        if needs:
            for e in card.effects:
                production = e.get('production')
                if isinstance(production, list):
                    for slot in production:
                        if any(r in needs for r in slot):
                            value += self.w['plan']
        return value


class BoardFocusAgent(BoardAwareAgent):
    """BoardAwareAgent that also leans into the colours its wonder board rewards."""

    name = 'board-focus'

    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = 'board-focus'):
        super().__init__(weights, name=name)

    @staticmethod
    def board_colors(obs: Observation) -> Set[str]:
        """Colours the board's stages point to (shields → red, science → green, ...)."""
        board = obs.wonder_board
        if board is None:
            return set()
        tally: Counter = Counter()
        for stage in board.stages:
            for e in stage['effects']:  # type: ignore[attr-defined]
                if e.get('shields'):
                    tally['red'] += 2
                if e.get('science') or e.get('science_wild') or e.get('science_copies'):
                    tally['green'] += 2
                if e.get('production'):
                    tally['brown'] += 1
                    tally['grey'] += 1
                if e.get('coins') or e.get('trade'):
                    tally['yellow'] += 1
                if e.get('guild_copies'):
                    tally['purple'] += 2
                if _num(e.get('victory_points')) >= 5:
                    tally['blue'] += 1
        if board.starting_resource in _RAW_OR_MANUFACTURED and board.starting_resource in {'glass', 'linen', 'paper'}:
            tally['green'] += 1  # manufactured-goods boards make science cheaper
        if not tally:
            return {'blue'}
        top = max(tally.values())
        return {color for color, n in tally.items() if n == top}

    def _card_value(self, obs: Observation, card: CardView) -> float:
        value = super()._card_value(obs, card)
        if card.color in self.board_colors(obs):
            value += self.w['focus']
        return value
