"""Armada for scripted agents: what advancing a fleet is worth, and which island to keep.

Values are victory-point equivalents (3 coins ≈ 1 VP), read only from the :class:`Observation`. The fleet
spaces are the same on every Shipyard board, so they are written out here; only the yellow coins differ
between the editions (the first edition's board figure prints 1 more each), which this ignores.
"""

from __future__ import annotations

from typing import List, Optional

from wonders.cards.base import Card
from wonders.effects.library import VictoryPoints
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import Observation

# Per space 0 (start) .. 6, the value the space shows (or gives on arrival).
NAVAL = {'red': (0, 1, 2, 3, 4, 5, 8), 'blue': (0, 0, 0, 0, 0, 0, 1), 'yellow': (0, 0, 0, 0, 0, 0, 2)}
BLUE_VP = (0, 1, 2, 4, 6, 8, 10)
LEVEL = (0, 1, 1, 2, 2, 3, 4)
YELLOW_COINS = (0, 0, 2, 3, 4, 5, 0)
TAX = (0, 0, 1, 0, 2, 0, 4)
ISLAND_AT = {2: 1, 4: 2, 6: 3}  # green space -> island level explored
ISLAND_VALUE = {1: 2.5, 2: 3.5, 3: 5.0}
COIN_VP = 1 / 3
NAVAL_VP = 0.8  # VP of one naval strength per naval conflict still to come (rough share of the tokens)


def naval_game(obs: Observation) -> bool:
    """Whether ``obs`` is from an Armada game (the seat has a Shipyard)."""
    return bool(obs.naval_costs)


def fleet_value(obs: Observation, track: str) -> Optional[float]:
    """VP equivalent of advancing the own ``track`` fleet one space now (``None`` at the end)."""
    pos = obs.me.fleets.get(track, 0)
    if pos >= 6:
        return None
    nxt = pos + 1
    conflicts = max(1, 4 - max(1, obs.age))  # naval conflicts still to come, this Age's included
    value = 0.0
    if track in NAVAL:
        value += (NAVAL[track][nxt] - NAVAL[track][pos]) * NAVAL_VP * conflicts
    if track == 'blue':
        value += BLUE_VP[nxt] - BLUE_VP[pos]
    elif track == 'yellow':
        others = max(1, obs.n_players - 1)
        value += YELLOW_COINS[nxt] * COIN_VP + TAX[nxt] * others * 0.2 + (LEVEL[nxt] - LEVEL[pos]) * 0.3 * conflicts
    elif track == 'green':
        if nxt in ISLAND_AT:
            value += ISLAND_VALUE[ISLAND_AT[nxt]]
        else:
            value += 0.4 * ISLAND_VALUE[ISLAND_AT[nxt + 1]]
    return value


def naval_cost(obs: Observation, track: str) -> float:
    """A rough coin cost of the next Naval Construction on ``track``: its coins, plus 2 per resource not produced."""
    cost = obs.naval_costs.get(track)
    if cost is None:
        return 0.0
    produced = set(r for slot in obs.production for r in slot)
    resources = cost.get('resources', {})
    assert isinstance(resources, dict)
    missing = sum(n for r, n in resources.items() if r not in produced)
    coins = cost.get('coins', 0)
    assert isinstance(coins, int)
    return coins + 2.0 * missing


def track_of(obs: Observation, action: Action) -> Optional[str]:
    if action.kind is ActionKind.DISCARD:
        return 'yellow'
    if action.kind is ActionKind.WONDER:
        return obs.wonder_track
    return action.card.color.value


def with_naval(obs: Observation, action: Action, margin: float = 0.25) -> Action:
    """``action`` with a Naval Construction when that is legal and worth its cost (a sale: more than the coins)."""
    if action.naval or not naval_game(obs):
        return action
    upgraded = action.with_naval()
    if upgraded not in obs.legal_actions:
        return action
    track = track_of(obs, action)
    value = fleet_value(obs, track) if track is not None else None
    if track is None or value is None:
        return action
    if action.kind is ActionKind.DISCARD:
        return upgraded if value > 3 * COIN_VP else action
    return upgraded if value - naval_cost(obs, track) * COIN_VP > margin else action


def best_fleet(options: List[Card], obs: Observation) -> Card:
    """The fleet (a fleet card) whose free advance is worth most."""
    return max(options, key=lambda c: fleet_value(obs, c.color.value) or 0.0)


def island_value(card: Card, obs: Observation) -> float:
    """A rough VP value of keeping ``card`` (an island), from its effects' hooks."""
    conflicts = max(1, 4 - max(1, obs.age))
    value = 0.0
    for effect in card.effects:
        if isinstance(effect, VictoryPoints):
            value += effect.points
        value += 1.5 * len(effect.production())
        value += 1.0 * effect.shields() * conflicts / 2
        value += NAVAL_VP * effect.naval_strength() * conflicts
        for track in effect.fleet_advances():
            value += fleet_value(obs, track) or 0.0
        value += 1.5 * sum(effect.fleet_choices())
        if effect.science_wild() or effect.science_most():
            value += 4.0
        if effect.coin_loss_immune() or effect.naval_opt_out() or effect.fleet_income() or effect.wonder_fleet_choice():
            value += 2.0
        counts = effect.counts()
        if counts or type(effect).__name__ == 'BestColor':
            value += 4.0
        if effect.resource_discount('naval'):
            value += 2.0
    return value


def best_island(options: List[Card], obs: Observation) -> Card:
    return max(options, key=lambda c: island_value(c, obs))
