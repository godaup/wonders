"""Representation encoders: turn an Observation into an agent-friendly form.

The Observation is representation-agnostic on purpose. An :class:`Encoder` maps it
to whatever a given agent wants — a structured dict, a flat float vector, or (later)
a graph/tensor. Keeping this separate means several representations can be compared
without changing the engine or the agents' decision logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

from wonders.cards.base import OFFICIAL_COLORS, ScienceSymbol
from wonders.engine.observation import CityView, Observation

_COLORS = [c.value for c in OFFICIAL_COLORS]  # fixed: expansion colours must not resize inputs
_SYMBOLS = [s.value for s in ScienceSymbol]


class Encoder(ABC):
    """Maps an Observation to a representation consumable by an agent."""

    @abstractmethod
    def encode(self, obs: Observation):
        raise NotImplementedError


class DictEncoder(Encoder):
    """The full structured state as a nested (JSON-able) dict."""

    def encode(self, obs: Observation) -> Dict[str, object]:
        return obs.to_dict()


class FlatEncoder(Encoder):
    """A fixed-length numeric vector: self + both neighbours + turn context.

    Deliberately small and illustrative — it shows the seam for numeric encoders.
    With ``include_history`` it appends per-opponent counts of past action kinds, so
    time-dependent decisions have signal from the game history.
    """

    def __init__(self, include_history: bool = False):
        self.include_history = include_history

    def _city(self, c: CityView) -> List[float]:
        vec = [
            float(c.coins),
            float(c.shields),
            float(c.stages_built),
            float(c.military_tokens),
            float(c.score),
        ]
        vec += [float(c.science.get(s, 0)) for s in _SYMBOLS]
        vec += [float(c.cards_by_color.get(col, 0)) for col in _COLORS]
        return vec

    def encode(self, obs: Observation) -> List[float]:
        vec: List[float] = []
        for city in (obs.me, obs.left, obs.right):
            vec += self._city(city)
        vec += [1.0 if obs.age == a else 0.0 for a in (1, 2, 3)]
        vec += [
            float(obs.turn),
            float(len(obs.hand)),
            1.0 if obs.wonder_stage_affordable else 0.0,
            float(obs.discard_count),
        ]
        if self.include_history:
            vec += self._history(obs)
        return vec

    def _history(self, obs: Observation) -> List[float]:
        # For each seat: how many builds / wonders / discards / free builds it has visibly made.
        counts = {seat: [0.0, 0.0, 0.0, 0.0] for seat in range(obs.n_players)}
        kinds = {'build': 0, 'wonder': 1, 'discard': 2, 'free': 3}
        for move in obs.history:
            counts[move.seat][kinds[move.kind]] += 1.0
        out: List[float] = []
        for seat in range(obs.n_players):
            out += counts[seat]
        return out


DECISIONS = ('play', 'last_card', 'discard_build')
LEADERS_DECISIONS = DECISIONS + ('draft', 'recruit', 'leader_build', 'copy_leader')  # with Leaders


class CardEncoder(Encoder):
    """A fixed-length vector that names cards: multi-hot hand and tableaus.

    Layout (``vocab`` = the card-name vocabulary, e.g. ``ActionSpace.card_names``):

    * per city (me, left, right): :class:`FlatEncoder`'s scalar city features, then a
      multi-hot over ``vocab`` of the structures it has built;
    * a multi-hot over ``vocab`` of the cards in hand (the options being chosen from);
    * per hand card name: cost to build it now in coins (0 if not buildable);
    * turn context: Age one-hot, turn, hand size, discard count, wonder stage
      affordable, free builds left, and a one-hot of the decision kind (``decisions``;
      pass :data:`LEADERS_DECISIONS` when the Leaders draft/recruitment can occur).

    The size depends only on the vocabulary, not on the player count.
    """

    def __init__(self, vocab: Sequence[str], decisions: Sequence[str] = DECISIONS):
        self.vocab = list(vocab)
        self.decisions = tuple(decisions)
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.vocab)}
        self._flat = FlatEncoder()

    @property
    def size(self) -> int:
        city = 5 + len(_SYMBOLS) + len(_COLORS) + len(self.vocab)
        return 3 * city + 2 * len(self.vocab) + 3 + 5 + len(self.decisions)

    def _multi_hot(self, names) -> List[float]:
        vec = [0.0] * len(self.vocab)
        for name in names:
            vec[self._index[name]] = 1.0
        return vec

    def encode(self, obs: Observation) -> List[float]:
        vec: List[float] = []
        for city in (obs.me, obs.left, obs.right):
            vec += self._flat._city(city)
            vec += self._multi_hot(city.cards)
        vec += self._multi_hot(c.name for c in obs.hand)
        costs = [0.0] * len(self.vocab)
        for card in obs.hand:
            if card.build_coin_cost is not None:
                costs[self._index[card.name]] = float(card.build_coin_cost)
        vec += costs
        vec += [1.0 if obs.age == a else 0.0 for a in (1, 2, 3)]
        vec += [
            float(obs.turn),
            float(len(obs.hand)),
            float(obs.discard_count),
            1.0 if obs.wonder_stage_affordable else 0.0,
            float(obs.free_builds_left),
        ]
        vec += [1.0 if obs.decision == d else 0.0 for d in self.decisions]
        return vec
