"""Representation encoders: turn an Observation into an agent-friendly form.

The Observation is representation-agnostic on purpose. An :class:`Encoder` maps it
to whatever a given agent wants — a structured dict, a flat float vector, or (later)
a graph/tensor. Keeping this separate means several representations can be compared
without changing the engine or the agents' decision logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from wonders.cards.base import Color, ScienceSymbol
from wonders.engine.observation import CityView, Observation

_COLORS = [c.value for c in Color]
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
