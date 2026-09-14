"""Game phases.

A game moves through three Ages, each a sequence of simultaneous turns, with a
military-conflict resolution at the end of every Age, then final scoring.
"""

from __future__ import annotations

import enum


class Phase(enum.Enum):
    SETUP = 'setup'
    AGE_I = 'age_1'
    AGE_II = 'age_2'
    AGE_III = 'age_3'
    CONFLICT = 'conflict'  # end-of-age military resolution
    SCORING = 'scoring'
    FINISHED = 'finished'

    @classmethod
    def for_age(cls, age: int) -> 'Phase':
        return {1: cls.AGE_I, 2: cls.AGE_II, 3: cls.AGE_III}[age]

    @property
    def pass_left(self) -> bool:
        """Ages I and III pass hands clockwise (left); Age II passes right."""
        return self in (Phase.AGE_I, Phase.AGE_III)
