"""The three things a player may do with the card they choose on their turn."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from wonders.cards.base import Card


class ActionKind(enum.Enum):
    BUILD = 'build'  # build the structure into your city
    WONDER = 'wonder'  # bury the card to build your next wonder stage
    DISCARD = 'discard'  # discard for 3 coins


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    card: Card

    @classmethod
    def build(cls, card: Card) -> 'Action':
        return cls(ActionKind.BUILD, card)

    @classmethod
    def wonder(cls, card: Card) -> 'Action':
        return cls(ActionKind.WONDER, card)

    @classmethod
    def discard(cls, card: Card) -> 'Action':
        return cls(ActionKind.DISCARD, card)
