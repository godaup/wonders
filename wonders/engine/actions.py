"""The three things a player may do with the card they choose on their turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from wonders.cards.base import Card, IdentityEnum


class ActionKind(IdentityEnum):
    BUILD = 'build'  # build the structure into your city
    WONDER = 'wonder'  # bury the card to build your next wonder stage
    DISCARD = 'discard'  # discard for 3 coins
    FREE = 'free'  # build without paying: Olympia's free build, or Halicarnassus from the discard pile


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

    @classmethod
    def free(cls, card: Card) -> 'Action':
        return cls(ActionKind.FREE, card)


class DecisionKind(IdentityEnum):
    """Why the engine is asking for a decision."""

    PLAY = 'play'  # the regular simultaneous turn: every seat plays a card from hand
    LAST_CARD = 'last_card'  # Babylon B: play the 7th card instead of discarding it
    DISCARD_BUILD = 'discard_build'  # Halicarnassus: pick a discarded card to build for free
    DRAFT = 'draft'  # Leaders: keep one leader from the passed hand (expressed as BUILD of it)
    RECRUIT = 'recruit'  # Leaders: start of an Age, play one leader (recruit / wonder / discard)
    LEADER_BUILD = 'leader_build'  # Rome B: recruit an extra leader from hand right away (BUILD)
    COPY_LEADER = 'copy_leader'  # Courtesans' Guild: pick a neighbour's leader to copy (BUILD = pick)


@dataclass
class DecisionPoint:
    """A request for actions: each seat in ``options`` chooses one card from its list.

    For ``PLAY`` and ``LAST_CARD`` the options are the seat's hand; for
    ``DISCARD_BUILD`` they are the buildable cards in the discard pile.
    """

    age: int
    turn: int
    kind: DecisionKind
    options: Dict[int, List[Card]]
