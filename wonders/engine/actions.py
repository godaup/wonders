"""The three things a player may do with the card they choose on their turn."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

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
    # Optional (left, right) coins to pay the neighbours for resources (BUILD / WONDER). ``None``
    # pays the cheapest way. Not part of equality, so an action with a chosen split still
    # matches its legal action (and its action-space index).
    payment: Optional[Tuple[int, int]] = field(default=None, compare=False)
    # Optional index into ``Game.stage_options(seat)`` for a WONDER action: which stage to build when
    # there is a choice (the Great Wall builds its stages in any order; the Manneken Pis picks the
    # Great Wall stage it copies). ``None`` builds the first affordable option. Not part of equality.
    stage: Optional[int] = field(default=None, compare=False)
    # Armada: also undertake a Naval Construction (advance the fleet of the card's colour, or the wonder's
    # fleet, paying its cost with the card's); on a DISCARD, sell to advance the yellow fleet instead of
    # taking coins. Part of equality: it changes what happens (and the action-space index).
    naval: bool = False

    def paying(self, left: int, right: int) -> 'Action':
        """This action with a chosen split of the trade between the neighbours."""
        return replace(self, payment=(left, right))

    def with_naval(self) -> 'Action':
        """This action with a Naval Construction (Armada)."""
        return replace(self, naval=True)

    def staged(self, option: int) -> 'Action':
        """This WONDER action building the stage option ``option`` (see ``Game.stage_options``)."""
        return replace(self, stage=option)

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
    ENTOMB_LEADER = 'entomb_leader'  # Abu Simbel: pick a recruited leader to entomb (BUILD = pick)
    EXPLORE = 'explore'  # Armada: keep one of the islands offered (BUILD = pick)
    ADVANCE_FLEET = 'advance_fleet'  # Armada: advance a fleet of your choice for free (BUILD of a fleet = pick)


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
