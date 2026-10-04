"""Model vocabularies: the card names and decision kinds a network's inputs/outputs cover.

A network's action space (card name × kind) and :class:`~wonders.agents.encoders.CardEncoder`
input both depend on the card vocabulary, and the encoder also on the decision kinds.
:func:`vocab_for` gives the vocabulary of one expansion combination; :func:`universal_vocab`
covers *every* combination, so a single network can play base, Leaders and Cities games at
any table size (cards outside a game's content simply never appear, their inputs stay 0
and their actions are always masked).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple

from wonders.agents.encoders import DECISIONS, LEADERS_DECISIONS, CardEncoder
from wonders.cards.content import load_content
from wonders.env.action_space import ActionSpace
from wonders.ruleset import Ruleset

ALL_CARD_EXPANSIONS: Tuple[str, ...] = ('base', 'leaders', 'cities')


@dataclass(frozen=True)
class Vocab:
    """Card names (sorted) and decision kinds; picklable and hashable."""

    card_names: Tuple[str, ...]
    decisions: Tuple[str, ...]

    def action_space(self) -> ActionSpace:
        return ActionSpace(self.card_names)

    def encoder(self) -> CardEncoder:
        return CardEncoder(self.card_names, self.decisions)

    @property
    def obs_size(self) -> int:
        return self.encoder().size

    @property
    def n_actions(self) -> int:
        return self.action_space().n

    def covers(self, card_names: Sequence[str]) -> bool:
        return set(card_names) <= set(self.card_names)


@lru_cache(maxsize=None)
def vocab_for(expansions: Tuple[str, ...]) -> Vocab:
    """The vocabulary of one expansion combination (``base`` is always included)."""
    ruleset = Ruleset.of(*expansions)
    names = tuple(sorted({c.name for c in load_content(*ruleset.content_expansions).cards}))
    return Vocab(names, LEADERS_DECISIONS if ruleset.uses('leaders') else DECISIONS)


def universal_vocab() -> Vocab:
    """Every card of every shipped expansion, and every decision kind."""
    return vocab_for(ALL_CARD_EXPANSIONS)
