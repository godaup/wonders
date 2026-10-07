"""Model vocabularies: the card names and decision kinds a network's inputs/outputs cover.

A network's action space (card name × kind) and :class:`~wonders.agents.encoders.CardEncoder`
input both depend on the card vocabulary, and the encoder also on the decision kinds.
:func:`vocab_for` gives the vocabulary of one expansion combination; :func:`universal_vocab`
covers *every* combination, so a single network can play base, Leaders and Cities games at
any table size (cards outside a game's content simply never appear, their inputs stay 0
and their actions are always masked).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, Sequence, Tuple, Union

from wonders.agents.encoders import DECISIONS, LEADERS_DECISIONS, CardEncoder
from wonders.agents.table_encoder import TableEncoder
from wonders.cards.content import load_content
from wonders.env.action_space import ActionSpace
from wonders.ruleset import Ruleset

ALL_CARD_EXPANSIONS: Tuple[str, ...] = ('base', 'leaders', 'cities')


ENCODERS: Tuple[str, ...] = ('card-v1', 'table-v2')
VectorEncoder = Union[CardEncoder, TableEncoder]  # encoders with a fixed vector ``size``


DEFAULT_ENCODER = 'table-v2'  # for new models; checkpoints without an encoder entry load as 'card-v1'


@dataclass(frozen=True)
class Vocab:
    """Card names (sorted), decision kinds, wonder names and encoder version; picklable.

    ``encoder_kind`` picks the observation encoder: ``'card-v1'``
    (:class:`~wonders.agents.encoders.CardEncoder`) or ``'table-v2'``
    (:class:`~wonders.agents.table_encoder.TableEncoder`, which also needs ``wonders``).
    """

    card_names: Tuple[str, ...]
    decisions: Tuple[str, ...]
    wonders: Tuple[str, ...] = ()
    encoder_kind: str = DEFAULT_ENCODER

    def action_space(self) -> ActionSpace:
        return ActionSpace(self.card_names)

    def encoder(self) -> VectorEncoder:
        if self.encoder_kind == 'table-v2':
            return TableEncoder(self.card_names, self.decisions, self.wonders)
        if self.encoder_kind == 'card-v1':
            return CardEncoder(self.card_names, self.decisions)
        raise ValueError(f'unknown encoder {self.encoder_kind!r}; choose from {ENCODERS}')

    def meta(self) -> Dict[str, Any]:
        """What a checkpoint stores to rebuild this vocabulary (besides the card names)."""
        return {'decisions': list(self.decisions), 'wonders': list(self.wonders), 'encoder': self.encoder_kind}

    @classmethod
    def from_checkpoint(cls, card_names: Sequence[str], meta: Dict[str, Any]) -> 'Vocab':
        """Rebuild from a checkpoint (older checkpoints: base decisions, the v1 encoder)."""
        return cls(
            tuple(card_names),
            tuple(meta.get('decisions', DECISIONS)),
            tuple(meta.get('wonders', ())),
            meta.get('encoder', 'card-v1'),
        )

    def with_encoder(self, kind: str) -> 'Vocab':
        return replace(self, encoder_kind=kind)

    @property
    def obs_size(self) -> int:
        return self.encoder().size

    @property
    def n_actions(self) -> int:
        return self.action_space().n

    def covers(self, card_names: Sequence[str]) -> bool:
        return set(card_names) <= set(self.card_names)


@lru_cache(maxsize=None)
def vocab_for(expansions: Tuple[str, ...], encoder: str = DEFAULT_ENCODER) -> Vocab:
    """The vocabulary of one expansion combination (``base`` is always included)."""
    ruleset = Ruleset.of(*expansions)
    content = load_content(*ruleset.content_expansions)
    names = tuple(sorted({c.name for c in content.cards}))
    wonders = tuple(sorted({w.name for w in content.wonders}))
    return Vocab(names, LEADERS_DECISIONS if ruleset.uses('leaders') else DECISIONS, wonders, encoder)


def universal_vocab(encoder: str = DEFAULT_ENCODER) -> Vocab:
    """Every card and wonder of every shipped expansion, and every decision kind."""
    return vocab_for(ALL_CARD_EXPANSIONS, encoder)
